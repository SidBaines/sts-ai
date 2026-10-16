"""Independent B1 actor/critic towers, initialized from scratch.

Checkpoint contract: {"model_state": model.state_dict(), "metadata": model.metadata()}.
Architecture lives in metadata["model_config"]. Load rejects encoder/vocabulary drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from sts_policy.entity_encoding import (
    ACTION_TYPES, ENCODER_SHA256, ENCODER_VERSION, ENTITY_TYPES, VOCAB_SIZE, VOCABULARY_SHA256,
)


@dataclass(frozen=True)
class EntityModelConfig:
    width: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    phase: str = 'combat'


class _EntityTower(nn.Module):
    def __init__(self, config: EntityModelConfig):
        super().__init__()
        width = config.width
        self.identity = nn.Embedding(VOCAB_SIZE, width, padding_idx=0)
        self.entity_type = nn.Embedding(len(ENTITY_TYPES) + 1, width, padding_idx=0)
        self.action_type = nn.Embedding(len(ACTION_TYPES) + 1, width, padding_idx=0)
        self.numeric = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.power = nn.Embedding(VOCAB_SIZE, width, padding_idx=0)
        # Nonlinear fusion preserves which amount belongs to which named power,
        # including zero/negative amounts. A sum of separate name and amount
        # embeddings would lose this pairing on owners with multiple powers.
        self.power_value = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, width))
        self.recent = nn.GRU(width, width, batch_first=True)
        layer = nn.TransformerEncoderLayer(width, config.heads, dim_feedforward=width * 2,
                                           dropout=config.dropout, activation='gelu',
                                           batch_first=True, norm_first=False)
        self.attention = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        # TransformerEncoder clones the template's parameters; initialize layers
        # independently instead of beginning with identical layer weights.
        for block in self.attention.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)
        self.pool = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())

    def encode(self, row: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = (self.identity(row['entity_ids']) + self.entity_type(row['entity_types'])
                      + self.identity(row['entity_aux']).sum(-2) + self.numeric(row['entity_numeric']))
        if row['power_ids'].numel():
            amount = row['power_amounts'].unsqueeze(-1)
            available = row['power_amount_available'].unsqueeze(-1).to(amount.dtype)
            powers = self.power_value(torch.cat((self.power(row['power_ids']), amount, available), -1))
            # One row at a time, public owner indices only. No cross-enemy pooling.
            additions = torch.zeros_like(embeddings).index_add(0, row['power_owner'], powers)
            embeddings = embeddings + additions
        states = self.attention(embeddings.unsqueeze(0)).squeeze(0)
        pooled = states.mean(0)
        recent_context = torch.zeros_like(pooled)
        if row['recent_ids'].numel():
            recent = (self.identity(row['recent_ids']) + self.action_type(row['recent_types'])
                      + self.numeric(row['recent_numeric']))
            _, hidden = self.recent(recent.unsqueeze(0))
            recent_context = hidden[-1, 0]
        return states, self.pool(torch.cat((pooled, recent_context), -1))


class _Actor(nn.Module):
    def __init__(self, config: EntityModelConfig):
        super().__init__()
        w = config.width
        self.tower = _EntityTower(config)
        self.action_numeric = nn.Sequential(nn.Linear(32, w), nn.SiLU(), nn.Linear(w, w))
        self.text = nn.Linear(32, w, bias=False)
        self.score = nn.Sequential(nn.Linear(w * 4, w), nn.SiLU(), nn.Linear(w, 1))

    def forward(self, row: dict[str, torch.Tensor]) -> torch.Tensor:
        entities, state = self.tower.encode(row)
        action = (self.tower.identity(row['action_ids']) + self.tower.action_type(row['action_types'])
                  + self.tower.identity(row['action_aux']).sum(-2)
                  + self.action_numeric(row['action_numeric']) + self.text(row['action_text']))
        # Source membership includes the single source, or every selected hand
        # entity for a multi-select. Mean + explicit count avoids size-only scale.
        members = row['action_source_members']
        mask = row['action_source_mask'].unsqueeze(-1)
        source = (entities[members.clamp_min(0)] * mask).sum(1) / mask.sum(1).clamp_min(1)
        targets = row['action_target']
        target = entities[targets.clamp_min(0)] * (targets >= 0).unsqueeze(-1)
        features = torch.cat((action, source, target, state.expand_as(action)), -1)
        return self.score(features).squeeze(-1)


class _Critic(nn.Module):
    def __init__(self, config: EntityModelConfig):
        super().__init__()
        self.tower = _EntityTower(config)
        self.action_numeric = nn.Sequential(nn.Linear(32, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.text = nn.Linear(32, config.width, bias=False)
        self.value = nn.Sequential(nn.Linear(config.width * 2, config.width), nn.SiLU(), nn.Linear(config.width, 1))

    def forward(self, row: dict[str, torch.Tensor]) -> torch.Tensor:
        _, state = self.tower.encode(row)
        # Reward/shop/discovery offers need not already be owned entities. The
        # critic must observe their identities and prices too, using its own
        # embeddings and parameters rather than reading actor activations.
        actions = (self.tower.identity(row['action_ids']) + self.tower.action_type(row['action_types'])
                   + self.tower.identity(row['action_aux']).sum(-2)
                   + self.action_numeric(row['action_numeric']) + self.text(row['action_text']))
        return self.value(torch.cat((state, actions.mean(0)), -1)).squeeze(-1)


class EntityActorCritic(nn.Module):
    """One phase per model; no shared actor/critic or combat/OOC parameters.

    actor_logits unpads each row before arithmetic. This makes its logits
    bit-identical on CPU across different batch partners and padding lengths,
    including duplicate-card ties. Default dropout is zero. Use eval() for policy
    inference if explicitly configuring nonzero dropout. training_logits uses
    masked, vectorized arithmetic for optimization. No global thread changes.
    """
    def __init__(self, width=128, layers=2, heads=4, dropout=0.0, phase='combat'):
        super().__init__()
        if phase not in {'combat', 'ooc'}:
            raise ValueError('phase must be combat or ooc')
        if width < 1 or heads < 1 or width % heads or layers < 1:
            raise ValueError('positive layers/heads and width divisible by heads required')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.config = EntityModelConfig(width, layers, heads, dropout, phase)
        self.actor = _Actor(self.config)
        self.critic = _Critic(self.config)

    @staticmethod
    def _rows(batch: dict[str, torch.Tensor]):
        mask = batch['entity_mask']
        if mask.ndim != 2 or batch['action_mask'].ndim != 2 or len(mask) != len(batch['action_mask']):
            raise ValueError('expected entity/action masks [B,N]/[B,A]')
        for i in range(len(mask)):
            # Collator produces prefix masks. Enforce instead of silently making
            # references wrong for a hand-built, interspersed-mask batch.
            counts = {prefix: int(batch[prefix + '_mask'][i].sum().item())
                      for prefix in ('entity', 'action', 'power', 'recent')}
            if not counts['entity'] or not counts['action']:
                raise ValueError('each model row requires state entities and legal actions')
            row = {}
            for name, value in batch.items():
                prefix = name.split('_')[0]
                if prefix not in counts or name.endswith('_counts'):
                    continue
                row[name] = value[i, :counts[prefix]]
            for prefix, count in counts.items():
                original = batch[prefix + '_mask'][i]
                if not bool(original[:count].all()) or bool(original[count:].any()):
                    raise ValueError('model requires prefix padding masks')
            for name in ('power_owner', 'action_source', 'action_target', 'action_source_members'):
                refs = row[name]
                if bool((refs < -1).any()) or bool((refs >= counts['entity']).any()):
                    raise ValueError(f'out-of-range entity reference in {name}')
            if bool((row['power_owner'] < 0).any()):
                raise ValueError('real powers require public owners')
            # Remove membership padding too, so reduction shapes are independent
            # of the largest multi-select in other observations in the batch.
            member_width = max(1, int(row['action_source_mask'].sum(-1).max().item()))
            row['action_source_members'] = row['action_source_members'][:, :member_width]
            row['action_source_mask'] = row['action_source_mask'][:, :member_width]
            yield row

    def actor_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Actor only: never executes or reads the critic."""
        maximum = batch['action_mask'].shape[1]
        logits = [torch.nn.functional.pad(self.actor(row), (0, maximum - len(row['action_ids'])), value=-torch.inf)
                  for row in self._rows(batch)]
        return torch.stack(logits)

    def critic_values(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Critic-only fitting path with no actor execution or gradients."""
        return torch.stack([self.critic(row) for row in self._rows(batch)])

    def training_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Vectorized actor arithmetic for trusted, CPU-validated collator output.

        Padding is masked throughout. No per-observation device synchronization;
        singleton evaluation deliberately continues to use actor_logits.
        """
        b, tower = batch, self.actor.tower
        entities = (tower.identity(b['entity_ids']) + tower.entity_type(b['entity_types'])
                    + tower.identity(b['entity_aux']).sum(-2) + tower.numeric(b['entity_numeric']))
        size, count, width = entities.shape
        powers = tower.power_value(torch.cat((tower.power(b['power_ids']),
            b['power_amounts'].unsqueeze(-1), b['power_amount_available'].unsqueeze(-1).to(entities.dtype)), -1))
        owners = b['power_owner'].clamp_min(0) + torch.arange(size, device=entities.device)[:, None] * count
        additions = torch.zeros_like(entities).reshape(-1, width).index_add(
            0, owners.reshape(-1), (powers * b['power_mask'].unsqueeze(-1)).reshape(-1, width))
        entities = tower.attention(entities + additions.reshape_as(entities),
                                   src_key_padding_mask=~b['entity_mask'])
        mask = b['entity_mask'].unsqueeze(-1)
        pooled = (entities * mask).sum(1) / mask.sum(1).clamp_min(1)
        recent = (tower.identity(b['recent_ids']) + tower.action_type(b['recent_types'])
                  + tower.numeric(b['recent_numeric']))
        history, _ = tower.recent(recent)
        lengths = b['recent_mask'].sum(1)
        context = history[torch.arange(size, device=entities.device), (lengths - 1).clamp_min(0)]
        context = context * (lengths > 0).unsqueeze(-1)
        state = tower.pool(torch.cat((pooled, context), -1))
        action = (tower.identity(b['action_ids']) + tower.action_type(b['action_types'])
                  + tower.identity(b['action_aux']).sum(-2) + self.actor.action_numeric(b['action_numeric'])
                  + self.actor.text(b['action_text']))
        rows = torch.arange(size, device=entities.device)[:, None, None]
        members = entities[rows, b['action_source_members'].clamp_min(0)]
        member_mask = b['action_source_mask'].unsqueeze(-1)
        source = (members * member_mask).sum(2) / member_mask.sum(2).clamp_min(1)
        target = entities[rows.squeeze(-1), b['action_target'].clamp_min(0)]
        target = target * (b['action_target'] >= 0).unsqueeze(-1)
        features = torch.cat((action, source, target, state[:, None].expand_as(action)), -1)
        return self.actor.score(features).squeeze(-1).masked_fill(~b['action_mask'], -torch.inf)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_logits(batch), self.critic_values(batch)

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        return self.actor.parameters()

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        return self.critic.parameters()

    def metadata(self) -> dict:
        return {'kind': 'entity_fresh_v1', 'encoder_version': ENCODER_VERSION,
                'encoder_sha256': ENCODER_SHA256,
                'vocabulary_sha256': VOCABULARY_SHA256, 'vocabulary_size': VOCAB_SIZE,
                'model_config': asdict(self.config), 'numeric_dim': 32, 'action_text_dim': 32,
                'parameter_count': sum(p.numel() for p in self.parameters())}


def policy_from_payload(payload: dict) -> EntityActorCritic:
    """Load only B1 payloads; no legacy-policy conversion or prior weights."""
    metadata = payload['metadata']
    expected = {'kind': 'entity_fresh_v1', 'encoder_version': ENCODER_VERSION,
                'encoder_sha256': ENCODER_SHA256,
                'vocabulary_sha256': VOCABULARY_SHA256, 'vocabulary_size': VOCAB_SIZE}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'incompatible entity checkpoint {key}: {metadata.get(key)!r}')
    config = metadata['model_config']
    model = EntityActorCritic(**config)
    model.load_state_dict(payload['model_state'], strict=True)
    return model
