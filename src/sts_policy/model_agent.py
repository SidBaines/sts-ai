"""CLI plugin and batched tensor plumbing for trained non-LLM policies."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .encoding import ENCODER_VERSION, collate_observations, encode_observation
from .model import CandidateActorCritic
from .training import load_checkpoint
from .losses import masked_distribution


def collate_public(observations: Sequence[dict[str, Any]], device: str = "cpu"):
    return collate_observations([encode_observation(observation) for observation in observations], device=device)


def cached_public_collator(observations: Sequence[dict[str, Any]], *, tensor_cache: bool = False):
    """Preencode a fixed immutable BC dataset once, retaining objects against ID reuse.

    Rollout inference should use collate_public: learner observations continuously
    change, and an unbounded rollout cache would waste memory.
    """
    originals = tuple(observations)
    encoded = {id(obs): encode_observation(obs) for obs in originals}
    groups = {}
    locations = {}
    tensors = {}
    if tensor_cache:
        for identity, observation in encoded.items():
            count = len(observation.action_indices)
            bucket = (max(1, count) - 1).bit_length()
            group = groups.setdefault(bucket, [])
            locations[identity] = (bucket, len(group), count)
            group.append(observation)
        tensors = {bucket: collate_observations(group, device="cpu") for bucket, group in groups.items()}

    def collate(batch: Sequence[dict[str, Any]], device: str = "cpu"):
        # Closure deliberately holds originals, ensuring cache IDs cannot be reused.
        _ = originals
        if tensor_cache and batch and all(id(obs) in locations for obs in batch):
            rows = [locations[id(obs)] for obs in batch]
            max_actions = max(1, max(row[2] for row in rows))
            batch_groups = {}
            for output_index, (bucket, row, _) in enumerate(rows):
                batch_groups.setdefault(bucket, []).append((output_index, row))
            if len(batch_groups) == 1:
                bucket, selected = next(iter(batch_groups.items()))
                indices = torch.tensor([row for _, row in selected], dtype=torch.long)
                state, actions, mask = [torch.index_select(value, 0, indices) for value in tensors[bucket]]
                return state.to(device), actions[:, :max_actions].to(device), mask[:, :max_actions].to(device)
            from .encoding import STATE_DIM, ACTION_DIM
            state = torch.empty(len(batch), STATE_DIM)
            actions = torch.zeros(len(batch), max_actions, ACTION_DIM)
            mask = torch.zeros(len(batch), max_actions, dtype=torch.bool)
            for bucket, selected in batch_groups.items():
                output_indices = torch.tensor([index for index, _ in selected], dtype=torch.long)
                source_indices = torch.tensor([row for _, row in selected], dtype=torch.long)
                source_state, source_actions, source_mask = tensors[bucket]
                width = min(max_actions, source_actions.shape[1])
                state[output_indices] = source_state[source_indices]
                actions[output_indices, :width] = source_actions[source_indices, :width]
                mask[output_indices, :width] = source_mask[source_indices, :width]
            return state.to(device), actions.to(device), mask.to(device)
        return collate_observations([encoded[id(obs)] if id(obs) in encoded else encode_observation(obs)
                                     for obs in batch], device=device)
    return collate


def forward_policy(model: torch.nn.Module, batch: Any):
    logits, values = model(*batch)
    return logits, values, batch[2]


class ModelAgent:
    def __init__(self, checkpoint: str, device: str = "cpu", deterministic: bool = True,
                 seed: int = 0) -> None:
        payload = load_checkpoint(checkpoint)
        version = payload["metadata"].get("encoder_version")
        if version != ENCODER_VERSION:
            raise ValueError(f"checkpoint encoder {version!r} differs from {ENCODER_VERSION!r}")
        self.model = CandidateActorCritic(**payload["model_config"])
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(device).eval()
        self.device, self.deterministic = device, deterministic
        self.reset(seed)

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.device == "mps":
            torch.mps.manual_seed(seed)

    @torch.no_grad()
    def choose_actions(self, observations: Sequence[dict[str, Any]]) -> list[int]:
        logits, _, mask = forward_policy(self.model, collate_public(observations, self.device))
        positions = (logits.argmax(-1) if self.deterministic
                     else masked_distribution(logits, mask).sample())
        return [int(obs["legal_actions"][position]["index"])
                for obs, position in zip(observations, positions.cpu().tolist())]

    def choose_action(self, observation: dict[str, Any]) -> int:
        return self.choose_actions([observation])[0]


class CompositeAgent:
    """Compose independently trained policies using the public decision phase."""
    def __init__(self, combat_checkpoint: str, ooc_checkpoint: str, device: str = "cpu",
                 deterministic: bool = True, seed: int = 0) -> None:
        self.combat = ModelAgent(combat_checkpoint, device, deterministic, seed)
        self.ooc = ModelAgent(ooc_checkpoint, device, deterministic, seed)

    def reset(self, seed: int) -> None:
        self.combat.reset(seed)
        self.ooc.reset(seed)

    def choose_actions(self, observations: Sequence[dict[str, Any]]) -> list[int]:
        results = [0] * len(observations)
        for combat, agent in ((True, self.combat), (False, self.ooc)):
            indices = [i for i, obs in enumerate(observations) if (obs.get("phase") == "combat") == combat]
            if indices:
                actions = agent.choose_actions([observations[i] for i in indices])
                for index, action in zip(indices, actions):
                    results[index] = action
        return results

    def choose_action(self, observation: dict[str, Any]) -> int:
        return self.choose_actions([observation])[0]


class CompositeActorCritic(torch.nn.Module):
    """Vectorized public-phase dispatch for evaluating two specialized policies."""
    def __init__(self, combat: torch.nn.Module, ooc: torch.nn.Module) -> None:
        super().__init__()
        self.combat, self.ooc = combat, ooc

    def forward(self, state, actions, mask):
        combat_logits, combat_values = self.combat(state, actions, mask)
        ooc_logits, ooc_values = self.ooc(state, actions, mask)
        # public_candidate_v2 numeric feature 7 is the documented combat flag.
        phase = state[:, 7] > .5
        return (torch.where(phase[:, None], combat_logits, ooc_logits),
                torch.where(phase, combat_values, ooc_values))


def make_agent(checkpoint: str | None = None, device: str = "cpu", deterministic: bool = True,
               seed: int = 0, combat_checkpoint: str | None = None,
               ooc_checkpoint: str | None = None) -> ModelAgent | CompositeAgent:
    if combat_checkpoint is not None or ooc_checkpoint is not None:
        if checkpoint is not None or combat_checkpoint is None or ooc_checkpoint is None:
            raise ValueError("supply either checkpoint or both combat_checkpoint and ooc_checkpoint")
        return CompositeAgent(combat_checkpoint, ooc_checkpoint, device, deterministic, seed)
    if checkpoint is None:
        raise ValueError("a policy checkpoint is required")
    return ModelAgent(checkpoint, device, deterministic, seed)
