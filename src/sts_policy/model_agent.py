"""CLI plugin and batched tensor plumbing for trained non-LLM policies."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .encoding import ENCODER_VERSION, EncodedObservation, collate_observations, encode_observation
from .model import CandidateActorCritic
from .training import load_checkpoint
from .losses import masked_distribution


def collate_public_with_encodings(observations: Sequence[dict[str, Any]], device: str = "cpu"):
    """Return the exact public encodings used to assemble this inference batch."""
    encoded = [encode_observation(observation) for observation in observations]
    return collate_observations(encoded, device=device), encoded


def collate_public(observations: Sequence[dict[str, Any]], device: str = "cpu"):
    return collate_public_with_encodings(observations, device)[0]


def cached_public_collator(observations: Sequence[dict[str, Any]], *, tensor_cache: bool = False,
                           preencoded: Sequence[EncodedObservation | None] | None = None):
    """Preencode a fixed immutable BC dataset once, retaining objects against ID reuse.

    Optional aligned encodings reuse the exact arrays retained from sampling.
    Originals remain strongly referenced, preventing object-ID reuse. This cache
    is local to an immutable dataset/buffer, never a global rollout cache.
    """
    originals = tuple(observations)
    if preencoded is not None and len(preencoded) != len(originals):
        raise ValueError("preencoded observations must align with the immutable buffer")
    supplied = preencoded if preencoded is not None else [None] * len(originals)
    encoded = {}
    for obs, cached in zip(originals, supplied):
        if cached is not None and cached.action_indices.tolist() != [int(action["index"]) for action in obs["legal_actions"]]:
            raise ValueError("cached encoding legal menu identity differs from observation")
        encoded[id(obs)] = cached if cached is not None else encode_observation(obs)
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
        self.model = policy_from_payload(payload)
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

    def metadata(self) -> dict[str, Any]:
        if not isinstance(self.combat, CandidateActorCritic) or not isinstance(self.ooc, CandidateActorCritic):
            raise ValueError("serializable composite heads must be CandidateActorCritic policies")
        return {"kind": "composite", "combat": self.combat.metadata(), "ooc": self.ooc.metadata()}

    def forward(self, state, actions, mask):
        # public_candidate_v2 numeric feature 7 is the documented combat flag.
        phase = state[:, 7] > .5
        combat_rows = phase.nonzero(as_tuple=True)[0]
        if combat_rows.numel() == len(state) or combat_rows.numel() == 0:
            active, inactive = ((self.combat, self.ooc) if combat_rows.numel() == len(state)
                                else (self.ooc, self.combat))
            logits, values = active(state, actions, mask)
            if torch.is_grad_enabled():
                # Preserve the former zero gradients of inactive parameters;
                # None would change optimizers' weight-decay/state semantics.
                zero = values.new_zeros(())
                for parameter in inactive.parameters():
                    if parameter.requires_grad and parameter.numel():
                        zero = zero + parameter.reshape(-1)[0] * 0.
                logits, values = logits + zero, values + zero
            return logits, values
        ooc_rows = (~phase).nonzero(as_tuple=True)[0]
        combat_logits, combat_values = self.combat(
            state.index_select(0, combat_rows), actions.index_select(0, combat_rows),
            mask.index_select(0, combat_rows))
        ooc_logits, ooc_values = self.ooc(
            state.index_select(0, ooc_rows), actions.index_select(0, ooc_rows),
            mask.index_select(0, ooc_rows))
        logits = combat_logits.new_empty((len(state), actions.shape[1]))
        values = combat_values.new_empty((len(state),))
        logits = logits.index_copy(0, combat_rows, combat_logits).index_copy(0, ooc_rows, ooc_logits)
        values = values.index_copy(0, combat_rows, combat_values).index_copy(0, ooc_rows, ooc_values)
        return logits, values


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


def policy_from_config(config: dict[str, Any]) -> torch.nn.Module:
    """Discriminated policy configuration; absent kind preserves legacy candidates."""
    if not isinstance(config, dict):
        raise ValueError("policy model_config must be a dictionary")
    options = dict(config)
    kind = options.pop("kind", "candidate")
    if kind == "candidate":
        from .encoding import STATE_DIM, ACTION_DIM
        if set(options) - {"size", "context", "state_dim", "action_dim"}:
            raise ValueError("unknown candidate model configuration fields")
        if options.get("state_dim", STATE_DIM) != STATE_DIM or options.get("action_dim", ACTION_DIM) != ACTION_DIM:
            raise ValueError("policy feature dimensions differ from the current public encoder")
        return CandidateActorCritic(**options)
    if kind == "composite":
        if set(options) != {"combat", "ooc"}:
            raise ValueError("composite config requires exactly combat and ooc heads")
        combat = policy_from_config(options["combat"])
        ooc = policy_from_config(options["ooc"])
        if not isinstance(combat, CandidateActorCritic) or not isinstance(ooc, CandidateActorCritic):
            raise ValueError("nested composite policy heads are unsupported")
        return CompositeActorCritic(combat, ooc)
    raise ValueError(f"unsupported policy model kind: {kind!r}")


def policy_from_payload(payload: dict[str, Any]) -> torch.nn.Module:
    """Validate the public encoder and exact model structure before loading weights."""
    from .training import CHECKPOINT_VERSION
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported policy checkpoint version")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("encoder_version") != ENCODER_VERSION:
        raise ValueError("checkpoint encoder version mismatch")
    config = payload.get("model_config")
    model = policy_from_config(config)
    kind = config.get("kind", "candidate")
    if metadata.get("model_kind", kind) != kind:
        raise ValueError("checkpoint metadata/model-config kind mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    return model
