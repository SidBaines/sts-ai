"""Public strategic frontiers from recorded native TRAIN trajectories."""
from __future__ import annotations

from typing import Any

from .curriculum import TaskSpec, public_hash, world_split


def strategic_frontier(world: dict[str, Any], act: int) -> dict[str, Any] | None:
    """Prefer the last campfire choice, otherwise the latest real OOC choice.

    A final-boss combat start can produce wins without any trainable strategic
    decisions. This frontier retains at least the selected OOC choice before
    that continuation. Native preview markers reconstruct the recorded prefix;
    the new continuation then uses learned public control without native play.
    """
    seed = int(world["seed"])
    if world_split(seed) != "train" or act not in (2, 3):
        raise ValueError("strategic frontiers require TRAIN Act 2/3 worlds")
    prefix, previews, choices, rests = [], [], [], []
    for row in world["records"]:
        if int(row["step"]) != len(prefix):
            raise ValueError("recorded decisions must be contiguous for exact prefix reconstruction")
        obs = row["observation"]
        state = obs["state"]
        if obs["phase"] == "out_of_combat" and state["act"] == act and len(obs["legal_actions"]) > 1:
            spec = TaskSpec(seed=seed, kind="run", prefix=tuple(prefix), ooc_previews=tuple(previews),
                expected_public_hash=public_hash(obs), expected_provenance=world["provenance"],
                horizon=4096, max_act=3, battle_simulations=int(world["settings"]["simulations"]), seek_combat=False)
            candidate = {**spec.to_dict(), "floor": state["floor"], "start_act": act,
                "start_screen": state["screen_state"], "start_legal_count": len(obs["legal_actions"]),
                "frontier_step": row["step"], "source_policy": "native search combat / scripted OOC",
                "selection": "last campfire choice, otherwise latest OOC choice"}
            choices.append(candidate)
            if str(state["screen_state"]).endswith("REST_ROOM"):
                rests.append(candidate)
        if "executed_index" not in row:
            raise ValueError("executed action identity is required; teacher labels cannot replace it")
        prefix.append(int(row["executed_index"]))
        previews.append(obs["phase"] == "out_of_combat")
    selected = rests or choices
    return selected[-1] if selected else None
