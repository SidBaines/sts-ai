"""Experiment task semantics, entirely outside the immutable environment base."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

TASK_VERSION = "sts_curriculum_v1"
REWARD_VERSION = "sts_curriculum_reward_v2"


def validate_provenance(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Reject replay across native builds or Python public interfaces."""
    keys = ("python_interface_sha256", "runtime_python_version", "environment_version",
            "observation_version", "native_provenance")
    for key in keys:
        if key not in expected or key not in actual or expected[key] != actual[key]:
            raise ValueError(f"Replay provenance mismatch: {key}")


def public_hash(observation: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def world_split(seed: int) -> str:
    """Stable world-level split; every fight/state from a world shares its split."""
    bucket = int(hashlib.sha256(f"sts-curriculum-world-v1:{seed}".encode()).hexdigest()[:8], 16) % 10
    return "final" if bucket == 0 else "dev" if bucket == 1 else "train"


@dataclass(frozen=True)
class TaskSpec:
    seed: int
    kind: str = "battle"
    prefix: tuple[int, ...] = ()
    expected_public_hash: str | None = None
    ooc_previews: tuple[bool, ...] = ()
    horizon: int = 256
    max_act: int = 1
    battle_simulations: int = 50
    ascension: int = 0
    seek_combat: bool = True
    expected_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"battle", "ooc", "run"}:
            raise ValueError("kind must be battle, ooc or run")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**63:
            raise ValueError("invalid world seed")
        if self.horizon < 1 or self.battle_simulations < 1 or not 1 <= self.max_act <= 3:
            raise ValueError("invalid task bounds")
        if self.ooc_previews and len(self.ooc_previews) != len(self.prefix):
            raise ValueError("ooc_previews must match prefix length")
        if any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in self.prefix):
            raise ValueError("prefix must contain nonnegative legal menu indices")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "task_version": TASK_VERSION, "split": world_split(self.seed)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskSpec:
        keys = cls.__dataclass_fields__
        data = {k: v for k, v in value.items() if k in keys}
        data["prefix"] = tuple(data.get("prefix", ()))
        data["ooc_previews"] = tuple(data.get("ooc_previews", ()))
        if data.get("expected_provenance") is None and "provenance" in value:
            data["expected_provenance"] = value["provenance"]
        return cls(**data)


def player_hp(observation: dict[str, Any]) -> float:
    state = observation["state"]
    return float(state.get("combat", {}).get("player_cur_hp", state.get("cur_hp", 0)))


def potential(observation: dict[str, Any]) -> float:
    """Bounded public HP/enemy-progress potential; block deliberately excluded.

    Block disappears each turn; naive positive block rewards encourage cycling.
    Potential shaping must use gamma*Phi(next)-Phi(previous), zeroing terminal Phi.
    """
    state = observation["state"]
    maximum = max(1.0, float(state.get("max_hp", 80)))
    combat = state.get("combat", {})
    enemies = combat.get("enemies", [])
    alive_hp = sum(max(0.0, float(e.get("hp", e.get("cur_hp", 0)))) for e in enemies)
    enemy_max = sum(max(1.0, float(e.get("max_hp", e.get("hp", 1)))) for e in enemies)
    progress = 1.0 - alive_hp / max(1.0, enemy_max) if enemies else 0.0
    return 0.25 * player_hp(observation) / maximum + 0.10 * progress


def reward_components(previous: dict[str, Any], observation: dict[str, Any], *,
                      task_outcome: str, terminated: bool, mode: str = "potential",
                      gamma: float = 1.0, progress_floor_weight: float = 0.05,
                      terminal_hp_weight: float = 0.0) -> dict[str, float]:
    """Logged external curriculum rewards; horizon cutoffs are not losses."""
    if mode not in {"potential", "progress", "sparse"}:
        raise ValueError("reward_mode must be potential, progress or sparse")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("invalid discount")
    if not math.isfinite(progress_floor_weight) or not math.isfinite(terminal_hp_weight):
        raise ValueError("reward weights must be finite")
    success = task_outcome in {"battle_victory", "victory", "act_clear"}
    failure = task_outcome in {"battle_defeat", "defeat"}
    terminal = 1.0 if success else -1.0 if failure else 0.0
    shaping = gamma * (0.0 if terminated else potential(observation)) - potential(previous) if mode == "potential" else 0.0
    # Floor progression is a curriculum hypothesis, not policy invariant shaping.
    # The engine's floor counter is run-global; if a platform resets it at act
    # boundaries, awarding the terminal act-clear success still gives the goal.
    progress = progress_floor_weight * max(0, int(observation["state"].get("floor", 0)) - int(previous["state"].get("floor", 0))) if mode == "progress" else 0.0
    hp_bonus = terminal_hp_weight * player_hp(observation) / max(1.0, float(observation["state"].get("max_hp", 80))) if success and terminated else 0.0
    return {"terminal": terminal, "potential": shaping, "progress": progress,
            "terminal_hp": hp_bonus}


def task_reward(previous: dict[str, Any], observation: dict[str, Any], *, kind: str,
                task_outcome: str, terminated: bool, gamma: float = 1.0,
                mode: str = "potential", progress_floor_weight: float = 0.05,
                terminal_hp_weight: float = 0.0) -> float:
    return sum(reward_components(previous, observation, task_outcome=task_outcome,
                                 terminated=terminated, mode=mode, gamma=gamma,
                                 progress_floor_weight=progress_floor_weight,
                                 terminal_hp_weight=terminal_hp_weight).values())
