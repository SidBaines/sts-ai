"""Fight-panel teacher: choose among OOC options by scoring their afterstates.

Shared by the overlay evaluation, label collection, and (later) live use.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from sts_policy.curriculum import TaskSpec

from .panel import PanelWorker


def is_card_choice(description: str) -> bool:
    return description.startswith("take card ") or description.startswith("skip card")


def is_rest_choice(description: str) -> bool:
    return description.startswith("rest") or description.startswith("smith")


DECISION_TYPES: dict[str, Callable[[str], bool]] = {"card": is_card_choice, "rest": is_rest_choice}


def teacher_fights(encounters: dict[str, Any], boss_samples: int = 4, elite_samples: int = 2,
                   hallway: int = 2) -> list[tuple[str, str, int]]:
    fights = [(encounters["boss"], "BOSS", k) for k in range(boss_samples)]
    for elite in encounters["elites"][:2]:
        fights += [(elite, "ELITE", k) for k in range(elite_samples)]
    fights += [(e, "MONSTER", 0) for e in encounters["hallway"][:hallway]]
    return fights


def score_fights(fights: Sequence[dict[str, Any]]) -> float | None:
    """Scalar deck/HP strength: boss win dominates, elites next, HP as tie-break."""
    valid = [f for f in fights if "fault" not in f and not f.get("ub")]
    if not valid:
        return None

    def mean(room: str, key: str) -> float:
        rows = [f for f in valid if f["room"] == room]
        if not rows:
            return 0.0
        if key == "win":
            return sum(f["win"] for f in rows) / len(rows)
        return sum(f["hp_after"] / max(1, f["max_hp"]) for f in rows) / len(rows)

    return (mean("BOSS", "win") + 0.5 * mean("ELITE", "win") + 0.25 * mean("BOSS", "hp")
            + 0.25 * mean("ELITE", "hp") + 0.1 * mean("MONSTER", "hp"))


def expand_rest_options(panel: PanelWorker, spec: TaskSpec, legal: Sequence[dict[str, Any]]) -> list[list[int]]:
    """rest -> [rest]; smith -> [smith, select card i] for each distinct upgradable card."""
    options: list[list[int]] = []
    for action in legal:
        description = action["description"]
        if description.startswith("rest"):
            options.append([int(action["index"])])
        elif description.startswith("smith"):
            follow = panel.menu(TaskSpec(seed=spec.seed, kind="run", prefix=tuple(spec.prefix) + (int(action["index"]),),
                                         max_act=spec.max_act, horizon=spec.horizon, seek_combat=False))
            seen = set()
            for choice in follow:
                label = choice["description"].split(":", 1)[-1].strip()
                if label in seen:
                    continue
                seen.add(label)
                options.append([int(action["index"]), int(choice["index"])])
    return options


BOSS_FLOORS = (16, 33, 50)


def teach(panel: PanelWorker, spec: TaskSpec, options: Sequence[Sequence[int]], checkpoint: str, *,
          boss_samples: int = 4, elite_samples: int = 2, hallway: int = 2,
          full_hp_boss: bool | None = None) -> list[dict[str, Any]]:
    """Score each option sequence; returns [{option, score, fights}] in input order.

    The boss is fought at full HP by default (a rest usually precedes it), except
    within two floors of the boss, where the HP actually carried into it is used.
    """
    encounters = panel.encounters(spec)
    if full_hp_boss is None:
        floor = int(encounters["floor"])
        full_hp_boss = not any(0 <= b - floor <= 2 for b in BOSS_FLOORS)
    fights = teacher_fights(encounters, boss_samples, elite_samples, hallway)
    scored = panel.afterstates(spec, options, fights, checkpoint, False, ["BOSS"] if full_hp_boss else [])
    return [{"option": list(o), "score": score_fights(r.get("fights", [])) if "fault" not in r else None,
             "fault": r.get("fault")} for o, r in zip(options, scored)]
