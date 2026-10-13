"""Earlier strategic starts from exact executed learned-policy histories."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from .curriculum import TaskSpec, public_hash, world_split
from .workers import CurriculumWorker
from sts_env.env import SimulatorError

FRONTIER_VERSION = "learned_ooc_early_menu_frontiers_v2"
PROVENANCE_KEYS = ("environment_version", "observation_version", "python_interface_sha256",
                   "runtime_python_version", "native_provenance")


def validate_source(task: TaskSpec) -> None:
    if task.kind != "run" or task.prefix or task.max_act != 3 or world_split(task.seed) != "train":
        raise ValueError("early frontier sources must be fresh TRAIN runs through Act 3")


def eligible_source(observation: dict[str, Any]) -> bool:
    state = observation["state"]
    late_act1_relic = (state["act"]==1 and state.get("floor",0)>=16
                      and str(state.get("screen_state","")).endswith("BOSS_RELIC_REWARDS"))
    return observation["phase"]=="out_of_combat" and len(observation["legal_actions"])>1 and (
        state["act"] in (2,3) or late_act1_relic)


def menu_stage(observation: dict[str, Any]) -> tuple[int, str, tuple[str, ...], tuple[str, ...]]:
    state = observation["state"]
    kinds = tuple(sorted({str(action.get("public",{}).get("kind","unknown")) for action in observation["legal_actions"]}))
    selections = tuple(sorted({str(action["public"]["selection_type"]) for action in observation["legal_actions"]
                               if "selection_type" in action.get("public",{})}))
    return int(state["act"]),str(state["screen_state"]),kinds,selections


def frontier_before_action(task: TaskSpec, observation: dict[str, Any], info: dict[str, Any],
                           selected_index: int) -> dict[str, Any] | None:
    validate_source(task)
    state = observation["state"]
    if not eligible_source(observation):
        return None
    if selected_index not in {int(action["index"]) for action in observation["legal_actions"]}:
        raise ValueError("recorded source choice is outside the current menu")
    prefix, previews = tuple(info["action_prefix"]), tuple(info["ooc_previews"])
    if len(prefix) != info["task_steps"] or len(previews) != len(prefix) or info["world_seed"] != task.seed:
        raise ValueError("source prefix must contain every actual preceding decision")
    provenance = {key: deepcopy(info[key]) for key in PROVENANCE_KEYS}
    start = TaskSpec(seed=task.seed, kind="run", max_act=3, horizon=4096, ascension=task.ascension,
        battle_simulations=task.battle_simulations, prefix=prefix, ooc_previews=previews,
        expected_public_hash=public_hash(observation), expected_provenance=provenance, seek_combat=False)
    _,screen,kinds,selections = menu_stage(observation)
    family = ("card_only_rewards" if "take_card" in kinds and not {"take_gold","take_potion","take_relic"}.intersection(kinds)
              else "mixed_card_rewards" if "take_card" in kinds else "other_menu") if screen.endswith("REWARDS") else "other_menu"
    return {**start.to_dict(), "frontier_version":FRONTIER_VERSION, "start_act":state["act"],
        "start_screen":state["screen_state"], "floor":state["floor"],
        "start_menu_kinds":list(kinds), "start_selection_types":list(selections), "decision_family":family,
        "frontier_step":info["task_steps"], "start_legal_count":len(observation["legal_actions"]),
        "source_executed_index":selected_index, "source_observation":deepcopy(observation),
        "selection":"first public menu stage in Acts 2/3, plus late Act 1 boss relic"}


class EarlyFrontierWorker(CurriculumWorker):
    """Record in the parent before stepping, never query an OOC teacher."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frontiers: list[dict[str, Any]] = []
        self.seen: set[tuple[int, str, tuple[str, ...], tuple[str, ...]]] = set()
        self.source_task: TaskSpec | None = None

    def reset(self, spec):
        task = TaskSpec.from_dict(spec) if isinstance(spec, dict) else spec
        validate_source(task)
        result = super().reset(task)
        self.source_task, self.seen = task, set()
        return result

    def step(self, action):
        if self.source_task is None or self.observation is None or self.info is None:
            raise RuntimeError("reset before recording source actions")
        if eligible_source(self.observation):
            key = menu_stage(self.observation)
            if key not in self.seen:
                try:
                    row = frontier_before_action(self.source_task, self.observation, self.info, action)
                except (KeyError, ValueError) as error:
                    raise SimulatorError(f"Cannot record a trustworthy frontier: {error}") from error
                self.frontiers.append(row)
                self.seen.add(key)
        return super().step(action)


def completed_frontiers(records: list[dict[str, Any]], collection: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    faulted = {row["world_seed"] for row in collection["faults"]}
    completed = {row["world_seed"] for row in collection["episodes"]
                 if row["terminated"] and not row["truncated"]} - faulted
    usable, quarantined = [], []
    for row in records:
        if row["seed"] in completed:
            usable.append(row)
        else:
            quarantined.append({**row, "quarantine_reason":"source world faulted" if row["seed"] in faulted
                                else "source world did not complete"})
    return usable, quarantined


def coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"frontiers":len(records), "independent_source_worlds":len({row["seed"] for row in records}),
        "frontiers_by_act":dict(Counter(str(row["start_act"]) for row in records)),
        "source_worlds_by_act":{str(act):len({row["seed"] for row in records if row["start_act"]==act}) for act in (1,2,3)},
        "screens_by_act":dict(Counter(f"act{row['start_act']}:{row['start_screen']}" for row in records)),
        "decision_families":dict(Counter(row.get("decision_family","historical screen selection") for row in records)),
        "note":"Different screens/acts from one world are correlated; frontier count is not independent world count."}
