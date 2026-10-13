"""Optional privileged controller retaining a validated winning sequence.

This differs from per-action teacher replanning. Cached plan contents stay in
the isolated native process; teacher_action returns a current public index.
Step info retains the existing privileged diagnostic contract, with additional
scalar counters. This is never a public policy or a PPO controller.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .workers import CurriculumEpisode

CONTROLLER_VERSION = "privileged_winning_plan_reuse_v2"


class NativePlanEpisode(CurriculumEpisode):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._clear_plan()
        self.stats = {}

    def _clear_plan(self) -> None:
        self.plan = deque()
        self.pending_index = None
        self.plan_settings = None
        self.known_winning_plan = False

    def reset(self, spec):
        self._clear_plan()
        self.stats = {key:0 for key in ("search_queries","winning_plans","cache_hits","invalidations",
                                      "exhausted_in_combat","winning_plan_defeats")}
        return super().reset(spec)

    def _info(self, info, outcome):
        return {**super()._info(info,outcome), "native_plan_stats":{
            "controller_version":CONTROLLER_VERSION, **self.stats}}

    def _cached_index(self):
        row = self.plan[0]
        try:
            index = int(self.episode.env.display_index_for_bits(row["bits"]))
        except ValueError:
            return None
        # Public wrappers add computed damage and selection labels. Compare the
        # same native description format used by search_best_action, rather than
        # rejecting a valid plan because presentation changed.
        battle = self.episode.env.bc
        native = next((action for action in battle.legal_actions(include_card_type=True)
                       if int(action.bits)==row["bits"]),None)
        description = str(native.describe(battle,include_card_type=True)) if native is not None else None
        turn = int(self.observation["state"]["combat"]["turn"])
        if description!=row["description"] or turn!=row["turn"]:
            return None
        return index

    def teacher_action(self, simulations=50, search_seed=None):
        if self.finished or simulations < 1:
            raise RuntimeError("positive search budget and an active task required")
        if self.observation["phase"] != "combat":
            self._clear_plan()
            return super().teacher_action(simulations,search_seed)
        settings = (simulations,self.spec.seed if search_seed is None else search_seed)
        if self.plan_settings is not None and self.plan_settings!=settings:
            self.stats["invalidations"] += 1
            self._clear_plan()
        # Repeated queries without an intervening action must not consume a plan.
        if self.pending_index is not None:
            return self.pending_index
        if self.plan:
            index = self._cached_index()
            if index is not None:
                self.plan.popleft()
                self.stats["cache_hits"] += 1
                self.pending_index = index
                return index
            self.stats["invalidations"] += 1
            self._clear_plan()
        elif self.known_winning_plan:
            self.stats["exhausted_in_combat"] += 1
            self._clear_plan()
        result = self.episode.env.search_best_combat_action(simulations,search_seed=settings[1])
        self.stats["search_queries"] += 1
        bits = int(result["bits"])
        index = int(self.episode.env.display_index_for_bits(bits))
        sequence = result.get("best_sequence",[])
        if (result.get("winning_sequence_found") and result.get("selection_method")=="winning_sequence"
                and sequence and int(sequence[0]["bits"])==bits):
            self.plan = deque({"bits":int(row["bits"]),"turn":int(row["turn"]),
                               "description":str(row["description"])} for row in sequence[1:])
            self.known_winning_plan = True
            self.stats["winning_plans"] += 1
        self.pending_index, self.plan_settings = index, settings
        return index

    def step(self, action):
        was_combat = self.observation["phase"]=="combat"
        old_battle = getattr(self.episode.env,"bc",None) if was_combat else None
        if was_combat and (self.plan or self.known_winning_plan) and action!=self.pending_index:
            self.stats["invalidations"] += 1
            self._clear_plan()
        result = super().step(action)
        self.pending_index = None
        if was_combat and (result[0]["phase"]!="combat" or result[2]):
            if self.known_winning_plan and old_battle is not None and str(old_battle.outcome).endswith("PLAYER_LOSS"):
                self.stats["winning_plan_defeats"] += 1
            self._clear_plan()
        return (*result[:4],{**result[4],"native_plan_stats":{
            "controller_version":CONTROLLER_VERSION, **self.stats}})
