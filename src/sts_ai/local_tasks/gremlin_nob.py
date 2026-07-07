from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sts_ai.local_tasks import base


class GremlinNobTask:
    task_id = "gremlin_nob"
    enemy_name = "GREMLIN_NOB"

    def build_manifest(
        self,
        source_rollout_dir: Path,
        *,
        holdout_mod: int = 4,
        holdout_remainder: int = 0,
    ) -> dict[str, Any]:
        source_rollout_dir = Path(source_rollout_dir)
        windows: list[dict[str, Any]] = []
        for rollout in base.discover_source_rollouts(source_rollout_dir):
            records = base.load_jsonl(rollout.jsonl_path)
            world_seed, rollout_index = base.parse_rollout_stem(rollout.stem)
            for ordinal, (start, end) in enumerate(self._fight_windows(records)):
                window_records = records[start : end + 1]
                metrics = self._metrics_from_source_window(window_records)
                label = self._label(metrics)
                window_id = f"{rollout.stem}_w{ordinal}"
                windows.append(
                    {
                        "task_id": self.task_id,
                        "window_id": window_id,
                        "source_stem": rollout.stem,
                        "world_seed": world_seed,
                        "rollout_index": rollout_index,
                        "ordinal": ordinal,
                        "split": base.split_for_seed(
                            world_seed,
                            holdout_mod=holdout_mod,
                            holdout_remainder=holdout_remainder,
                        ),
                        "start_index": start,
                        "end_index": end,
                        "pre_actions": [
                            record["selected_action"]
                            for record in records[:start]
                            if record.get("action_executed", True)
                            and isinstance(record.get("selected_action"), dict)
                            and record.get("selected_action")
                        ],
                        "label": label,
                        "reward": self.reward_from_metrics(metrics),
                        "rwr_multiplicity": self.rwr_multiplicity(metrics),
                        "metrics": metrics,
                    }
                )

        return {
            "task_id": self.task_id,
            "version": 1,
            "source_rollout_dir": str(source_rollout_dir),
            "split_rule": {
                "kind": "mod_holdout",
                "holdout_mod": holdout_mod,
                "holdout_remainder": holdout_remainder,
            },
            "n_windows": len(windows),
            "split_counts": base.split_counts(windows),
            "label_counts": base.label_counts(windows),
            "windows": windows,
        }

    def completion_reason(self, summary: dict[str, Any]) -> str | None:
        combat = summary.get("combat")
        if not isinstance(combat, dict):
            outcome = str(summary.get("outcome", ""))
            if "LOSS" in outcome or "DEATH" in outcome:
                return "player_loss"
            return "task_complete"
        if int(combat.get("player_cur_hp", summary.get("cur_hp", 0)) or 0) <= 0:
            return "player_loss"
        nob = self._live_nob(combat.get("enemies", []))
        if nob is None:
            return "task_complete"
        return None

    def metrics_from_episode(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
        stopped_reason: str,
        window: dict[str, Any],
    ) -> dict[str, Any]:
        source_metrics = dict(window.get("metrics") or {})
        entry_hp = self._episode_entry_hp(decisions, source_metrics)
        exit_hp = self._episode_exit_hp(decisions, terminal_state)
        completion = self.completion_reason(terminal_state)
        survived = completion == "task_complete" and exit_hp > 0 and stopped_reason == "task_complete"
        hp_loss = max(0, entry_hp - exit_hp)
        counts = Counter(
            classify_action(self._decision_action_description(decision))
            for decision in decisions
        )
        metrics = {
            "entry_hp": entry_hp,
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "survived": survived,
            "label": self._label({"survived": survived, "hp_loss": hp_loss}),
            "reward": self.reward_from_metrics({"survived": survived, "hp_loss": hp_loss}),
            "n_decisions": len(decisions),
            "n_turns": self._episode_n_turns(decisions),
            "action_counts": {kind: int(counts.get(kind, 0)) for kind in ACTION_KINDS},
        }
        return metrics

    def reward_from_metrics(self, metrics: dict[str, Any]) -> float:
        if not bool(metrics.get("survived", False)):
            return -1.0
        hp_loss = max(0, int(metrics.get("hp_loss", 0)))
        return 1.0 - min(hp_loss, 40) / 40.0

    def rwr_multiplicity(
        self,
        metrics: dict[str, Any],
        *,
        max_multiplier: int = 4,
    ) -> int:
        if not bool(metrics.get("survived", False)):
            return 0
        hp_loss = int(metrics.get("hp_loss", 0))
        # Deterministic local-task RWR: strong wins replicate, costly wins still
        # contribute once, losses contribute zero by default.
        import math

        return max(1, min(max_multiplier, round(math.exp((20 - hp_loss) / 10.0))))

    def _fight_windows(self, records: list[dict[str, Any]]) -> list[tuple[int, int]]:
        indices = [
            i
            for i, record in enumerate(records)
            if self._record_has_live_nob(record)
        ]
        if not indices:
            return []
        windows: list[tuple[int, int]] = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            windows.append((start, prev))
            start = prev = idx
        windows.append((start, prev))
        return windows

    def _record_has_live_nob(self, record: dict[str, Any]) -> bool:
        if record.get("phase") != "combat":
            return False
        combat = (record.get("state") or {}).get("combat")
        if not isinstance(combat, dict):
            return False
        return self._live_nob(combat.get("enemies", [])) is not None

    def _metrics_from_source_window(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        first_combat = records[0]["state"]["combat"]
        entry_hp = int(first_combat.get("player_cur_hp", 0) or 0)
        max_hp = int(first_combat.get("player_max_hp", 0) or 0)
        last_after = records[-1].get("after_state") or {}
        exit_hp = self._hp_from_summary(last_after)
        survived = self.completion_reason(last_after) == "task_complete" and exit_hp > 0
        hp_loss = max(0, entry_hp - exit_hp)
        turns = [
            int(((record.get("state") or {}).get("combat") or {}).get("turn", 0) or 0)
            for record in records
        ]
        counts = Counter(
            classify_action(base.selected_action_description(record))
            for record in records
        )
        return {
            "entry_hp": entry_hp,
            "max_hp": max_hp,
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "survived": survived,
            "n_turns": max(turns) if turns else 0,
            "n_decisions": len(records),
            "action_counts": {kind: int(counts.get(kind, 0)) for kind in ACTION_KINDS},
        }

    def _label(self, metrics: dict[str, Any]) -> str:
        if not bool(metrics.get("survived", False)):
            return "loss"
        return "convincing" if int(metrics.get("hp_loss", 0)) <= 20 else "won"

    def _live_nob(self, enemies: Any) -> dict[str, Any] | None:
        if not isinstance(enemies, list):
            return None
        for enemy in enemies:
            if (
                isinstance(enemy, dict)
                and enemy.get("name") == self.enemy_name
                and bool(enemy.get("alive", False))
            ):
                return enemy
        return None

    def _hp_from_summary(self, summary: dict[str, Any]) -> int:
        combat = summary.get("combat")
        if isinstance(combat, dict):
            return int(combat.get("player_cur_hp", summary.get("cur_hp", 0)) or 0)
        return int(summary.get("cur_hp", 0) or 0)

    def _episode_entry_hp(
        self,
        decisions: list[Any],
        source_metrics: dict[str, Any],
    ) -> int:
        if not decisions:
            return int(source_metrics.get("entry_hp", 0) or 0)
        first = decisions[0]
        state = getattr(first, "state", None)
        if not isinstance(state, dict) and isinstance(first, dict):
            state = first.get("state")
        combat = (state or {}).get("combat") if isinstance(state, dict) else None
        if isinstance(combat, dict):
            return int(combat.get("player_cur_hp", 0) or 0)
        return int(source_metrics.get("entry_hp", 0) or 0)

    def _episode_exit_hp(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
    ) -> int:
        if decisions:
            last = decisions[-1]
            after_state = getattr(last, "after_state", None)
            if not isinstance(after_state, dict) and isinstance(last, dict):
                after_state = last.get("after_state")
            if isinstance(after_state, dict):
                return self._hp_from_summary(after_state)
        return self._hp_from_summary(terminal_state)

    def _episode_n_turns(self, decisions: list[Any]) -> int:
        turns: list[int] = []
        for decision in decisions:
            state = getattr(decision, "state", None)
            if not isinstance(state, dict) and isinstance(decision, dict):
                state = decision.get("state")
            combat = (state or {}).get("combat") if isinstance(state, dict) else None
            if isinstance(combat, dict):
                turns.append(int(combat.get("turn", 0) or 0))
        return max(turns) if turns else 0

    def _decision_action_description(self, decision: Any) -> str:
        selected = getattr(decision, "selected_action", None)
        if not isinstance(selected, dict) and isinstance(decision, dict):
            selected = decision.get("selected_action")
        if not isinstance(selected, dict):
            return ""
        return str(selected.get("description", ""))


ACTION_KINDS = ("attack", "block", "skill", "other")


def classify_action(description: str) -> str:
    if not description.startswith("play"):
        return "other"
    if "(deal" in description:
        return "attack"
    low = description.lower()
    block_cards = (
        "defend",
        "shrug it off",
        "iron wave",
        "true grit",
        "ghostly armor",
        "impervious",
        "power through",
        "sentinel",
    )
    if any(card in low for card in block_cards):
        return "block"
    return "skill"
