from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar

from sts_ai.local_tasks import base
from sts_ai.local_tasks.gremlin_nob import ACTION_KINDS, classify_action


_DECK_RE = re.compile(r"^Deck: \(\d+\): \{(.*)\}$", re.MULTILINE)
_RELICS_RE = re.compile(r"^Relics: \{(.*)\}$", re.MULTILINE)
_POTIONS_RE = re.compile(r"^Potions: (.*)$", re.MULTILINE)
_TARGET_RE = re.compile(r"\[enemy (\d+)\]")


class LocalTaskStartError(RuntimeError):
    """A replay reached a different start state than the manifest records."""


class EliteFightTask:
    """Replay-window task for an encounter with a fixed enemy composition.

    This is intentionally used only by the post-Nob tasks. Gremlin Nob keeps
    its original implementation and version-1 manifest unchanged.
    """

    task_id: ClassVar[str]
    expected_enemies: ClassVar[tuple[tuple[str, int], ...]]
    convincing_hp_loss: ClassVar[int] = 20
    reward_hp_loss_cap: ClassVar[int] = 40

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
            source_meta = (
                base.load_json(rollout.meta_path)
                if rollout.meta_path is not None
                else {}
            )
            world_seed, rollout_index = base.parse_rollout_stem(rollout.stem)
            for ordinal, (start, end) in enumerate(self._fight_windows(records)):
                window_records = records[start : end + 1]
                metrics = self._metrics_from_source_window(window_records)
                fixed_start = self._start_signature(records[start].get("state") or {})
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
                        "source_window_length": end - start + 1,
                        "pre_actions": [
                            record["selected_action"]
                            for record in records[:start]
                            if record.get("action_executed", True)
                            and isinstance(record.get("selected_action"), dict)
                            and record.get("selected_action")
                        ],
                        "encounter": self._encounter_identity(),
                        "fixed_start": fixed_start,
                        "fixed_start_sha256": self._signature_hash(fixed_start),
                        "entry_loadout": self._entry_loadout(records, start),
                        "source_policy": self._source_policy(source_meta),
                        "label": self._label(metrics),
                        "reward": self.reward_from_metrics(metrics),
                        "rwr_multiplicity": self.rwr_multiplicity(metrics),
                        "metrics": metrics,
                    }
                )

        return {
            "task_id": self.task_id,
            "version": 1,
            "encounter": self._encounter_identity(),
            "source_rollout_dir": str(source_rollout_dir),
            "split_rule": {
                "kind": "mod_holdout",
                "unit": "world_seed",
                "holdout_mod": holdout_mod,
                "holdout_remainder": holdout_remainder,
            },
            "reward_spec": {
                "loss_or_incomplete": -1.0,
                "win": f"1 - min(hp_loss, {self.reward_hp_loss_cap}) / "
                f"{self.reward_hp_loss_cap}",
                "range": [-1.0, 1.0],
            },
            "n_windows": len(windows),
            "split_counts": base.split_counts(windows),
            "label_counts": base.label_counts(windows),
            "windows": windows,
        }

    def validate_start(self, summary: dict[str, Any], window: dict[str, Any]) -> None:
        expected = window.get("fixed_start")
        if not isinstance(expected, dict):
            raise LocalTaskStartError(
                f"window {window.get('window_id')!r} has no fixed_start signature"
            )
        actual = self._start_signature(summary)
        if actual != expected:
            raise LocalTaskStartError(
                f"replay start diverged for {window.get('window_id')!r}: "
                f"expected={expected!r}, actual={actual!r}"
            )

    def completion_reason(self, summary: dict[str, Any]) -> str | None:
        combat = summary.get("combat")
        if not isinstance(combat, dict):
            outcome = str(summary.get("outcome", ""))
            if "LOSS" in outcome or "DEATH" in outcome:
                return "player_loss"
            return "task_complete"
        if int(combat.get("player_cur_hp", summary.get("cur_hp", 0)) or 0) <= 0:
            return "player_loss"
        if not self._live_expected_enemies(combat.get("enemies", [])):
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
        won = (
            completion == "task_complete"
            and exit_hp > 0
            and stopped_reason == "task_complete"
        )
        hp_loss = max(0, entry_hp - exit_hp)
        counts = Counter(
            classify_action(self._decision_action_description(decision))
            for decision in decisions
        )
        metrics: dict[str, Any] = {
            "entry_hp": entry_hp,
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "completed": completion is not None,
            "won": won,
            # Kept for the existing local-task comparison/reporting contract.
            "survived": won,
            "completion_reason": completion,
            "label": self._label({"survived": won, "hp_loss": hp_loss}),
            "reward": self.reward_from_metrics({"survived": won, "hp_loss": hp_loss}),
            "n_decisions": len(decisions),
            "n_turns": self._episode_n_turns(decisions),
            "action_counts": {
                kind: int(counts.get(kind, 0)) for kind in ACTION_KINDS
            },
        }
        metrics.update(self._encounter_metrics(decisions, terminal_state))
        return metrics

    def reward_from_metrics(self, metrics: dict[str, Any]) -> float:
        if not bool(metrics.get("survived", metrics.get("won", False))):
            return -1.0
        hp_loss = max(0, int(metrics.get("hp_loss", 0)))
        return 1.0 - min(hp_loss, self.reward_hp_loss_cap) / self.reward_hp_loss_cap

    def rwr_multiplicity(
        self,
        metrics: dict[str, Any],
        *,
        max_multiplier: int = 4,
    ) -> int:
        if not bool(metrics.get("survived", metrics.get("won", False))):
            return 0
        import math

        hp_loss = int(metrics.get("hp_loss", 0))
        return max(1, min(max_multiplier, round(math.exp((20 - hp_loss) / 10.0))))

    def _fight_windows(self, records: list[dict[str, Any]]) -> list[tuple[int, int]]:
        indices = [
            i for i, record in enumerate(records) if self._record_is_live_encounter(record)
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

    def _record_is_live_encounter(self, record: dict[str, Any]) -> bool:
        if record.get("phase") != "combat":
            return False
        combat = (record.get("state") or {}).get("combat")
        if not isinstance(combat, dict):
            return False
        enemies = combat.get("enemies", [])
        return self._matches_encounter(enemies) and bool(
            self._live_expected_enemies(enemies)
        )

    def _metrics_from_source_window(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        first_combat = records[0]["state"]["combat"]
        entry_hp = int(first_combat.get("player_cur_hp", 0) or 0)
        max_hp = int(first_combat.get("player_max_hp", 0) or 0)
        last_after = records[-1].get("after_state") or {}
        exit_hp = self._hp_from_summary(last_after)
        won = self.completion_reason(last_after) == "task_complete" and exit_hp > 0
        hp_loss = max(0, entry_hp - exit_hp)
        turns = [self._turn_from_decision(record) for record in records]
        counts = Counter(
            classify_action(base.selected_action_description(record)) for record in records
        )
        metrics: dict[str, Any] = {
            "entry_hp": entry_hp,
            "max_hp": max_hp,
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "completed": self.completion_reason(last_after) is not None,
            "won": won,
            "survived": won,
            "n_turns": max(turns) if turns else 0,
            "n_decisions": len(records),
            "action_counts": {
                kind: int(counts.get(kind, 0)) for kind in ACTION_KINDS
            },
        }
        metrics.update(self._encounter_metrics(records, last_after))
        return metrics

    def _encounter_metrics(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
    ) -> dict[str, Any]:
        _ = decisions, terminal_state
        return {}

    def _label(self, metrics: dict[str, Any]) -> str:
        if not bool(metrics.get("survived", metrics.get("won", False))):
            return "loss"
        return (
            "convincing"
            if int(metrics.get("hp_loss", 0)) <= self.convincing_hp_loss
            else "won"
        )

    def _expected_counter(self) -> Counter[str]:
        return Counter(dict(self.expected_enemies))

    def _matches_encounter(self, enemies: Any) -> bool:
        if not isinstance(enemies, list):
            return False
        actual = Counter(
            str(enemy.get("name")) for enemy in enemies if isinstance(enemy, dict)
        )
        return actual == self._expected_counter()

    def _live_expected_enemies(self, enemies: Any) -> list[dict[str, Any]]:
        expected_names = set(dict(self.expected_enemies))
        if not isinstance(enemies, list):
            return []
        return [
            enemy
            for enemy in enemies
            if isinstance(enemy, dict)
            and str(enemy.get("name")) in expected_names
            and bool(enemy.get("alive", False))
        ]

    def _encounter_identity(self) -> dict[str, Any]:
        return {
            "enemy_counts": {
                name: count for name, count in self.expected_enemies
            },
            "exact_composition": True,
        }

    def _start_signature(self, summary: dict[str, Any]) -> dict[str, Any]:
        combat = summary.get("combat")
        if not isinstance(combat, dict):
            return {"combat": None}
        enemies = combat.get("enemies", [])
        enemy_rows = []
        if isinstance(enemies, list):
            for ordinal, enemy in enumerate(enemies):
                if not isinstance(enemy, dict):
                    continue
                enemy_rows.append(
                    {
                        "index": int(enemy.get("index", ordinal) or 0),
                        "name": str(enemy.get("name", "")),
                        "cur_hp": int(enemy.get("cur_hp", 0) or 0),
                        "max_hp": int(enemy.get("max_hp", 0) or 0),
                        "block": int(enemy.get("block", 0) or 0),
                        "intent": str(enemy.get("intent", "")),
                        "intent_damage": int(enemy.get("intent_damage", 0) or 0),
                        "intent_hits": int(enemy.get("intent_hits", 0) or 0),
                        "strength": int(enemy.get("strength", 0) or 0),
                        "vulnerable": int(enemy.get("vulnerable", 0) or 0),
                        "weak": int(enemy.get("weak", 0) or 0),
                        "poison": int(enemy.get("poison", 0) or 0),
                        "alive": bool(enemy.get("alive", False)),
                    }
                )
        return {
            "act": int(summary.get("act", 0) or 0),
            "floor": int(summary.get("floor", 0) or 0),
            "room": str(summary.get("room", "")),
            "turn": int(combat.get("turn", 0) or 0),
            "input_state": str(combat.get("input_state", "")),
            "battle_outcome": str(combat.get("battle_outcome", "")),
            "player_cur_hp": int(
                combat.get("player_cur_hp", summary.get("cur_hp", 0)) or 0
            ),
            "player_max_hp": int(
                combat.get("player_max_hp", summary.get("max_hp", 0)) or 0
            ),
            "player_block": int(combat.get("player_block", 0) or 0),
            "player_energy": int(combat.get("player_energy", 0) or 0),
            "enemies": sorted(enemy_rows, key=lambda row: row["index"]),
        }

    def _signature_hash(self, signature: dict[str, Any]) -> str:
        payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _entry_loadout(
        self,
        records: list[dict[str, Any]],
        start: int,
    ) -> dict[str, Any]:
        deck: list[str] = []
        relics: list[dict[str, Any]] = []
        potions: list[str] = []
        found_deck = found_relics = found_potions = False
        for record in reversed(records[: start + 1]):
            state_text = str(record.get("state_text", ""))
            if not found_deck:
                match = _DECK_RE.search(state_text)
                if match:
                    deck = self._comma_items(match.group(1))
                    found_deck = True
            if not found_relics:
                match = _RELICS_RE.search(state_text)
                if match:
                    relics = [self._parse_relic(item) for item in self._comma_items(match.group(1))]
                    found_relics = True
            if not found_potions:
                match = _POTIONS_RE.search(state_text)
                if match:
                    potions = self._comma_items(match.group(1))
                    found_potions = True
            if found_deck and found_relics and found_potions:
                break
        return {
            "deck": deck,
            "relics": relics,
            "potions": potions,
            "audit_complete": found_deck and found_relics and found_potions,
        }

    def _comma_items(self, raw: str) -> list[str]:
        stripped = raw.strip()
        if not stripped or stripped.lower() == "none":
            return []
        return [item.strip() for item in stripped.split(",") if item.strip()]

    def _parse_relic(self, item: str) -> dict[str, Any]:
        name, separator, raw_counter = item.rpartition(":")
        if not separator:
            return {"name": item, "counter": None}
        try:
            counter: int | str = int(raw_counter)
        except ValueError:
            counter = raw_counter
        return {"name": name, "counter": counter}

    def _source_policy(self, meta: dict[str, Any]) -> dict[str, Any]:
        extra = meta.get("extra") or {}
        config = (extra.get("agent_config") or {})
        return {
            "agent": meta.get("agent"),
            "model_id": config.get("model_id", meta.get("model_id")),
            "adapter_path": config.get("adapter_path"),
            "reasoning_mode": config.get("reasoning_mode"),
            "temperature": config.get("temperature", meta.get("temperature")),
            "top_p": config.get("top_p"),
            "top_k": config.get("top_k"),
            "combat_observation": extra.get(
                "combat_observation",
                meta.get("combat_observation", "legacy"),
            ),
            "git_sha": meta.get("git_sha"),
        }

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
        state = self._decision_mapping(decisions[0], "state")
        combat = state.get("combat") if isinstance(state, dict) else None
        if isinstance(combat, dict):
            return int(combat.get("player_cur_hp", 0) or 0)
        return int(source_metrics.get("entry_hp", 0) or 0)

    def _episode_exit_hp(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
    ) -> int:
        if decisions:
            after_state = self._decision_mapping(decisions[-1], "after_state")
            if isinstance(after_state, dict):
                return self._hp_from_summary(after_state)
        return self._hp_from_summary(terminal_state)

    def _episode_n_turns(self, decisions: list[Any]) -> int:
        turns = [self._turn_from_decision(decision) for decision in decisions]
        return max(turns) if turns else 0

    def _turn_from_decision(self, decision: Any) -> int:
        state = self._decision_mapping(decision, "state")
        combat = state.get("combat") if isinstance(state, dict) else None
        return int(combat.get("turn", 0) or 0) if isinstance(combat, dict) else 0

    def _decision_action_description(self, decision: Any) -> str:
        selected = self._decision_mapping(decision, "selected_action")
        return str(selected.get("description", "")) if isinstance(selected, dict) else ""

    def _decision_mapping(self, decision: Any, name: str) -> Any:
        value = getattr(decision, name, None)
        if value is None and isinstance(decision, dict):
            value = decision.get(name)
        return value

    def _enemy_rows(self, decision: Any, field: str = "state") -> list[dict[str, Any]]:
        summary = self._decision_mapping(decision, field)
        combat = summary.get("combat") if isinstance(summary, dict) else None
        enemies = combat.get("enemies") if isinstance(combat, dict) else None
        if not isinstance(enemies, list):
            return []
        return [enemy for enemy in enemies if isinstance(enemy, dict)]


class LagavulinTask(EliteFightTask):
    task_id = "lagavulin"
    expected_enemies = (("LAGAVULIN", 1),)

    def _encounter_metrics(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
    ) -> dict[str, Any]:
        _ = terminal_state
        sleeping_turns: set[int] = set()
        attacks_while_sleeping = 0
        setup_cards_while_sleeping = 0
        wake_turn: int | None = None
        for decision in decisions:
            enemies = self._enemy_rows(decision)
            lagavulin = next(
                (enemy for enemy in enemies if enemy.get("name") == "LAGAVULIN"),
                None,
            )
            if lagavulin is None or not bool(lagavulin.get("alive", False)):
                continue
            turn = self._turn_from_decision(decision)
            sleeping = str(lagavulin.get("intent")) == "LAGAVULIN_SLEEP"
            action = self._decision_action_description(decision)
            if sleeping:
                sleeping_turns.add(turn)
                kind = classify_action(action)
                attacks_while_sleeping += int(kind == "attack")
                setup_cards_while_sleeping += int(action.startswith("play") and kind != "attack")
            elif wake_turn is None:
                wake_turn = turn
        return {
            "sleeping_turns": len(sleeping_turns),
            "wake_turn": wake_turn,
            "attacks_while_sleeping": attacks_while_sleeping,
            "setup_cards_while_sleeping": setup_cards_while_sleeping,
        }


class SentriesTask(EliteFightTask):
    task_id = "sentries"
    expected_enemies = (("SENTRY", 3),)

    def _encounter_metrics(
        self,
        decisions: list[Any],
        terminal_state: dict[str, Any],
    ) -> dict[str, Any]:
        _ = terminal_state
        first_kill_indices: list[int] = []
        first_kill_turn: int | None = None
        target_counts: Counter[int] = Counter()
        for decision in decisions:
            description = self._decision_action_description(decision)
            target_match = _TARGET_RE.search(description)
            if target_match:
                target_counts[int(target_match.group(1))] += 1
            if first_kill_indices:
                continue
            before = {
                int(enemy.get("index", ordinal) or 0)
                for ordinal, enemy in enumerate(self._enemy_rows(decision))
                if enemy.get("name") == "SENTRY" and bool(enemy.get("alive", False))
            }
            after = {
                int(enemy.get("index", ordinal) or 0)
                for ordinal, enemy in enumerate(self._enemy_rows(decision, "after_state"))
                if enemy.get("name") == "SENTRY" and bool(enemy.get("alive", False))
            }
            killed = sorted(before - after)
            if killed and len(before) == 3:
                first_kill_indices = killed
                first_kill_turn = self._turn_from_decision(decision)
        first_kill_index = (
            first_kill_indices[0] if len(first_kill_indices) == 1 else None
        )
        return {
            "first_kill_index": first_kill_index,
            "first_kill_indices": first_kill_indices,
            "first_kill_turn": first_kill_turn,
            "simultaneous_first_kills": len(first_kill_indices) > 1,
            "outer_sentry_killed_first": first_kill_index in {0, 2},
            "middle_sentry_killed_first": first_kill_index == 1,
            "targeted_action_counts": {
                str(index): int(target_counts.get(index, 0)) for index in range(3)
            },
        }
