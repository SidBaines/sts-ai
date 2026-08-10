from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.local_task_build_sft import parse_args as parse_local_sft_args
from scripts.local_task_eval import parse_args as parse_eval_args
from scripts.local_task_grpo import parse_args as parse_grpo_args
from sts_ai.interactive.replay import ReplayError
from sts_ai.local_tasks import base, get_task, task_ids
from sts_ai.local_tasks.elite_fights import (
    LagavulinTask,
    LocalTaskStartError,
    SentriesTask,
)
from sts_ai.local_tasks.gremlin_nob import GremlinNobTask, classify_action
from sts_ai.local_tasks.pg_dataset import build_local_pg_dataset
from sts_ai.local_tasks.runner import _resolve_task_replay_action
from sts_ai.local_tasks.sft_dataset import build_local_sft_dataset


FRAMING = "Neutral local task framing."


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        _ = tokenize
        return f"wrapped({enable_thinking},{add_generation_prompt})::{messages[0]['content']}"

    def encode(self, text, add_special_tokens=True):
        _ = add_special_tokens
        return text.split()


class OffsetCharTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        _ = tokenize, enable_thinking
        rendered = f"<user>{messages[0]['content']}<turn><model>"
        if len(messages) == 2:
            return rendered + messages[1]["content"] + "<turn>"
        return rendered if add_generation_prompt else rendered.removesuffix("<model>")

    def encode(self, text, add_special_tokens=True):
        prefix = [999_999] if add_special_tokens else []
        return prefix + [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("action masking must request plain offsets")
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _nob_state(*, player_hp: int = 60, nob_hp: int = 80, alive: bool = True, turn: int = 1):
    return {
        "phase": "combat",
        "cur_hp": player_hp,
        "combat": {
            "turn": turn,
            "player_cur_hp": player_hp,
            "player_max_hp": 80,
            "enemies": [
                {
                    "name": "GREMLIN_NOB",
                    "alive": alive,
                    "cur_hp": nob_hp,
                    "strength": 0,
                }
            ],
        },
    }


def _elite_state(
    names: list[str],
    *,
    player_hp: int = 60,
    enemy_hp: list[int] | None = None,
    alive: list[bool] | None = None,
    intents: list[str] | None = None,
    turn: int = 1,
):
    enemy_hp = enemy_hp or [40] * len(names)
    alive = alive or [True] * len(names)
    intents = intents or ["ATTACK"] * len(names)
    return {
        "phase": "combat",
        "act": 1,
        "floor": 11,
        "room": "Room.ELITE",
        "cur_hp": player_hp,
        "max_hp": 80,
        "combat": {
            "turn": turn,
            "player_cur_hp": player_hp,
            "player_max_hp": 80,
            "enemies": [
                {
                    "index": index,
                    "name": name,
                    "alive": alive[index],
                    "cur_hp": enemy_hp[index],
                    "max_hp": 40,
                    "block": 0,
                    "intent": intents[index],
                }
                for index, name in enumerate(names)
            ],
        },
    }


def _record(
    *,
    world_seed: int,
    decision_index: int,
    selected: str = "play Strike -> GREMLIN_NOB [enemy 0] (deal 6)",
    state: dict | None = None,
    after_state: dict | None = None,
    valid: bool = True,
    retries: int = 0,
):
    return {
        "world_seed": world_seed,
        "decision_index": decision_index,
        "phase": "combat" if state else "out_of_combat",
        "state": state or {"phase": "out_of_combat"},
        "state_text": f"state {world_seed}/{decision_index}",
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "end turn"},
            {"index": 1, "bits": 1, "description": selected},
        ],
        "selected_action": {"index": 1, "bits": 1, "description": selected},
        "agent": {
            "action_index": 1,
            "raw_response": '{"reasoning": "ok", "action_index": 1}',
            "valid": valid,
            "retries": retries,
        },
        "after_state": after_state or {},
        "action_executed": True,
    }


def _meta(world_seed: int, rollout_index: int = 0):
    return {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "framing": FRAMING,
        "extra": {"agent_config": {"reasoning_mode": "none"}},
    }


def _write_rollout(root: Path, world_seed: int, records: list[dict], rollout_index: int = 0):
    stem = f"seed_{world_seed}_r{rollout_index}"
    (root / f"{stem}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / f"{stem}.meta.json").write_text(
        json.dumps(_meta(world_seed, rollout_index)),
        encoding="utf-8",
    )


class GremlinNobTaskTest(unittest.TestCase):
    def test_classify_action(self):
        self.assertEqual(classify_action("play Strike -> GREMLIN_NOB (deal 6)"), "attack")
        self.assertEqual(classify_action("play Defend"), "block")
        self.assertEqual(classify_action("play Bash"), "skill")
        self.assertEqual(classify_action("end turn"), "other")

    def test_build_manifest_extracts_window_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                5,
                [
                    _record(world_seed=5, decision_index=0, selected="choose path"),
                    _record(
                        world_seed=5,
                        decision_index=1,
                        state=_nob_state(player_hp=60, nob_hp=80, turn=1),
                        after_state=_nob_state(player_hp=55, nob_hp=60, turn=1),
                    ),
                    _record(
                        world_seed=5,
                        decision_index=2,
                        state=_nob_state(player_hp=55, nob_hp=20, turn=2),
                        after_state={"phase": "out_of_combat", "cur_hp": 45, "max_hp": 80},
                    ),
                ],
            )

            manifest = GremlinNobTask().build_manifest(root)

            self.assertEqual(manifest["n_windows"], 1)
            window = manifest["windows"][0]
            self.assertEqual(window["split"], "train")
            self.assertEqual(window["start_index"], 1)
            self.assertEqual(window["end_index"], 2)
            self.assertEqual(len(window["pre_actions"]), 1)
            self.assertEqual(window["label"], "convincing")
            self.assertEqual(window["metrics"]["hp_loss"], 15)
            self.assertEqual(window["rwr_multiplicity"], 2)
            self.assertNotIn("start_state_signature", window)

    def test_local_sft_dataset_uses_only_task_window_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                5,
                [
                    _record(world_seed=5, decision_index=0, selected="choose path"),
                    _record(
                        world_seed=5,
                        decision_index=1,
                        state=_nob_state(player_hp=60),
                        after_state=_nob_state(player_hp=55),
                    ),
                    _record(
                        world_seed=5,
                        decision_index=2,
                        state=_nob_state(player_hp=55, turn=2),
                        after_state={"phase": "out_of_combat", "cur_hp": 45, "max_hp": 80},
                    ),
                ],
            )
            manifest = GremlinNobTask().build_manifest(root)

            examples, out_manifest = build_local_sft_dataset(
                manifest,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                split="train",
                label_mode="won",
                weighting_mode="rwr",
            )

            self.assertEqual(out_manifest["n_unique_examples"], 2)
            self.assertEqual(out_manifest["n_examples"], 4)
            self.assertEqual({example["decision_index"] for example in examples}, {1, 2})
            self.assertEqual({example["local_task"] for example in examples}, {"gremlin_nob"})
            self.assertEqual(out_manifest["multiplicity_histogram"], {2: 1})
            self.assertEqual(out_manifest["loss_mask_mode"], "completion")

            action_examples, action_manifest = build_local_sft_dataset(
                manifest,
                framing=FRAMING,
                tokenizer=OffsetCharTokenizer(),
                tokenizer_id="offset-tokenizer",
                split="train",
                label_mode="won",
                weighting_mode="filter",
                loss_mask_mode="action",
            )
            self.assertEqual(action_manifest["loss_mask_mode"], "action")
            self.assertEqual(
                action_manifest["token_accounting"]["n_examples_counted"],
                len(action_examples),
            )
            self.assertTrue(
                all(example["loss_mask_mode"] == "action" for example in action_examples)
            )

    def test_local_pg_dataset_uses_task_reward_group_advantages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rollout_index, reward in ((0, 0.0), (1, 1.0)):
                stem = f"seed_10_r{rollout_index}"
                (root / f"{stem}.jsonl").write_text(
                    json.dumps(_record(world_seed=10, decision_index=0, state=_nob_state())) + "\n",
                    encoding="utf-8",
                )
                meta = {
                    "world_seed": 10,
                    "rollout_index": rollout_index,
                    "framing": FRAMING,
                    "stopped_reason": "task_complete",
                    "extra": {
                        "agent_config": {"reasoning_mode": "none"},
                        "local_task": {
                            "task_id": "gremlin_nob",
                            "window_id": "seed_10_r0_w0",
                            "reward": reward,
                            "metrics": {"hp_loss": 0},
                        },
                    },
                }
                (root / f"{stem}.meta.json").write_text(json.dumps(meta), encoding="utf-8")

            examples, manifest = build_local_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                mode="group",
                eps=0.0,
            )

            self.assertEqual(manifest["n_episodes_with_advantage"], 2)
            self.assertEqual(
                [(example["stem"], example["advantage"]) for example in examples],
                [("seed_10_r0", -1.0), ("seed_10_r1", 1.0)],
            )
            self.assertEqual(manifest["loss_mask_mode"], "completion")

            action_examples, action_manifest = build_local_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=OffsetCharTokenizer(),
                tokenizer_id="offset-tokenizer",
                mode="group",
                eps=0.0,
                loss_mask_mode="action",
            )
            self.assertEqual(action_manifest["loss_mask_mode"], "action")
            self.assertEqual(
                action_manifest["token_accounting"]["n_examples_counted"],
                len(action_examples),
            )


class AdditionalEliteTasksTest(unittest.TestCase):
    def test_registry_exposes_all_local_tasks(self):
        self.assertEqual(task_ids(), ("gremlin_nob", "lagavulin", "sentries"))
        self.assertIsInstance(get_task("lagavulin"), LagavulinTask)
        self.assertIsInstance(get_task("sentries"), SentriesTask)

    def test_lagavulin_manifest_records_fixed_start_loadout_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = _record(world_seed=5, decision_index=0, selected="choose path")
            before["state_text"] = (
                "Deck: (3): {Strike,Defend,Bash+,}\n"
                "Relics: {Burning Blood:0,Anchor:0,}\n"
                "Potions: Weak Potion"
            )
            asleep = _elite_state(
                ["LAGAVULIN"],
                player_hp=60,
                enemy_hp=[111],
                intents=["LAGAVULIN_SLEEP"],
                turn=1,
            )
            awake = _elite_state(
                ["LAGAVULIN"],
                player_hp=60,
                enemy_hp=[100],
                intents=["LAGAVULIN_ATTACK"],
                turn=2,
            )
            won = {"phase": "out_of_combat", "cur_hp": 50, "max_hp": 80}
            _write_rollout(
                root,
                5,
                [
                    before,
                    _record(
                        world_seed=5,
                        decision_index=1,
                        selected="play Strike -> LAGAVULIN [enemy 0] (deal 6)",
                        state=asleep,
                        after_state=awake,
                    ),
                    _record(
                        world_seed=5,
                        decision_index=2,
                        state=awake,
                        after_state=won,
                    ),
                ],
            )
            meta_path = root / "seed_5_r0.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({"agent": "vllm", "git_sha": "abc123"})
            meta["extra"]["agent_config"].update(
                {"model_id": "example/model", "temperature": 0.7}
            )
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

            manifest = LagavulinTask().build_manifest(root)

            self.assertEqual(manifest["n_windows"], 1)
            self.assertEqual(manifest, LagavulinTask().build_manifest(root))
            window = manifest["windows"][0]
            self.assertEqual(window["encounter"]["enemy_counts"], {"LAGAVULIN": 1})
            self.assertEqual(window["entry_loadout"]["deck"], ["Strike", "Defend", "Bash+"])
            self.assertEqual(
                window["entry_loadout"]["relics"],
                [
                    {"name": "Burning Blood", "counter": 0},
                    {"name": "Anchor", "counter": 0},
                ],
            )
            self.assertEqual(window["entry_loadout"]["potions"], ["Weak Potion"])
            self.assertTrue(window["entry_loadout"]["audit_complete"])
            self.assertEqual(window["source_policy"]["model_id"], "example/model")
            self.assertEqual(window["source_policy"]["combat_observation"], "legacy")
            self.assertEqual(len(window["fixed_start_sha256"]), 64)
            self.assertNotIn("start_state_signature", window)
            self.assertEqual(window["metrics"]["wake_turn"], 2)
            self.assertEqual(window["metrics"]["attacks_while_sleeping"], 1)
            self.assertEqual(window["metrics"]["hp_loss"], 10)
            self.assertEqual(window["label"], "convincing")

            LagavulinTask().validate_start(asleep, window)
            divergent = copy.deepcopy(asleep)
            divergent["combat"]["player_cur_hp"] = 59
            with self.assertRaises(LocalTaskStartError):
                LagavulinTask().validate_start(divergent, window)

    def test_sentries_window_persists_after_first_kill_and_tracks_priority(self):
        names = ["SENTRY", "SENTRY", "SENTRY"]
        all_alive = _elite_state(names, player_hp=60, turn=1)
        left_dead = _elite_state(
            names,
            player_hp=55,
            enemy_hp=[0, 35, 30],
            alive=[False, True, True],
            turn=2,
        )
        won = {"phase": "out_of_combat", "cur_hp": 42, "max_hp": 80}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                8,
                [
                    _record(world_seed=8, decision_index=0, selected="choose path"),
                    _record(
                        world_seed=8,
                        decision_index=1,
                        selected="play Strike -> SENTRY [enemy 0] (deal 6)",
                        state=all_alive,
                        after_state=left_dead,
                    ),
                    _record(
                        world_seed=8,
                        decision_index=2,
                        selected="play Strike -> SENTRY [enemy 2] (deal 6)",
                        state=left_dead,
                        after_state=won,
                    ),
                ],
            )

            manifest = SentriesTask().build_manifest(root)

            self.assertEqual(manifest["n_windows"], 1)
            window = manifest["windows"][0]
            self.assertEqual((window["start_index"], window["end_index"]), (1, 2))
            self.assertEqual(window["encounter"]["enemy_counts"], {"SENTRY": 3})
            self.assertEqual(window["metrics"]["first_kill_index"], 0)
            self.assertTrue(window["metrics"]["outer_sentry_killed_first"])
            self.assertFalse(window["metrics"]["middle_sentry_killed_first"])
            self.assertEqual(
                window["metrics"]["targeted_action_counts"],
                {"0": 1, "1": 0, "2": 1},
            )
            self.assertTrue(window["metrics"]["won"])
            self.assertGreaterEqual(window["reward"], -1.0)
            self.assertLessEqual(window["reward"], 1.0)

    def test_incomplete_episode_gets_loss_reward(self):
        state = _elite_state(["LAGAVULIN"], player_hp=60)
        metrics = LagavulinTask().metrics_from_episode(
            [],
            state,
            "max_decisions",
            {"metrics": {"entry_hp": 60}},
        )
        self.assertFalse(metrics["completed"])
        self.assertFalse(metrics["won"])
        self.assertFalse(metrics["survived"])
        self.assertEqual(metrics["label"], "loss")
        self.assertEqual(metrics["reward"], -1.0)

    def test_historical_combat_action_without_upgrade_marker_still_replays(self):
        env = SimpleNamespace(
            legal_actions=lambda: [
                SimpleNamespace(
                    index=3,
                    bits=17,
                    description="play Bash+ (cost 2) -> CULTIST (deal 10)",
                )
            ]
        )
        recorded = {
            "index": 3,
            "bits": 17,
            "description": "play Bash (cost 2) -> CULTIST (deal 10)",
        }

        self.assertEqual(_resolve_task_replay_action(env, recorded), 3)

        select_env = SimpleNamespace(
            legal_actions=lambda: [
                SimpleNamespace(
                    index=1,
                    bits=9,
                    description="select card for HEADBUTT: Rampage=11+",
                )
            ]
        )
        self.assertEqual(
            _resolve_task_replay_action(
                select_env,
                {
                    "index": 1,
                    "bits": 9,
                    "description": "select card for HEADBUTT: Rampage",
                },
            ),
            1,
        )

        discovery_env = SimpleNamespace(
            legal_actions=lambda: [
                SimpleNamespace(
                    index=0,
                    bits=1073741824,
                    description="select card for DISCOVERY: Impervious",
                )
            ]
        )
        self.assertEqual(
            _resolve_task_replay_action(
                discovery_env,
                {
                    "index": 0,
                    "bits": 1073741824,
                    "description": "select card for DISCOVERY (option 0)",
                },
            ),
            0,
        )

        unplayable_env = SimpleNamespace(
            legal_actions=lambda: [
                SimpleNamespace(
                    index=2,
                    bits=3,
                    description="play Dazed (cost unplayable)",
                )
            ]
        )
        self.assertEqual(
            _resolve_task_replay_action(
                unplayable_env,
                {"index": 2, "bits": 3, "description": "play Dazed (cost -2)"},
            ),
            2,
        )

        potion_env = SimpleNamespace(
            legal_actions=lambda: [
                SimpleNamespace(
                    index=4,
                    bits=536870912,
                    description="drink potion Weak Potion -> RED_LOUSE [enemy 0]",
                ),
                SimpleNamespace(
                    index=5,
                    bits=536936448,
                    description="drink potion Weak Potion -> RED_LOUSE [enemy 1]",
                ),
            ]
        )
        self.assertEqual(
            _resolve_task_replay_action(
                potion_env,
                {
                    "index": 4,
                    "bits": 536870912,
                    "description": "drink potion Weak Potion -> RED_LOUSE",
                },
            ),
            4,
        )
        with self.assertRaises(ReplayError):
            _resolve_task_replay_action(
                potion_env,
                {
                    "index": 4,
                    "bits": 123,
                    "description": "drink potion Weak Potion -> RED_LOUSE",
                },
            )

    def test_local_eval_combat_observation_cli_defaults_and_override(self):
        required = [
            "--task",
            "lagavulin",
            "--manifest",
            "manifest.json",
            "--model",
            "example/model",
            "--output-dir",
            "eval",
        ]
        self.assertEqual(parse_eval_args(required).combat_observation, "legacy")
        self.assertEqual(
            parse_eval_args(required).output_contract,
            "reasoning_action",
        )
        self.assertEqual(
            parse_eval_args(
                required + ["--combat-observation", "combat_public_v1"]
            ).combat_observation,
            "combat_public_v1",
        )
        self.assertEqual(
            parse_eval_args(
                required + ["--output-contract", "action_only"]
            ).output_contract,
            "action_only",
        )

    def test_local_grpo_combat_observation_cli_defaults_and_override(self):
        required = [
            "--task",
            "sentries",
            "--base-model",
            "example/model",
            "--tokenizer",
            "example/tokenizer",
            "--manifest",
            "manifest.json",
            "--out-dir",
            "run",
            "--num-iterations",
            "2",
        ]
        self.assertEqual(parse_grpo_args(required).combat_observation, "legacy")
        self.assertEqual(
            parse_grpo_args(
                required + ["--combat-observation", "combat_public_v1"]
            ).combat_observation,
            "combat_public_v1",
        )

    def test_local_sft_cli_defaults_to_action_loss(self):
        required = [
            "--task",
            "lagavulin",
            "--manifest",
            "manifest.json",
            "--tokenizer",
            "example/tokenizer",
            "--out",
            "dataset.jsonl",
        ]
        self.assertEqual(parse_local_sft_args(required).loss_mask, "action")
        self.assertEqual(
            parse_local_sft_args(required + ["--loss-mask", "completion"]).loss_mask,
            "completion",
        )

class InjectLocalTaskMetaTest(unittest.TestCase):
    """inject_local_task_meta reads the trace from disk: a resumed eval must be
    able to repair metas written by an earlier crashed process, whose episodes
    never appear in the resuming run's in-memory results."""

    def _window(self):
        return {
            "window_id": "seed_8_r0_w0",
            "source_stem": "seed_8_r0",
            "split": "holdout",
            "metrics": {"entry_hp": 60},
        }

    def _write_episode(self, root: Path):
        won_after = _nob_state(player_hp=50, nob_hp=0, alive=False, turn=3)
        records = [
            _record(
                world_seed=8,
                decision_index=0,
                state=_nob_state(player_hp=60, nob_hp=80, turn=1),
                after_state=_nob_state(player_hp=55, nob_hp=40, turn=2),
            ),
            _record(
                world_seed=8,
                decision_index=1,
                state=_nob_state(player_hp=55, nob_hp=40, turn=2),
                after_state=won_after,
            ),
        ]
        jsonl_path = root / "seed_8_r0.jsonl"
        jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        meta_path = root / "seed_8_r0.meta.json"
        meta_path.write_text(
            json.dumps({"world_seed": 8, "stopped_reason": "terminal"}),
            encoding="utf-8",
        )
        return meta_path, jsonl_path

    def test_repairs_meta_without_in_memory_result(self):
        from sts_ai.local_tasks.runner import inject_local_task_meta

        task = GremlinNobTask()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path, jsonl_path = self._write_episode(Path(tmp))

            self.assertTrue(
                inject_local_task_meta(meta_path, jsonl_path, task, self._window())
            )

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            local_task = meta["extra"]["local_task"]
            self.assertEqual(local_task["window_id"], "seed_8_r0_w0")
            self.assertEqual(local_task["label"], "convincing")  # 10 HP lost
            self.assertAlmostEqual(local_task["reward"], 0.75)
            self.assertTrue(local_task["metrics"]["survived"])
            self.assertEqual(local_task["metrics"]["hp_loss"], 10)
            self.assertEqual(meta["stopped_reason"], "task_complete")

    def test_idempotent_and_missing_meta_returns_false(self):
        from sts_ai.local_tasks.runner import inject_local_task_meta

        task = GremlinNobTask()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path, jsonl_path = self._write_episode(Path(tmp))
            inject_local_task_meta(meta_path, jsonl_path, task, self._window())
            first = meta_path.read_text(encoding="utf-8")
            inject_local_task_meta(meta_path, jsonl_path, task, self._window())
            self.assertEqual(meta_path.read_text(encoding="utf-8"), first)

            self.assertFalse(
                inject_local_task_meta(
                    Path(tmp) / "absent.meta.json", jsonl_path, task, self._window()
                )
            )


class BaseHelpersTest(unittest.TestCase):
    def test_split_for_seed(self):
        self.assertEqual(base.split_for_seed(8, holdout_mod=4, holdout_remainder=0), "holdout")
        self.assertEqual(base.split_for_seed(9, holdout_mod=4, holdout_remainder=0), "train")


if __name__ == "__main__":
    unittest.main()
