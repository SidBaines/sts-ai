from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.local_tasks import base
from sts_ai.local_tasks.gremlin_nob import GremlinNobTask, classify_action
from sts_ai.local_tasks.pg_dataset import build_local_pg_dataset
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
