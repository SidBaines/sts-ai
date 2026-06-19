"""Unit tests for the policy-gradient dataset builder."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.train.pg_dataset import build_pg_dataset


FRAMING = "Test framing: choose the strongest legal action."


class FakeTokenizer:
    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return (
            f"wrapped(thinking={enable_thinking},gen={add_generation_prompt})\n"
            f"{messages[0]['content']}"
        )

    def encode(self, text, add_special_tokens=True):
        return text.split()


def _meta(
    *,
    world_seed: int,
    rollout_index: int = 0,
    framing: str = FRAMING,
    reasoning_mode: str | None = "none",
    outcome: str = "GameOutcome.UNDECIDED",
    final_act: int = 1,
    final_floor: int = 1,
    stopped_reason: str = "terminal",
    n_invalid: int = 0,
    n_decisions: int = 1,
) -> dict:
    meta = {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "framing": framing,
        "outcome": outcome,
        "final_act": final_act,
        "final_floor": final_floor,
        "stopped_reason": stopped_reason,
        "n_invalid": n_invalid,
        "n_decisions": n_decisions,
        "extra": {"agent_config": {}},
    }
    if reasoning_mode is not None:
        meta["extra"]["agent_config"]["reasoning_mode"] = reasoning_mode
    return meta


def _record(
    *,
    world_seed: int,
    decision_index: int,
    phase: str = "combat",
    action_executed: bool = True,
    valid: bool = True,
    retries: int = 0,
) -> dict:
    return {
        "world_seed": world_seed,
        "decision_index": decision_index,
        "phase": phase,
        "state_text": f"Seed {world_seed}, decision {decision_index}",
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "defend"},
            {"index": 1, "bits": 1, "description": "strike"},
        ],
        "selected_action": {"index": 1},
        "agent": {
            "action_index": 1,
            "raw_response": f'{{"reasoning": "ok", "action_index": {decision_index}}}',
            "valid": valid,
            "retries": retries,
        },
        "after_state": {},
        "action_executed": action_executed,
    }


def _write_rollout(root: Path, meta: dict, records: list[dict]) -> Path:
    stem = f"seed_{meta['world_seed']}_r{meta['rollout_index']}"
    jsonl_path = root / f"{stem}.jsonl"
    meta_path = root / f"{stem}.meta.json"
    jsonl_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return jsonl_path


class BuildPgDatasetTest(unittest.TestCase):
    def test_offline_advantages_are_broadcast_to_all_kept_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=1, final_floor=5, n_decisions=2),
                [
                    _record(world_seed=1, decision_index=0),
                    _record(world_seed=1, decision_index=1),
                ],
            )
            _write_rollout(
                root,
                _meta(world_seed=2, final_floor=10),
                [_record(world_seed=2, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(world_seed=3, final_floor=15),
                [_record(world_seed=3, decision_index=0)],
            )

            examples, manifest = build_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
            )

            self.assertEqual(manifest["mode"], "offline")
            self.assertEqual(manifest["baseline"], "median")
            self.assertEqual(manifest["n_trajectories_with_advantage"], 3)
            self.assertEqual(manifest["n_examples"], 4)
            self.assertEqual(manifest["advantage_report"]["baseline_value"], 10)
            self.assertEqual(
                [
                    (example["stem"], example["decision_index"], example["advantage"])
                    for example in examples
                ],
                [
                    ("seed_1_r0", 0, -5.0),
                    ("seed_1_r0", 1, -5.0),
                    ("seed_2_r0", 0, 0.0),
                    ("seed_3_r0", 0, 5.0),
                ],
            )

    def test_simulator_error_trajectory_is_excluded_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=1, final_floor=5),
                [_record(world_seed=1, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(
                    world_seed=2,
                    final_floor=999,
                    stopped_reason="simulator_error",
                ),
                [_record(world_seed=2, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(world_seed=3, final_floor=15),
                [_record(world_seed=3, decision_index=0)],
            )

            examples, manifest = build_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
            )

            self.assertEqual(
                {example["stem"] for example in examples},
                {"seed_1_r0", "seed_3_r0"},
            )
            self.assertEqual(manifest["n_trajectories_with_advantage"], 2)
            self.assertEqual(manifest["advantage_report"]["n_simulator_error_excluded"], 1)

    def test_unfaithful_or_invalid_decisions_are_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=5, final_floor=50, n_decisions=4),
                [
                    _record(world_seed=5, decision_index=0, valid=False),
                    _record(world_seed=5, decision_index=1, retries=1),
                    _record(world_seed=5, decision_index=2, action_executed=False),
                    _record(world_seed=5, decision_index=3),
                ],
            )

            examples, manifest = build_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
            )

            self.assertEqual([example["decision_index"] for example in examples], [3])
            self.assertEqual(
                manifest["skipped_record_counts"],
                {
                    "agent_invalid": 1,
                    "agent_retried": 1,
                    "action_not_executed": 1,
                },
            )

    def test_group_mode_broadcasts_world_seed_relative_advantages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=10, rollout_index=0, final_floor=10, n_decisions=2),
                [
                    _record(world_seed=10, decision_index=0),
                    _record(world_seed=10, decision_index=1),
                ],
            )
            _write_rollout(
                root,
                _meta(world_seed=10, rollout_index=1, final_floor=20),
                [_record(world_seed=10, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(world_seed=20, rollout_index=0, final_floor=10),
                [_record(world_seed=20, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(world_seed=20, rollout_index=1, final_floor=20),
                [_record(world_seed=20, decision_index=0)],
            )

            examples, manifest = build_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                mode="group",
                eps=0.0,
            )

            self.assertEqual(manifest["mode"], "group")
            self.assertTrue(manifest["std_norm"])
            self.assertEqual(manifest["eps"], 0.0)
            self.assertEqual(manifest["n_trajectories_with_advantage"], 4)
            self.assertEqual(
                [
                    (example["stem"], example["decision_index"], example["advantage"])
                    for example in examples
                ],
                [
                    ("seed_10_r0", 0, -1.0),
                    ("seed_10_r0", 1, -1.0),
                    ("seed_10_r1", 0, 1.0),
                    ("seed_20_r0", 0, -1.0),
                    ("seed_20_r1", 0, 1.0),
                ],
            )

    def test_manifest_contains_chat_template_hash_and_advantage_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=7, final_floor=50),
                [_record(world_seed=7, decision_index=0)],
            )

            _examples, manifest = build_pg_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
            )

            self.assertEqual(manifest["tokenizer_id"], "fake-tokenizer")
            self.assertIsInstance(manifest["chat_template_hash"], str)
            self.assertEqual(len(manifest["chat_template_hash"]), 16)
            self.assertEqual(manifest["framing"], FRAMING)
            self.assertEqual(manifest["generation_framing"], FRAMING)
            self.assertEqual(manifest["reasoning_mode"], "none")
            self.assertFalse(manifest["enable_thinking"])
            self.assertFalse(manifest["induce_reasoning"])
            self.assertIn("advantage_report", manifest)
            self.assertIn("label_report", manifest)


if __name__ == "__main__":
    unittest.main()
