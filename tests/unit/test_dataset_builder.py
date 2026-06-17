"""Unit tests for the offline SFT dataset builder."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.train.dataset_builder import build_dataset, discover_rollouts


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


class DiscoverRolloutsTest(unittest.TestCase):
    def test_pairs_jsonl_and_meta_and_ignores_missing_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired = _write_rollout(root, _meta(world_seed=2), [])
            missing = root / "seed_3_r0.jsonl"
            missing.write_text("", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            _write_rollout(nested, _meta(world_seed=4), [])

            pairs = discover_rollouts(root)

            self.assertEqual(pairs, [(paired, paired.with_suffix(".meta.json"))])


class BuildDatasetTest(unittest.TestCase):
    def test_only_kept_trajectories_contribute_examples_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=1, outcome="GameOutcome.VICTORY", final_act=2, final_floor=50),
                [_record(world_seed=1, decision_index=0)],
            )
            _write_rollout(
                root,
                _meta(world_seed=2, final_act=1, final_floor=1),
                [_record(world_seed=2, decision_index=0)],
            )

            examples, manifest = build_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                min_positives=1,
            )

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["stem"], "seed_1_r0")
            self.assertEqual(examples[0]["keep_reason"], "victory")
            self.assertEqual(examples[0]["world_seed"], 1)
            self.assertFalse(manifest["filter_report"]["fallback_engaged"])
            self.assertEqual(manifest["n_kept_trajectories"], 1)
            self.assertEqual(manifest["n_examples"], 1)

    def test_unfaithful_or_invalid_decisions_are_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=5, outcome="GameOutcome.VICTORY", final_act=2, final_floor=50),
                [
                    _record(world_seed=5, decision_index=0, action_executed=False),
                    _record(world_seed=5, decision_index=1, retries=1),
                    _record(world_seed=5, decision_index=2, valid=False),
                    _record(world_seed=5, decision_index=3),
                ],
            )

            examples, manifest = build_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                min_positives=1,
            )

            self.assertEqual([example["decision_index"] for example in examples], [3])
            self.assertEqual(
                manifest["skipped_record_counts"],
                {
                    "action_not_executed": 1,
                    "agent_retried": 1,
                    "agent_invalid": 1,
                },
            )

    def test_manifest_contains_expected_provenance_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=7, outcome="GameOutcome.VICTORY", final_act=2, final_floor=50),
                [_record(world_seed=7, decision_index=0)],
            )

            _examples, manifest = build_dataset(
                root,
                framing=FRAMING,
                tokenizer=FakeTokenizer(),
                tokenizer_id="fake-tokenizer",
                min_positives=1,
            )

            self.assertEqual(manifest["tokenizer_id"], "fake-tokenizer")
            self.assertIsInstance(manifest["chat_template_hash"], str)
            self.assertEqual(len(manifest["chat_template_hash"]), 16)
            self.assertEqual(manifest["framing"], FRAMING)
            self.assertEqual(manifest["generation_framing"], FRAMING)
            self.assertEqual(manifest["reasoning_mode"], "none")
            self.assertFalse(manifest["enable_thinking"])
            self.assertFalse(manifest["induce_reasoning"])
            self.assertIn("filter_report", manifest)
            self.assertIn("skipped_record_counts", manifest)


class BuildDatasetGuardTest(unittest.TestCase):
    def test_require_no_thinking_rejects_native_reasoning_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(
                root,
                _meta(world_seed=11, reasoning_mode="native"),
                [],
            )

            with self.assertRaisesRegex(ValueError, "reasoning_mode"):
                build_dataset(
                    root,
                    framing=FRAMING,
                    tokenizer=FakeTokenizer(),
                    tokenizer_id="fake-tokenizer",
                )

    def test_mixed_reasoning_modes_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root, _meta(world_seed=12, reasoning_mode="none"), [])
            _write_rollout(root, _meta(world_seed=13, reasoning_mode="prompted"), [])

            with self.assertRaisesRegex(ValueError, "reasoning_mode"):
                build_dataset(
                    root,
                    framing=FRAMING,
                    tokenizer=FakeTokenizer(),
                    tokenizer_id="fake-tokenizer",
                    require_no_thinking=False,
                )

    def test_framing_mismatch_raises_when_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root, _meta(world_seed=14, framing="other framing"), [])

            with self.assertRaisesRegex(ValueError, "framing"):
                build_dataset(
                    root,
                    framing=FRAMING,
                    tokenizer=FakeTokenizer(),
                    tokenizer_id="fake-tokenizer",
                )


if __name__ == "__main__":
    unittest.main()
