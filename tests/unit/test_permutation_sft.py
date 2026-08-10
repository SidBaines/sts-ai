from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_paired_permutation_sft import _publish_fresh_files
from sts_ai.permutation_sft import (
    build_paired_datasets,
    finalize_manifest,
    render_jsonl,
)
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
    public_observation_hash,
)
from sts_ai.teacher_action_eval import action_completion
from sts_ai.train.sft_format import (
    assistant_turn_terminator,
    chat_template_probe_hash,
    tokenize_example,
)
from sts_ai.train.train_mlx import _validate_search_teacher_training_contract


MODEL_ID = "local/test-model"


class _CharacterTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        if not messages or messages[0]["role"] != "user":
            raise ValueError("unexpected messages")
        user = messages[0]["content"]
        embedded = user[:-1] if user.endswith("\n") else user
        prompt = (
            "<bos><turn>user\n"
            + embedded
            + "<end>\n<turn>model\n"
        )
        if add_generation_prompt:
            if len(messages) != 1:
                raise ValueError("unexpected generation messages")
            return prompt
        if len(messages) != 2 or messages[1]["role"] != "assistant":
            raise ValueError("unexpected completed messages")
        return prompt + messages[1]["content"] + "<end>\n"

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(character) for character in text]

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
    ):
        del add_special_tokens, return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


def _prompt(tokenizer, user_content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _row(
    tokenizer,
    actions,
    *,
    teacher_index,
    base_index,
    window_id,
    world_seed,
):
    indices = ", ".join(str(index) for index in range(len(actions)))
    state_text = f"Battle turn 1\nPlayer HP: {world_seed}/80"
    user_content = (
        "Choose one action.\n"
        f"Valid action_index values are: {indices}. Use only one.\n\n"
        f"GAME STATE\n{state_text}\n\n"
        "LEGAL ACTIONS\n"
        + "\n".join(
            f"{index}: {description}"
            for index, description in enumerate(actions)
        )
        + "\n"
    )
    completion = action_completion(teacher_index)
    state_hash = public_observation_hash(
        state_text,
        [{"description": description} for description in actions],
        observation_version=PUBLIC_OBSERVATION_VERSION,
    )
    consensus = {
        "action_counts": {str(teacher_index): 3},
        "consensus_action_index": teacher_index,
        "consensus_fraction": 1.0,
        "n_abstentions": 0,
        "n_queries": 3,
        "public_state_hash": state_hash,
        "tied": False,
        "unanimous": True,
    }
    row = {
        "assistant_turn_terminator": assistant_turn_terminator(tokenizer),
        "base_action_index": base_index,
        "completion": completion,
        "decision_index": 4,
        "loss_mask_mode": "action",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion},
        ],
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "output_contract": "action_only",
        "phase": "combat",
        "prompt": _prompt(tokenizer, user_content),
        "public_state_hash": state_hash,
        "search_reference": {
            **deepcopy(consensus),
            "budget_consensus": {"50000": deepcopy(consensus)},
            "eligible": True,
            "hidden_order_consensus": deepcopy(consensus),
            "simulations": 50000,
        },
        "source_stem": f"seed_{world_seed}_r0",
        "target_action_index": teacher_index,
        "teacher_action_index": teacher_index,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "window_id": window_id,
        "world_seed": world_seed,
    }
    tokenized = tokenize_example(row, tokenizer, loss_mask_mode="action")
    row["token_counts"] = {
        key: int(tokenized[key])
        for key in (
            "n_prompt_tokens",
            "n_completion_tokens",
            "n_format_tokens",
            "n_thought_tokens",
            "n_action_tokens",
            "n_supervised_format_tokens",
            "n_supervised_thought_tokens",
            "n_supervised_action_tokens",
            "n_supervised_tokens",
        )
    }
    return row


def _source(tokenizer):
    rows = [
        _row(
            tokenizer,
            ["play A", "play B", "end turn"],
            teacher_index=1,
            base_index=0,
            window_id="seed_7_r0_w0",
            world_seed=7,
        ),
        _row(
            tokenizer,
            ["play C", "end turn"],
            teacher_index=0,
            base_index=1,
            window_id="seed_8_r0_w0",
            world_seed=8,
        ),
    ]
    dataset_sha = hashlib.sha256(render_jsonl(rows)).hexdigest()
    manifest = {
        "kind": "search_teacher_sft",
        "version": 3,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "loss_mask_mode": "action",
        "output_contract": "action_only",
        "enable_thinking": False,
        "tokenizer_id": MODEL_ID,
        "chat_template_hash": chat_template_probe_hash(
            tokenizer,
            enable_thinking=False,
        ),
        "dataset_sha256": dataset_sha,
        "n_examples": len(rows),
        "source_labels": {
            "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return rows, manifest, dataset_sha, manifest_sha


class PairedPermutationSftTests(unittest.TestCase):
    def test_explicit_passes_are_paired_balanced_and_semantic(self):
        tokenizer = _CharacterTokenizer()
        rows, manifest, dataset_sha, manifest_sha = _source(tokenizer)

        built = build_paired_datasets(
            rows,
            manifest,
            tokenizer=tokenizer,
            model_id=MODEL_ID,
            repetitions=5,
            seed=17,
            source_dataset_sha256=dataset_sha,
            source_manifest_sha256=manifest_sha,
            require_paired_token_counts=True,
        )

        self.assertEqual(len(built.control_rows), 10)
        self.assertEqual(len(built.augmented_rows), 10)
        for pass_index in range(5):
            pass_rows = built.control_rows[pass_index * 2 : (pass_index + 1) * 2]
            self.assertEqual(
                {row["source_identity"] for row in pass_rows},
                {row["public_state_hash"] for row in rows},
            )
            self.assertEqual(
                {
                    row["permutation_augmentation"]["pass_index"]
                    for row in pass_rows
                },
                {pass_index},
            )
        self.assertEqual(
            [row["schedule_step"] for row in built.control_rows],
            list(range(10)),
        )
        self.assertEqual(
            [
                row["permutation_augmentation"]["pair_id"]
                for row in built.control_rows
            ],
            [
                row["permutation_augmentation"]["pair_id"]
                for row in built.augmented_rows
            ],
        )
        self.assertTrue(
            built.control_manifest["augmentation"]["paired_token_counts"][
                "all_paired_token_counts_match"
            ]
        )
        self.assertTrue(
            built.control_manifest["augmentation"]["paired_token_counts"][
                "all_batch1_padded_lengths_match"
            ]
        )
        self.assertEqual(
            built.control_manifest["augmentation"]["source_schedule_sha256"],
            built.augmented_manifest["augmentation"]["source_schedule_sha256"],
        )

        by_source = {}
        for row in built.augmented_rows:
            metadata = row["permutation_augmentation"]
            by_source.setdefault(metadata["source_row_index"], []).append(
                metadata["rotation"]
            )
            assigned = row["teacher_action_index"]
            legal_lines = row["messages"][0]["content"].split(
                "\nLEGAL ACTIONS\n",
                1,
            )[1].strip().splitlines()
            self.assertTrue(
                legal_lines[assigned].endswith(
                    metadata["semantic_teacher_action_description"]
                )
            )
            self.assertEqual(
                row["search_reference"]["public_state_hash"],
                row["public_state_hash"],
            )
            self.assertEqual(
                row["search_reference"]["consensus_action_index"],
                assigned,
            )
            provenance = row["permutation_augmentation"][
                "search_reference_provenance"
            ]
            self.assertFalse(
                provenance["teacher_search_reexecuted_for_assigned_order"]
            )
            self.assertEqual(
                row["source_search_reference_sha256"],
                provenance["source_search_reference_sha256"],
            )
        for source_index, rotations in by_source.items():
            menu_size = len(
                rows[source_index]["messages"][0]["content"].split(
                    "\nLEGAL ACTIONS\n",
                    1,
                )[1].strip().splitlines()
            )
            counts = [rotations.count(rotation) for rotation in range(menu_size)]
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertGreaterEqual(counts[0], 1)

    def test_deterministic_outputs_and_strict_v3_trainer_acceptance(self):
        tokenizer = _CharacterTokenizer()
        rows, manifest, dataset_sha, manifest_sha = _source(tokenizer)
        kwargs = {
            "tokenizer": tokenizer,
            "model_id": MODEL_ID,
            "repetitions": 3,
            "seed": 99,
            "source_dataset_sha256": dataset_sha,
            "source_manifest_sha256": manifest_sha,
        }
        first = build_paired_datasets(rows, manifest, **kwargs)
        second = build_paired_datasets(rows, manifest, **kwargs)
        self.assertEqual(first, second)

        for arm_rows, arm_manifest in (
            (first.control_rows, first.control_manifest),
            (first.augmented_rows, first.augmented_manifest),
        ):
            dataset_bytes = render_jsonl(arm_rows)
            finalized = finalize_manifest(arm_manifest, dataset_bytes)
            with tempfile.TemporaryDirectory() as tmp:
                dataset_path = Path(tmp) / "data.jsonl"
                dataset_path.write_bytes(dataset_bytes)
                digest = _validate_search_teacher_training_contract(
                    dataset_path=dataset_path,
                    manifest=finalized,
                    base_model=MODEL_ID,
                    expected_example_count=6,
                )
            self.assertEqual(digest, finalized["dataset_sha256"])

    def test_invalid_duplicate_description_and_reference_hash_fail_closed(self):
        tokenizer = _CharacterTokenizer()
        rows, manifest, _, manifest_sha = _source(tokenizer)
        duplicate = deepcopy(rows)
        user = duplicate[0]["messages"][0]["content"].replace(
            "1: play B",
            "1: play A",
        )
        duplicate[0]["messages"][0]["content"] = user
        duplicate[0]["prompt"] = _prompt(tokenizer, user)
        duplicate_sha = hashlib.sha256(render_jsonl(duplicate)).hexdigest()
        duplicate_manifest = deepcopy(manifest)
        duplicate_manifest["dataset_sha256"] = duplicate_sha
        with self.assertRaisesRegex(
            ValueError,
            "legal_action_descriptions_not_unique",
        ):
            build_paired_datasets(
                duplicate,
                duplicate_manifest,
                tokenizer=tokenizer,
                model_id=MODEL_ID,
                repetitions=2,
                seed=0,
                source_dataset_sha256=duplicate_sha,
                source_manifest_sha256=manifest_sha,
            )

        bad_reference = deepcopy(rows)
        bad_reference[0]["search_reference"]["public_state_hash"] = "0" * 64
        bad_sha = hashlib.sha256(render_jsonl(bad_reference)).hexdigest()
        bad_manifest = deepcopy(manifest)
        bad_manifest["dataset_sha256"] = bad_sha
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            build_paired_datasets(
                bad_reference,
                bad_manifest,
                tokenizer=tokenizer,
                model_id=MODEL_ID,
                repetitions=2,
                seed=0,
                source_dataset_sha256=bad_sha,
                source_manifest_sha256=manifest_sha,
            )

    def test_atomic_publisher_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing"
            fresh = root / "fresh"
            existing.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "must_be_fresh"):
                _publish_fresh_files(
                    ((existing, b"replace"), (fresh, b"new"))
                )
            self.assertEqual(existing.read_bytes(), b"keep")
            self.assertFalse(fresh.exists())


if __name__ == "__main__":
    unittest.main()
