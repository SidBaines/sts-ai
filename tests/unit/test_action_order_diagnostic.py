from __future__ import annotations

from copy import deepcopy
import json
import re
import unittest

from sts_ai.action_order_diagnostic import (
    all_nonzero_rotations,
    build_checkpoint_diagnostic,
    build_diagnostic_report,
    cyclic_index_maps,
    parse_teacher_menu,
    rotate_teacher_row,
    validate_reference_report,
)
from sts_ai.teacher import PUBLIC_OBSERVATION_VERSION, public_observation_hash
from sts_ai.teacher_action_eval import (
    CandidateSequenceScore,
    action_completion,
    build_teacher_action_report,
)


_LEGAL_ACTION_RE = re.compile(r"^([0-9]+): (.+)$", flags=re.MULTILINE)


def _render(messages: list[dict[str, str]]) -> str:
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("unexpected messages")
    content = messages[0]["content"]
    embedded = content[:-1] if content.endswith("\n") else content
    return (
        "<bos><start_of_turn>user\n"
        + embedded
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _make_row(
    actions: list[str],
    *,
    teacher_index: int = 0,
    state_text: str = "player hp: 40/80",
    window_id: str = "seed_7_r0_w0",
) -> dict[str, object]:
    indices = ", ".join(str(index) for index in range(len(actions)))
    action_lines = "\n".join(
        f"{index}: {description}"
        for index, description in enumerate(actions)
    )
    user_content = (
        "Choose one action.\n"
        f"Valid action_index values are: {indices}. Use only one.\n\n"
        f"GAME STATE\n{state_text}\n\n"
        f"LEGAL ACTIONS\n{action_lines}\n"
    )
    completion = action_completion(teacher_index)
    return {
        "loss_mask_mode": "action",
        "output_contract": "action_only",
        "prompt": _render([{"role": "user", "content": user_content}]),
        "completion": completion,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion},
        ],
        "target_action_index": teacher_index,
        "teacher_action_index": teacher_index,
        "window_id": window_id,
        "world_seed": 7,
        "decision_index": 11,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "public_state_hash": public_observation_hash(
            state_text,
            [{"description": description} for description in actions],
            observation_version=PUBLIC_OBSERVATION_VERSION,
        ),
    }


def _replace_user_content(
    row: dict[str, object],
    content: str,
) -> dict[str, object]:
    changed = deepcopy(row)
    messages = changed["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict)
    messages[0]["content"] = content
    changed["prompt"] = _render([{"role": "user", "content": content}])
    return changed


class _MenuScorer:
    def __init__(self, semantic_scores: dict[str, float] | None = None):
        self.semantic_scores = semantic_scores
        self.calls = 0

    def score_candidates(
        self,
        prompt: str,
        completions: list[str],
    ) -> list[CandidateSequenceScore]:
        self.calls += 1
        legal_block = prompt.split("\nLEGAL ACTIONS\n", 1)[1]
        legal_block = legal_block.split("<end_of_turn>", 1)[0]
        descriptions = [
            description
            for _, description in _LEGAL_ACTION_RE.findall(legal_block)
        ]
        values: list[CandidateSequenceScore] = []
        for completion in completions:
            action_index = int(json.loads(completion)["action_index"])
            if self.semantic_scores is None:
                score = -float(action_index)
            else:
                score = self.semantic_scores[descriptions[action_index]]
            values.append(CandidateSequenceScore(score, 1))
        return values


class ActionOrderParsingTests(unittest.TestCase):
    def test_parse_exact_menu_and_terminal_newline_normalization(self) -> None:
        row = _make_row(["A: keep 7 block", "B", "C"], teacher_index=1)
        parsed = parse_teacher_menu(row)

        self.assertEqual(
            parsed.action_descriptions,
            ("A: keep 7 block", "B", "C"),
        )
        self.assertEqual(parsed.teacher_action_index, 1)
        self.assertTrue(parsed.chat_template_strips_terminal_user_newline)
        self.assertEqual(parsed.state_text, "player hp: 40/80")
        self.assertEqual(parsed.source_public_state_hash, row["public_state_hash"])

    def test_rotation_golden_mapping_and_hashes(self) -> None:
        row = _make_row(["A: keep 7 block", "B", "C"], teacher_index=1)
        original = deepcopy(row)

        rotated = rotate_teacher_row(row, 1, prompt_renderer=_render)

        self.assertEqual(rotated.original_to_assigned, (1, 2, 0))
        self.assertEqual(rotated.assigned_to_original, (2, 0, 1))
        self.assertEqual(
            rotated.assigned_action_descriptions,
            ("C", "A: keep 7 block", "B"),
        )
        self.assertEqual(rotated.assigned_teacher_action_index, 2)
        self.assertEqual(
            rotated.scoring_row["completion"],
            '{"action_index":2}',
        )
        self.assertIn(
            "\nLEGAL ACTIONS\n0: C\n1: A: keep 7 block\n2: B",
            rotated.scoring_row["prompt"],
        )
        self.assertNotEqual(
            rotated.rotated_public_state_hash,
            row["public_state_hash"],
        )
        self.assertEqual(row, original)

    def test_all_rotations_are_bijections_and_cover_each_position(self) -> None:
        row = _make_row(["A", "B", "C", "D"], teacher_index=2)
        rotations = all_nonzero_rotations(row, prompt_renderer=_render)

        self.assertEqual([item.rotation for item in rotations], [1, 2, 3])
        self.assertEqual(
            {item.assigned_teacher_action_index for item in rotations},
            {0, 1, 3},
        )
        for item in rotations:
            for original, assigned in enumerate(item.original_to_assigned):
                self.assertEqual(item.assigned_to_original[assigned], original)
        self.assertEqual(cyclic_index_maps(3, 1), ((1, 2, 0), (2, 0, 1)))

    def test_malformed_rows_fail_closed(self) -> None:
        good = _make_row(["A", "B"], teacher_index=0)
        cases: list[tuple[str, dict[str, object]]] = []

        duplicate = _make_row(["A", "A"], teacher_index=0)
        cases.append(("legal_action_descriptions_not_unique", duplicate))

        noncontiguous_content = str(good["messages"][0]["content"]).replace(
            "\n1: B\n",
            "\n2: B\n",
        )
        cases.append(
            (
                "legal_action_lines_disagree_with_declaration",
                _replace_user_content(good, noncontiguous_content),
            )
        )

        extra_block_content = str(good["messages"][0]["content"]).replace(
            "\nLEGAL ACTIONS\n",
            "\nLEGAL ACTIONS\nnote\nLEGAL ACTIONS\n",
        )
        cases.append(
            (
                "legal_actions_block_count",
                _replace_user_content(good, extra_block_content),
            )
        )

        bad_hash = deepcopy(good)
        bad_hash["public_state_hash"] = "0" * 64
        cases.append(("source_public_state_hash_mismatch", bad_hash))

        for expected, row in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    parse_teacher_menu(row)


class ActionOrderDiagnosticTests(unittest.TestCase):
    def _reference(
        self,
        rows: list[dict[str, object]],
        scorer: _MenuScorer,
    ) -> dict[str, object]:
        return build_teacher_action_report(rows, scorer)

    def test_semantic_scorer_is_invariant_after_mapping_back(self) -> None:
        row = _make_row(["A", "B", "C"], teacher_index=1)
        scorer = _MenuScorer({"A": -3.0, "B": 0.0, "C": -2.0})
        reference = self._reference([row], scorer)

        report = build_checkpoint_diagnostic(
            [row],
            scorer,
            reference,
            checkpoint_label="semantic",
            prompt_renderer=_render,
        )

        self.assertEqual(report["n_source_rows"], 1)
        self.assertEqual(report["n_synthetic_rotated_rows"], 2)
        self.assertEqual(report["n_candidate_sequence_scores"], 6)
        self.assertEqual(report["rotated"]["top1_agreement"], 1.0)
        self.assertEqual(report["rotated"]["semantic_top1_invariance"], 1.0)
        self.assertEqual(
            report["rotated"]["mean_semantic_probability_total_variation"],
            0.0,
        )
        self.assertEqual(
            set(report["rotated_by_assigned_target_position"]),
            {"0", "2"},
        )
        self.assertEqual(set(report["rotated_by_window"]), {row["window_id"]})

    def test_position_scorer_exposes_noninvariance(self) -> None:
        row = _make_row(["A", "B", "C"], teacher_index=0)
        scorer = _MenuScorer()
        reference = self._reference([row], scorer)

        report = build_checkpoint_diagnostic(
            [row],
            scorer,
            reference,
            checkpoint_label="position",
            prompt_renderer=_render,
        )

        self.assertEqual(report["rotated"]["semantic_top1_invariance"], 0.0)
        self.assertEqual(report["rotated"]["top1_agreement"], 0.0)
        self.assertGreater(
            report["rotated"]["mean_semantic_probability_total_variation"],
            0.0,
        )

    def test_tied_semantic_top_set_is_invariant_even_if_top1_tiebreak_moves(
        self,
    ) -> None:
        row = _make_row(["A", "B", "C"], teacher_index=0)
        scorer = _MenuScorer({"A": 0.0, "B": -2.0, "C": 0.0})
        reference = self._reference([row], scorer)

        report = build_checkpoint_diagnostic(
            [row],
            scorer,
            reference,
            checkpoint_label="ties",
            prompt_renderer=_render,
        )

        self.assertEqual(report["rotated"]["semantic_top_set_invariance"], 1.0)
        self.assertLess(report["rotated"]["semantic_top1_invariance"], 1.0)
        self.assertTrue(all(row["top1_tied"] for row in report["rows"]))

    def test_source_macro_does_not_overweight_larger_menus(self) -> None:
        rows = [
            _make_row(
                ["Low", "Best"],
                teacher_index=1,
                window_id="small",
                state_text="small",
            ),
            _make_row(
                ["Best2", "Other", "Else", "More"],
                teacher_index=1,
                window_id="large",
                state_text="large",
            ),
        ]
        scores = {
            "Low": -1.0,
            "Best": 0.0,
            "Best2": 0.0,
            "Other": -1.0,
            "Else": -2.0,
            "More": -3.0,
        }
        scorer = _MenuScorer(scores)
        reference = self._reference(rows, scorer)

        report = build_checkpoint_diagnostic(
            rows,
            scorer,
            reference,
            checkpoint_label="mixed",
            prompt_renderer=_render,
        )

        self.assertEqual(report["rotated"]["n"], 4)
        self.assertEqual(report["rotated"]["top1_agreement"], 0.25)
        self.assertEqual(
            report["rotated_source_macro"]["mean_source_top1_agreement"],
            0.5,
        )

    def test_prompt_round_trip_fails_before_any_rotated_scoring(self) -> None:
        row = _make_row(["A", "B"], teacher_index=0)
        scorer = _MenuScorer({"A": 0.0, "B": -1.0})
        reference = self._reference([row], scorer)
        calls_after_reference = scorer.calls

        with self.assertRaisesRegex(
            ValueError,
            "runtime_chat_template_prompt_mismatch",
        ):
            build_checkpoint_diagnostic(
                [row],
                scorer,
                reference,
                checkpoint_label="bad-renderer",
                prompt_renderer=lambda messages: _render(messages) + "changed",
            )
        self.assertEqual(scorer.calls, calls_after_reference)

    def test_corrupt_reference_candidate_and_summary_are_rejected(self) -> None:
        row = _make_row(["A", "B"], teacher_index=0)
        scorer = _MenuScorer({"A": 0.0, "B": -1.0})
        reference = self._reference([row], scorer)

        corrupt_candidate = deepcopy(reference)
        corrupt_candidate["rows"][0]["candidates"][0][
            "sequence_log_probability"
        ] = -9.0
        with self.assertRaisesRegex(
            ValueError,
            "reference_report_candidate_mean_score_mismatch",
        ):
            validate_reference_report([row], corrupt_candidate)

        corrupt_window = deepcopy(reference)
        corrupt_window["per_window"][row["window_id"]][
            "top1_agreement"
        ] = 0.25
        with self.assertRaisesRegex(
            ValueError,
            "reference_report_overall_summary",
        ):
            validate_reference_report([row], corrupt_window)

    def test_wrapper_requires_same_transformations_for_all_checkpoints(
        self,
    ) -> None:
        row = _make_row(["A", "B"], teacher_index=0)
        scorer = _MenuScorer({"A": 0.0, "B": -1.0})
        reference = self._reference([row], scorer)
        checkpoint = build_checkpoint_diagnostic(
            [row],
            scorer,
            reference,
            checkpoint_label="one",
            prompt_renderer=_render,
        )
        other = deepcopy(checkpoint)
        other["checkpoint_label"] = "two"

        report = build_diagnostic_report(
            {"one": checkpoint, "two": other},
            provenance={"frozen": True},
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["mutates_source_artifacts"])

        other["transformation_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_reports_do_not_share_transformations",
        ):
            build_diagnostic_report(
                {"one": checkpoint, "two": other},
                provenance={},
            )


if __name__ == "__main__":
    unittest.main()
