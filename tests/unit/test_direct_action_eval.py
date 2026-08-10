from __future__ import annotations

import json
import re
import unittest

from sts_ai.direct_action_eval import (
    DirectActionScoreBatch,
    DirectActionTokenScore,
    GreedyCompletion,
    build_direct_action_report,
    isolate_policy_tokens,
)
from sts_ai.teacher import PUBLIC_OBSERVATION_VERSION, public_observation_hash
from sts_ai.teacher_action_eval import action_completion


_LEGAL_ACTION_RE = re.compile(r"^([0-9]+): (.+)$", flags=re.MULTILINE)


def _render(messages: list[dict[str, str]]) -> str:
    content = messages[0]["content"]
    embedded = content[:-1] if content.endswith("\n") else content
    return (
        "<bos><start_of_turn>user\n"
        + embedded
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _row(
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
        "teacher_selection_rule": "aggregated_root_visits",
        "teacher_privilege": "simulator_full_state",
        "public_state_hash": public_observation_hash(
            state_text,
            [{"description": description} for description in actions],
            observation_version=PUBLIC_OBSERVATION_VERSION,
        ),
    }


class _DirectScorer:
    def __init__(
        self,
        semantic_logits: dict[str, float] | None = None,
        *,
        greedy_text: str = '{"action_index":1}',
    ):
        self.semantic_logits = semantic_logits
        self.greedy_text = greedy_text
        self.calls: list[tuple[str, list[int]]] = []

    def score_action_tokens(
        self,
        prompt: str,
        action_indices: list[int],
    ) -> DirectActionScoreBatch:
        action_indices = list(action_indices)
        self.calls.append((prompt, action_indices))
        legal_block = prompt.split("\nLEGAL ACTIONS\n", 1)[1]
        legal_block = legal_block.split("<end_of_turn>", 1)[0]
        descriptions = [
            description
            for _, description in _LEGAL_ACTION_RE.findall(legal_block)
        ]
        candidates = []
        for action_index in action_indices:
            logit = (
                -float(action_index)
                if self.semantic_logits is None
                else self.semantic_logits[descriptions[action_index]]
            )
            candidates.append(
                DirectActionTokenScore(
                    action_index=action_index,
                    token_id=100 + action_index,
                    token_text=str(action_index),
                    logit=logit,
                    full_vocabulary_log_probability=logit - 10.0,
                )
            )
        return DirectActionScoreBatch(
            candidates=tuple(candidates),
            prompt_n_tokens=100,
            context_n_tokens=102,
            common_completion_prefix_token_ids=(1, 2),
            common_completion_suffix_token_ids=(3,),
            model_logits_dtype="bfloat16",
            scoring_dtype="float32",
            transformer_hidden_dtype="bfloat16",
            output_projection_weight_dtype="bfloat16",
            output_projection_mode=(
                "tied_embedding_full_vocabulary_float32_projection_and_softcap"
            ),
        )

    def generate_greedy(self, prompt: str, *, max_tokens: int) -> GreedyCompletion:
        return GreedyCompletion(
            text=self.greedy_text,
            n_tokens=5,
            max_tokens=max_tokens,
            finish_reason="stop",
        )


class PolicyTokenIsolationTests(unittest.TestCase):
    def test_isolates_one_candidate_varying_token(self) -> None:
        prefix, policy, suffix = isolate_policy_tokens(
            [
                [1, 2, 10, 3],
                [1, 2, 11, 3],
                [1, 2, 12, 3],
            ]
        )
        self.assertEqual(prefix, (1, 2))
        self.assertEqual(policy, (10, 11, 12))
        self.assertEqual(suffix, (3,))

    def test_rejects_multi_token_or_aliased_policy_values(self) -> None:
        cases = [
            [[1, 10, 20, 3], [1, 11, 21, 3]],
            [[1, 10, 3], [1, 10, 3]],
            [[1, 10, 3]],
            [[1, True, 3], [1, 11, 3]],
            [[1, 10.0, 3], [1, 11, 3]],
        ]
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    isolate_policy_tokens(values)


class DirectActionReportTests(unittest.TestCase):
    def test_reports_direct_ties_top_set_precision_and_greedy_validity(self) -> None:
        scorer = _DirectScorer(
            {"A": 0.0, "B": 0.0, "C": -2.0},
            greedy_text='{"action_index":1}',
        )
        report = build_direct_action_report(
            [_row(["A", "B", "C"], teacher_index=1)],
            scorer,
        )

        self.assertEqual(report["kind"], "teacher_direct_action_token_likelihood")
        self.assertIn("no JSON suffix", report["candidate_probability_definition"])
        scored = report["rows"][0]
        self.assertEqual(scored["top_action_indices"], [0, 1])
        self.assertTrue(scored["top1_tied"])
        self.assertFalse(scored["top1_agreement"])
        self.assertTrue(scored["teacher_in_top_set"])
        self.assertEqual(scored["model_logits_dtype"], "bfloat16")
        self.assertEqual(scored["scoring_dtype"], "float32")
        self.assertEqual(scored["transformer_hidden_dtype"], "bfloat16")
        self.assertIn(
            "float32_projection",
            scored["output_projection_mode"],
        )
        self.assertEqual(report["scoring_dtypes"], ["float32"])
        self.assertEqual(
            report["by_teacher_action_index"]["1"]["n"],
            1,
        )
        self.assertTrue(scored["greedy"]["json_object"])
        self.assertTrue(scored["greedy"]["schema_valid"])
        self.assertTrue(scored["greedy"]["action_is_legal"])
        self.assertTrue(scored["greedy"]["canonical_action_json"])
        self.assertTrue(scored["greedy"]["teacher_agreement"])
        self.assertEqual(report["overall"]["teacher_in_top_set_rate"], 1.0)
        self.assertEqual(report["overall"]["top1_tied_rate"], 1.0)

    def test_noncanonical_or_illegal_greedy_output_is_visible(self) -> None:
        scorer = _DirectScorer(greedy_text=' {"action_index":9} ')
        report = build_direct_action_report(
            [_row(["A", "B"], teacher_index=0)],
            scorer,
        )
        greedy = report["rows"][0]["greedy"]
        self.assertTrue(greedy["json_object"])
        self.assertTrue(greedy["schema_valid"])
        self.assertFalse(greedy["action_is_legal"])
        self.assertFalse(greedy["canonical_action_json"])

    def test_semantic_scores_remain_invariant_under_cyclic_remapping(self) -> None:
        scorer = _DirectScorer({"A": -3.0, "B": 0.0, "C": -2.0})
        report = build_direct_action_report(
            [_row(["A", "B", "C"], teacher_index=1)],
            scorer,
            greedy_max_tokens=None,
            include_cyclic_rotations=True,
            prompt_renderer=_render,
        )

        cyclic = report["cyclic_rotations"]
        self.assertEqual(cyclic["n_rotated_rows"], 2)
        self.assertEqual(len(cyclic["transformation_set_sha256"]), 64)
        self.assertEqual(cyclic["overall"]["teacher_top1_agreement"], 1.0)
        self.assertEqual(cyclic["overall"]["semantic_top1_invariance"], 1.0)
        self.assertEqual(cyclic["overall"]["semantic_top_set_invariance"], 1.0)
        self.assertEqual(
            cyclic["overall"][
                "mean_candidate_probability_tv_from_unpermuted"
            ],
            0.0,
        )
        for row in cyclic["rows"]:
            self.assertEqual(
                row["mapped_top1_action_index"],
                row["original_teacher_action_index"],
            )
            self.assertEqual(len(row["transformation_sha256"]), 64)
            self.assertEqual(
                sorted(row["original_to_assigned"]),
                list(range(row["menu_size"])),
            )

    def test_cyclic_report_has_position_groups_and_source_macro_weighting(self) -> None:
        report = build_direct_action_report(
            [
                _row(["A", "B"], teacher_index=0, window_id="window_a"),
                _row(
                    ["A", "B", "C", "D"],
                    teacher_index=2,
                    window_id="window_b",
                    state_text="player hp: 35/80",
                ),
            ],
            _DirectScorer({"A": 0.0, "B": -1.0, "C": -2.0, "D": -3.0}),
            greedy_max_tokens=None,
            include_cyclic_rotations=True,
            prompt_renderer=_render,
        )

        cyclic = report["cyclic_rotations"]
        self.assertEqual(cyclic["n_source_rows"], 2)
        self.assertEqual(cyclic["n_rotated_rows"], 4)
        self.assertEqual(cyclic["source_macro"]["n_source_rows"], 2)
        self.assertEqual(
            set(cyclic["by_assigned_teacher_action_index"]),
            {"0", "1", "3"},
        )
        self.assertEqual(set(cyclic["by_window"]), {"window_a", "window_b"})
        self.assertEqual(set(cyclic["by_menu_size"]), {"2", "4"})

    def test_position_scores_expose_semantic_noninvariance(self) -> None:
        report = build_direct_action_report(
            [_row(["A", "B", "C"], teacher_index=0)],
            _DirectScorer(),
            greedy_max_tokens=None,
            include_cyclic_rotations=True,
            prompt_renderer=_render,
        )
        cyclic = report["cyclic_rotations"]
        self.assertEqual(cyclic["overall"]["semantic_top1_invariance"], 0.0)
        self.assertGreater(
            cyclic["overall"][
                "mean_candidate_probability_tv_from_unpermuted"
            ],
            0.0,
        )

    def test_row_contract_failure_is_not_silently_skipped(self) -> None:
        row = _row(["A", "B"])
        row["completion"] = '{"action_index": 0}'
        with self.assertRaisesRegex(ValueError, "teacher row 0"):
            build_direct_action_report([row], _DirectScorer())


if __name__ == "__main__":
    unittest.main()
