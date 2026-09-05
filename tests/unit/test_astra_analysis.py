from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.astra_analyze_policy import (
    action_consistency, action_value_consistency, evaluate, indexed_rows,
    original_index_map, paired_regret_on_subset, summary, validate_arm,
    validate_comparison, values_for,
)


class TestAstraAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.label = {"probe_id": 7, "world_seed": 1001, "probe_screen": "REWARDS"}

    def test_malformed_response_never_inherits_fallback_action_value(self) -> None:
        prediction = {"decision": {"valid": False, "action_index": 0}}
        row = evaluate(prediction, self.label, {"0": 30, "1": 10})
        self.assertFalse(row["valid"])
        self.assertIsNone(row["regret"])
        result = summary([row])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["n_valid"], 0)
        self.assertIsNone(result["mean_regret_given_valid"])

    def test_claimed_valid_action_missing_from_teacher_values_is_unscored(self) -> None:
        prediction = {"decision": {"valid": True, "action_index": 2}}
        row = evaluate(prediction, self.label, {"0": 30, "1": 10})
        self.assertFalse(row["valid"])
        self.assertIsNone(row["normalized_regret"])

    def test_tied_values_count_as_zero_regret_but_not_informative(self) -> None:
        prediction = {"decision": {"valid": True, "action_index": 1}}
        row = evaluate(prediction, self.label, {"0": 16, "1": 16})
        self.assertTrue(row["top_set_match"])
        self.assertEqual(row["regret"], 0)
        self.assertIsNone(row["normalized_regret"])
        result = summary([row])
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["n_valid_informative"], 0)
        self.assertIsNone(result["top_set_rate_given_valid_informative"])

    def test_normalized_regret_is_relative_to_this_states_value_range(self) -> None:
        prediction = {"decision": {"valid": True, "action_index": 1}}
        row = evaluate(prediction, self.label, {"0": 30, "1": 20, "2": 10})
        self.assertEqual(row["regret"], 10)
        self.assertEqual(row["normalized_regret"], .5)
        self.assertFalse(row["top_set_match"])

    def test_incomplete_teacher_action_set_is_not_used(self) -> None:
        label = {"values": {"heuristic": {
            "complete_terminal_action_set": False, "action_final_floors": {"0": 10},
        }}}
        self.assertIsNone(values_for(label, "heuristic"))
        self.assertIsNone(values_for(label, "random3_mean"))

    def test_duplicate_probe_ids_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            indexed_rows([{"probe_id": 0}, {"probe_id": 0}], name="fixture")

    def test_reversed_menu_is_remapped_before_regret_scoring(self) -> None:
        original = [{"index": 0, "bits": 10, "description": "alpha"},
                    {"index": 1, "bits": 20, "description": "beta"}]
        label = {**self.label, "decision_index": 8, "legal_actions": original, "source_trace": "original.jsonl"}
        source = {**label, "state_text": "visible state",
                  "legal_actions": [{"index": 0, "bits": 20, "description": "beta"},
                                    {"index": 1, "bits": 10, "description": "alpha"}],
                  "probe_original_indices": {"0": 1, "1": 0}}
        prediction = {key: label[key] for key in ("probe_id", "world_seed", "decision_index", "probe_screen")}
        prediction["decision"] = {"valid": True, "action_index": 0, "raw_response": '{"action_index":0}'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(json.dumps(source) + "\n")
            config = {"mode": "probe", "contract": "action_only", "constrained": True,
                      "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "adapter": None, "adapter_identity": None}
            mapped, _sources, strict = validate_arm(config, [prediction], {7: label})
            self.assertEqual(strict[0]["chosen_index"], 0)
            self.assertEqual(strict[0]["original_chosen_index"], 1)
            self.assertEqual(mapped[0]["decision"]["action_index"], 1)
            self.assertEqual(evaluate(mapped[0], label, {"0": 20, "1": 30})["regret"], 0)
            # Strict action text must also be parsed against the displayed menu,
            # not the original menu whose positions have different meanings.
            config["contract"] = "action_text"
            prediction["decision"]["raw_response"] = '{"action":"beta"}'
            mapped, _sources, strict = validate_arm(config, [prediction], {7: label})
            self.assertEqual(strict[0]["chosen_index"], 0)
            self.assertEqual(mapped[0]["decision"]["action_index"], 1)
            path.write_text(json.dumps({**source, "state_text": "changed after generation"}) + "\n")
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                validate_arm(config, [prediction], {7: label})

    def test_nonbijective_and_false_identity_menu_mappings_fail(self) -> None:
        actions = [{"index": 0, "bits": 10, "description": "alpha"},
                   {"index": 1, "bits": 20, "description": "beta"}]
        label = {"legal_actions": actions}
        with self.assertRaisesRegex(ValueError, "bijection"):
            original_index_map({"legal_actions": actions, "probe_original_indices": [0, 0]}, label)
        with self.assertRaisesRegex(ValueError, "bits/description"):
            original_index_map({"legal_actions": actions, "probe_original_indices": [1, 0]}, label)

    def test_only_declared_comparison_interventions_are_allowed(self) -> None:
        config = {"model": "frozen_model", "temperature": 0, "batch_size": 16,
                  "max_tokens": 256, "max_retries": 1, "script_sha256": "same-code",
                  "contract": "action_text", "constrained": False, "adapter": None}
        source = {7: {"state_text": "old", "world_seed": 1001, "probe_id": 7}}
        enriched = {7: {"state_text": "old plus public information", "world_seed": 1001, "probe_id": 7}}
        validate_comparison(config, {**config, "constrained": True}, source, enriched)
        with self.assertRaisesRegex(ValueError, "generation/config"):
            validate_comparison(config, {**config, "temperature": .7}, source, enriched)
        with self.assertRaisesRegex(ValueError, "source-row"):
            validate_comparison(config, config, source, {7: {**enriched[7], "world_seed": 1002}})

    def test_missing_generation_provenance_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_comparison({}, {}, {}, {})

    def test_model_change_requires_exact_named_comparison(self) -> None:
        config = {"model": "e4b-bf16", "temperature": 0, "batch_size": 16,
                  "max_tokens": 256, "max_retries": 1, "script_sha256": "same-code"}
        changed = {**config, "model": "26b-a4b-4bit"}
        for pair in (None, ("base_text_free", "unregistered_model_arm")):
            with self.assertRaisesRegex(ValueError, "generation/config"):
                validate_comparison(config, changed, {}, {}, comparison_pair=pair)
        validate_comparison(config, changed, {}, {},
                            comparison_pair=("base_text_free", "gemma26_text_free"))
        with self.assertRaisesRegex(ValueError, "generation/config"):
            validate_comparison(config, {**changed, "temperature": .7}, {}, {},
                                comparison_pair=("base_text_free", "gemma26_text_free"))
        with self.assertRaisesRegex(ValueError, "generation/config"):
            validate_comparison(config, {**changed, "script_sha256": "different-generator"}, {}, {},
                                comparison_pair=("base_text_free", "gemma26_text_free"))

    def test_consistency_uses_remapped_indices_and_conditions_on_both_valid(self) -> None:
        original = [{"probe_id": i, "world_seed": 1001, "probe_screen": "NEOW",
                     "decision": {"valid": True, "action_index": i}} for i in range(3)]
        changed = [
            {**original[0], "decision": {"valid": True, "action_index": 0, "displayed_action_index": 2}},
            {**original[1], "decision": {"valid": True, "action_index": 0}},
            {**original[2], "decision": {"valid": False, "action_index": 2}},
        ]
        result = action_consistency(original, changed)
        self.assertEqual(result["n_states"], 3)
        self.assertEqual(result["n_common_valid"], 2)
        self.assertEqual(result["same_original_action_rate_given_both_valid"], .5)
        self.assertEqual(result["world_bootstrap_ci"], [.5, .5])

    def test_action_change_can_be_harmless_teacher_tie_even_in_informative_state(self) -> None:
        first = [{"probe_id": i, "decision": {"valid": True, "action_index": 0}} for i in range(3)]
        second = [{"probe_id": i, "decision": {"valid": True, "action_index": 1}} for i in range(3)]
        values = [{"0": 30, "1": 30, "2": 10}, {"0": 30, "1": 10}, {"0": 16, "1": 16}]
        labels = {i: {"values": {"heuristic": {"complete_terminal_action_set": True, "action_final_floors": value}}}
                  for i, value in enumerate(values)}
        result = action_value_consistency(first, second, labels, "heuristic")
        all_rows = result["complete_teacher_and_both_valid"]
        informative = result["informative_complete_teacher_and_both_valid"]
        self.assertEqual(all_rows["n"], 3)
        self.assertEqual(all_rows["n_different_action_equal_value"], 2)
        self.assertEqual(informative["n"], 2)
        self.assertEqual(informative["n_different_action_equal_value"], 1)
        self.assertEqual(informative["n_different_action_unequal_value"], 1)
        self.assertEqual(informative["same_action_rate"], 0)
        self.assertEqual(informative["equivalent_value_rate"], .5)

    def test_changed_observation_secondary_excludes_unchanged_states(self) -> None:
        first = [{"probe_id": i, "world_seed": 1001, "valid": True,
                  "informative": True, "regret": 4, "normalized_regret": .5,
                  "top_set_match": False} for i in range(2)]
        second = [{**first[0], "regret": 100}, {**first[1], "regret": 2}]
        result = paired_regret_on_subset(first, second, {1})
        self.assertEqual(result["n_common_valid_states"], 1)
        self.assertEqual(result["mean_world_regret_delta_second_minus_first"], -2)
        self.assertEqual(result["bootstrap_ci"], [-2, -2])


if __name__ == "__main__":
    unittest.main()
