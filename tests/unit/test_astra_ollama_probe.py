from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.astra_ollama_probe import (
    http_request, make_request, parse_response, run, summarize, validate_rows,
)
from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction


class TestAstraOllamaProbe(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {"probe_id": 4, "probe_screen": "MAP_SCREEN", "world_seed": 123,
                    "decision_index": 8, "source_trace": "frozen.jsonl", "state_text": "Map",
                    "legal_actions": [{"index": 0, "bits": 3, "description": "Choose map node x=3"}]}

    def test_request_preserves_shared_prompt_and_seed_independent_of_batch(self) -> None:
        request = make_request(self.row, model="gemma4:26b-mlx", temperature=.7,
                               max_tokens=256, seed=905, num_ctx=8192)
        self.assertEqual(request["messages"][0]["content"], render_action_prompt(
            "Map", [LegalAction(**self.row["legal_actions"][0])], output_contract="action_text"))
        self.assertEqual(request["options"]["seed"], 909)
        self.assertFalse(request["think"])
        self.assertFalse(request["stream"])
        self.assertNotIn("format", request)

    def test_schema_error_is_invalid_even_on_singleton_menu(self) -> None:
        wire = {"response": {"done": True, "message": {"content": '{"action":"x=3"}'}}}
        parsed = parse_response(self.row, wire, 256)
        self.assertFalse(parsed["decision"]["valid"])
        self.assertEqual(summarize([parsed], 1)["valid"], 0)

    def test_separate_thinking_never_replaces_actual_content(self) -> None:
        wire = {"response": {"done": True, "message": {
            "content": '{"action":"Choose map node x=3"}', "thinking": "private reasoning"},
            "eval_count": 9, "prompt_eval_count": 22}}
        parsed = parse_response(self.row, wire, 256)
        self.assertTrue(parsed["decision"]["valid"])
        self.assertEqual(parsed["decision"]["raw_response"], wire["response"]["message"]["content"])
        self.assertTrue(parsed["decision"]["metadata"]["separate_thinking_returned"])

    def test_incomplete_response_is_protocol_failure_not_policy_invalid(self) -> None:
        result = parse_response(self.row, {"response": {"done": False, "message": {"content": ""}}}, 256)
        self.assertIsNone(result["decision"])
        self.assertFalse(summarize([result], 1)["complete"])

    def test_timeout_kills_client_and_records_error(self) -> None:
        with patch("scripts.experiments.astra_ollama_probe.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)) as mocked:
            result = http_request("http://localhost/api/chat", {}, .25)
        self.assertEqual(result["error"], "client_wall_timeout")
        self.assertEqual(mocked.call_args.kwargs["timeout"], .25)

    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_rows([self.row, self.row])

    def test_prepare_only_never_calls_http_and_freezes_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cohort.jsonl"
            original = (json.dumps(self.row) + "\n").encode()
            source.write_bytes(original)
            out = Path(tmp) / "prepared"
            args = argparse.Namespace(source=source, out=out, model="gemma4:26b-mlx",
                                      base_url="http://127.0.0.1:11434", temperature=.7,
                                      max_tokens=256, num_ctx=8192, seed=905, limit=None,
                                      timeout=1, total_seconds=2, execute=False)
            with patch("scripts.experiments.astra_ollama_probe.http_request") as mocked:
                run(args)
            mocked.assert_not_called()
            self.assertEqual((out / "source_cohort.jsonl").read_bytes(), original)
            self.assertFalse((out / "report.json").exists())
            self.assertEqual(json.loads((out / "config.json").read_text())["selected_probe_ids"], [4])
            with self.assertRaises(FileExistsError):
                run(args)

    def test_error_stops_without_retry_or_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cohort.jsonl"
            source.write_text(json.dumps(self.row) + "\n" + json.dumps({**self.row, "probe_id": 9}) + "\n")
            args = argparse.Namespace(source=source, out=Path(tmp) / "run", model="gemma4:26b-mlx",
                                      base_url="http://127.0.0.1:11434", temperature=.7,
                                      max_tokens=256, num_ctx=8192, seed=905, limit=None,
                                      timeout=1, total_seconds=2, execute=True)
            replies = [{"response": {}, "status": 200}] * 3 + [{"error": "client_wall_timeout"}]
            with patch("scripts.experiments.astra_ollama_probe.http_request", side_effect=replies) as mocked:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    run(args)
            self.assertEqual(mocked.call_count, 4)
            report = json.loads((args.out / "report.json").read_text())
            self.assertEqual(report["attempted"], 1)
            self.assertEqual(report["expected"], 2)
            self.assertFalse(report["complete"])
