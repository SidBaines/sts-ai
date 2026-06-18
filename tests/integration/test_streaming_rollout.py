"""Integration test: the streaming orchestrator is trace-identical to the serial
path under a deterministic agent, and the vLLM path has a tiny pod smoke test.

Gated with @requires_simulator (drives the real engine). The vLLM smoke is also
gated with @requires_vllm and is intended for GPU pods. See tests/CLAUDE.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts_ai.agents import parse_json_action
from sts_ai.hinting import HintConfig

from tests.integration.test_combat_control import ScriptedCombatAgent
from tests.support import requires_simulator, requires_vllm


class ScriptedStreamingAgent(ScriptedCombatAgent):
    """Deterministic in-process GenerationBackend for streaming parity tests."""

    name = "scripted-streaming"

    def __init__(self):
        self.pending = {}

    def stream_submit(self, request_id, state_text, legal_actions, seed, retry=False):
        self.pending[request_id] = (state_text, legal_actions)

    def stream_poll(self):
        if not self.pending:
            return []
        rid = next(iter(self.pending))
        state_text, legal_actions = self.pending.pop(rid)
        decision = self.choose_action(state_text, legal_actions)
        return [
            (
                rid,
                {
                    "text": json.dumps({"action_index": decision.action_index}),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )
        ]

    def build_decision_from_text(self, text, prompt_tokens, completion_tokens, legal_actions):
        return parse_json_action(text, legal_actions)

    def stream_has_unfinished(self):
        return bool(self.pending)


def _make_env(seed: int):
    from sts_ai.lightspeed import LightspeedHybridEnv

    return LightspeedHybridEnv(world_seed=seed, battle_simulations=50, max_act=1, combat_control="llm")


def _hint_structure(result):
    structure = {}
    for decision in result.decisions:
        hint = decision.agent.get("metadata", {}).get("hint")
        if hint:
            structure[decision.decision_index] = {
                "final_action_index": hint.get("final_action_index"),
                "selected_action_index": decision.selected_action.get("index"),
                "launder_outcome": hint.get("launder_outcome"),
                "mistake_kind": hint.get("mistake_kind"),
            }
    return structure


@requires_simulator
class StreamingParityTest(unittest.TestCase):
    def test_streaming_concurrency1_matches_serial(self):
        from sts_ai.rollout import run_rollout
        from sts_ai.streaming_rollout import run_streaming_rollouts

        max_decisions = 40
        serial = run_rollout(_make_env(3), ScriptedStreamingAgent(), max_decisions=max_decisions)
        streaming = run_streaming_rollouts([(3, 0)], _make_env, ScriptedStreamingAgent(),
                                           concurrency=1, max_decisions=max_decisions)[0]

        self.assertEqual(serial.stopped_reason, streaming.stopped_reason)
        self.assertEqual(len(serial.decisions), len(streaming.decisions))
        self.assertGreater(len(serial.decisions), 0)
        for a, b in zip(serial.decisions, streaming.decisions):
            self.assertEqual(a.phase, b.phase)
            self.assertEqual(a.selected_action, b.selected_action)
            self.assertEqual(a.affordances, b.affordances)

    def test_multi_seed_streaming_runs(self):
        from sts_ai.streaming_rollout import run_streaming_rollouts

        results = run_streaming_rollouts([(3, 0), (4, 0)], _make_env, ScriptedStreamingAgent(),
                                         concurrency=2, max_decisions=30)
        self.assertEqual([r.world_seed for r in results], [3, 4])
        self.assertEqual([r.rollout_index for r in results], [0, 0])
        for r in results:
            self.assertGreater(len(r.decisions), 0)
            self.assertIsNone(r.error)

    def test_streaming_concurrency1_hint_structure_matches_serial(self):
        from sts_ai.rollout import run_rollout
        from sts_ai.streaming_rollout import run_streaming_rollouts

        max_decisions = 40
        hint_cfg = HintConfig(enabled=True)
        serial = run_rollout(
            _make_env(3),
            ScriptedStreamingAgent(),
            max_decisions=max_decisions,
            hint_cfg=hint_cfg,
        )
        streaming = run_streaming_rollouts(
            [(3, 0)],
            _make_env,
            ScriptedStreamingAgent(),
            concurrency=1,
            max_decisions=max_decisions,
            hint_cfg=hint_cfg,
        )[0]

        serial_hints = _hint_structure(serial)
        streaming_hints = _hint_structure(streaming)
        self.assertEqual(set(serial_hints), set(streaming_hints))
        self.assertEqual(serial_hints, streaming_hints)


@requires_simulator
@requires_vllm
class StreamingVllmSmokeTest(unittest.TestCase):
    def test_tiny_model_streaming_writes_outputs(self):
        from sts_ai.agents import VllmJsonAgent
        from sts_ai.streaming_rollout import run_streaming_rollouts

        # Pod smoke test: model load + GPU generation through the streaming path.
        agent = VllmJsonAgent(model_id="Qwen/Qwen3-0.6B", max_tokens=256, temperature=0.0)
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            results = run_streaming_rollouts(
                [(3, 0)],
                _make_env,
                agent,
                output_for=lambda ws, ri: tmp / f"seed_{ws}_r{ri}.jsonl",
                concurrency=1,
                max_decisions=3,
            )
            output_path = tmp / "seed_3_r0.jsonl"
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".meta.json").exists())
            self.assertIsNone(results[0].error)


if __name__ == "__main__":
    unittest.main()
