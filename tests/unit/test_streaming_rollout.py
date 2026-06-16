from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from sts_ai.schemas import AgentDecision, LegalAction, RolloutResult
from sts_ai.seeding import derive_batch_seed, derive_policy_seed
from sts_ai.streaming_rollout import run_streaming_rollouts
from tests.unit.test_parallel_rollout import FakeParallelEnv


class FakeStreamingAgent:
    name = "fake-streaming"

    def __init__(self) -> None:
        self.pending: dict[str, dict] = {}
        self.seen_seeds: dict[str, int] = {}
        self.order = "lifo"
        self.idle_every: int = 0
        self._poll_calls = 0

    def stream_submit(
        self,
        request_id: str,
        state_text: str,
        legal_actions: list[LegalAction],
        seed: int,
    ) -> None:
        self.pending[request_id] = {"legal_actions": legal_actions, "seed": seed}
        self.seen_seeds[request_id] = seed

    def stream_poll(self) -> list[tuple[str, dict]]:
        self._poll_calls += 1
        if (
            self.pending
            and self.idle_every > 1
            and self._poll_calls % self.idle_every == 0
        ):
            return []
        if not self.pending:
            return []
        rid = (
            next(reversed(self.pending))
            if self.order == "lifo"
            else next(iter(self.pending))
        )
        request = self.pending.pop(rid)
        legal_actions = request["legal_actions"]
        action_index = request["seed"] % len(legal_actions)
        return [
            (
                rid,
                {
                    "text": str(action_index),
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                },
            )
        ]

    def stream_has_unfinished(self) -> bool:
        return bool(self.pending)

    def build_decision_from_text(
        self,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        legal_actions: list[LegalAction],
    ) -> AgentDecision:
        return AgentDecision(
            action_index=int(text),
            raw_response=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _make_env_with_decisions(decisions: int) -> Callable[[int], FakeParallelEnv]:
    def make_env(world_seed: int) -> FakeParallelEnv:
        return FakeParallelEnv(world_seed=world_seed, decisions=decisions)

    return make_env


def _action_sequences(results: list[RolloutResult]) -> dict[tuple[int, int], list[int]]:
    return {
        (result.world_seed, result.rollout_index): [
            decision.selected_action["index"] for decision in result.decisions
        ]
        for result in results
    }


class StreamingRolloutSpecTest(unittest.TestCase):
    def test_returns_results_in_spec_order_with_correct_identity_and_counts(self) -> None:
        specs = [(7, 0), (7, 1), (8, 0)]

        results = run_streaming_rollouts(
            specs,
            _make_env_with_decisions(3),
            FakeStreamingAgent(),
            concurrency=2,
            max_decisions=200,
        )

        self.assertEqual([(r.world_seed, r.rollout_index) for r in results], specs)
        for result, (world_seed, rollout_index) in zip(results, specs):
            self.assertEqual(
                result.policy_seed,
                derive_policy_seed(world_seed, rollout_index),
            )
            self.assertEqual(result.rollout_index, rollout_index)
            self.assertEqual(len(result.decisions), 3)
            self.assertEqual(result.stopped_reason, "terminal")

    def test_completes_with_intermittent_empty_polls(self) -> None:
        specs = [(7, 0), (7, 1), (8, 0)]
        expected_results = run_streaming_rollouts(
            specs,
            _make_env_with_decisions(3),
            FakeStreamingAgent(),
            concurrency=2,
            max_decisions=200,
        )
        idle_agent = FakeStreamingAgent()
        idle_agent.idle_every = 2

        idle_results = run_streaming_rollouts(
            specs,
            _make_env_with_decisions(3),
            idle_agent,
            concurrency=2,
            max_decisions=200,
        )

        self.assertEqual(
            _action_sequences(idle_results),
            _action_sequences(expected_results),
        )
        self.assertEqual(
            [(len(result.decisions), result.stopped_reason) for result in idle_results],
            [(len(result.decisions), result.stopped_reason) for result in expected_results],
        )

    def test_per_request_seed_is_deterministic_across_completion_orders(self) -> None:
        specs = [(7, 0), (7, 1), (8, 0)]

        lifo_agent = FakeStreamingAgent()
        lifo_agent.order = "lifo"
        lifo_results = run_streaming_rollouts(
            specs,
            _make_env_with_decisions(4),
            lifo_agent,
            concurrency=2,
            max_decisions=200,
        )

        fifo_agent = FakeStreamingAgent()
        fifo_agent.order = "fifo"
        fifo_results = run_streaming_rollouts(
            specs,
            _make_env_with_decisions(4),
            fifo_agent,
            concurrency=2,
            max_decisions=200,
        )

        self.assertEqual(
            _action_sequences(lifo_results),
            _action_sequences(fifo_results),
        )
        self.assertEqual(
            lifo_agent.seen_seeds["7:1:2"],
            derive_batch_seed([(7, 1, 2)]),
        )

    def test_decision_record_shape_parity_and_meta_sidecar(self) -> None:
        specs = [(7, 0)]
        expected_keys = {
            "world_seed",
            "decision_index",
            "state",
            "state_text",
            "legal_actions",
            "selected_action",
            "agent",
            "after_state",
            "phase",
            "affordances",
            "policy_seed",
            "rollout_index",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "seed_7_r0.jsonl"
            run_streaming_rollouts(
                specs,
                _make_env_with_decisions(2),
                FakeStreamingAgent(),
                output_for=lambda ws, ri: root / f"seed_{ws}_r{ri}.jsonl",
                concurrency=1,
                max_decisions=200,
            )

            first_line = output_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(set(json.loads(first_line)), expected_keys)
            self.assertTrue(output_path.with_suffix(".meta.json").exists())

    def test_max_decisions_honored(self) -> None:
        results = run_streaming_rollouts(
            [(7, 0)],
            _make_env_with_decisions(10),
            FakeStreamingAgent(),
            concurrency=1,
            max_decisions=4,
        )

        self.assertEqual(len(results[0].decisions), 4)
        self.assertEqual(results[0].stopped_reason, "max_decisions")

    def test_simulator_error_isolated_to_one_slot(self) -> None:
        class ErrorEnv(FakeParallelEnv):
            def step(self, action_index: int) -> LegalAction:
                if self.world_seed == 99 and self.steps == 1:
                    raise RuntimeError("boom")
                return super().step(action_index)

        def make_env(world_seed: int) -> FakeParallelEnv:
            return ErrorEnv(world_seed=world_seed, decisions=4)

        results = run_streaming_rollouts(
            [(99, 0), (1, 0)],
            make_env,
            FakeStreamingAgent(),
            concurrency=2,
            max_decisions=200,
        )
        by_spec = {(r.world_seed, r.rollout_index): r for r in results}

        self.assertEqual(by_spec[(99, 0)].stopped_reason, "simulator_error")
        self.assertIsNotNone(by_spec[(99, 0)].error)
        self.assertEqual(by_spec[(99, 0)].error["phase"], "step")
        self.assertEqual(by_spec[(1, 0)].stopped_reason, "terminal")
        self.assertEqual(len(by_spec[(1, 0)].decisions), 4)


if __name__ == "__main__":
    unittest.main()
