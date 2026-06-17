from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sts_ai.parallel_rollout import run_parallel_rollouts
from sts_ai.schemas import AgentDecision, LegalAction, RolloutResult
from sts_ai.seeding import derive_batch_seed, derive_policy_seed


@dataclass
class FakeParallelEnv:
    world_seed: int
    decisions: int = 4
    steps: int = 0

    def advance_to_decision(self) -> int:
        return 0

    def is_terminal(self) -> bool:
        return self.steps >= self.decisions

    def phase(self) -> str:
        return "out_of_combat"

    def legal_actions(self) -> list[LegalAction]:
        return [
            LegalAction(index=index, bits=index, description=f"action {index}")
            for index in range(4)
        ]

    def describe_state(self) -> str:
        return f"seed={self.world_seed} step={self.steps}"

    def map_graph(self) -> None:
        return None

    def step(self, action_index: int) -> LegalAction:
        action = self.legal_actions()[action_index]
        self.steps += 1
        return action

    def summary(self) -> dict[str, Any]:
        return {
            "world_seed": self.world_seed,
            "step": self.steps,
            "done": self.is_terminal(),
        }

    @staticmethod
    def action_dict(action: LegalAction) -> dict[str, Any]:
        return {
            "index": action.index,
            "bits": action.bits,
            "description": action.description,
        }


class ReseededBatchAgent:
    name = "reseeded-batch"

    def __init__(self) -> None:
        self.rng: Optional[random.Random] = None
        self.reseed_calls: list[int] = []

    def reseed(self, policy_seed: int) -> None:
        self.reseed_calls.append(policy_seed)
        self.rng = random.Random(policy_seed)

    def choose_actions_batch(
        self,
        items: list[tuple[str, list[LegalAction]]],
    ) -> list[AgentDecision]:
        if self.rng is None:
            raise AssertionError("choose_actions_batch called before reseed")

        decisions: list[AgentDecision] = []
        for position, (_, legal_actions) in enumerate(items):
            # Keep batch positions in separate action ranges so sibling rollouts
            # are guaranteed to have distinct action streams while still drawing
            # each choice from the reseeded RNG.
            block = max(1, len(legal_actions) // max(1, len(items)))
            low = min(position * block, len(legal_actions) - 1)
            high = min(low + block, len(legal_actions))
            action_index = low + self.rng.randrange(max(1, high - low))
            decisions.append(
                AgentDecision(
                    action_index=action_index,
                    raw_response=f"rng batch position {position}",
                )
            )
        return decisions


class RetryBatchAgent:
    name = "retry-batch"
    max_retries = 1

    def __init__(self, always_invalid: bool = False) -> None:
        self.always_invalid = always_invalid
        self.retry_flags_seen: list[list[bool]] = []

    def reseed(self, policy_seed: int) -> None:
        return None

    def choose_actions_batch(
        self,
        items: list[tuple[str, list[LegalAction]]],
        retry_flags: list[bool] | None = None,
    ) -> list[AgentDecision]:
        flags = retry_flags or [False] * len(items)
        self.retry_flags_seen.append(list(flags))
        decisions: list[AgentDecision] = []
        for retry in flags:
            valid = retry and not self.always_invalid
            decisions.append(
                AgentDecision(
                    action_index=0,
                    raw_response="ok" if valid else "bad",
                    valid=valid,
                    metadata={} if valid else {"error": "no json object"},
                )
            )
        return decisions


def _make_env(world_seed: int) -> FakeParallelEnv:
    return FakeParallelEnv(world_seed=world_seed)


def _action_sequences(results: list[RolloutResult]) -> dict[tuple[int, int], list[int]]:
    return {
        (result.world_seed, result.rollout_index): [
            decision.selected_action["index"] for decision in result.decisions
        ]
        for result in results
    }


class ParallelRolloutSpecTest(unittest.TestCase):
    def test_same_specs_and_batch_size_rerun_identically_after_batch_reseed(self) -> None:
        specs = [(7, 0), (7, 1)]
        agent = ReseededBatchAgent()

        first = run_parallel_rollouts(
            specs, _make_env, agent, batch_size=2, max_decisions=4
        )
        second = run_parallel_rollouts(
            specs, _make_env, agent, batch_size=2, max_decisions=4
        )

        self.assertEqual(_action_sequences(first), _action_sequences(second))
        self.assertNotEqual(
            _action_sequences(first)[(7, 0)],
            _action_sequences(first)[(7, 1)],
        )
        expected_batch_seeds = [
            derive_batch_seed([(7, 0, decision_index), (7, 1, decision_index)])
            for decision_index in range(4)
        ]
        self.assertEqual(agent.reseed_calls[:4], expected_batch_seeds)
        self.assertEqual(agent.reseed_calls[4:], expected_batch_seeds)

    def test_same_world_seed_specs_do_not_collide(self) -> None:
        specs = [(11, 0), (11, 1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = run_parallel_rollouts(
                specs,
                _make_env,
                ReseededBatchAgent(),
                output_for=lambda ws, ri: root / f"seed_{ws}_r{ri}.jsonl",
                batch_size=2,
                max_decisions=1,
            )

            self.assertEqual([(r.world_seed, r.rollout_index) for r in results], specs)
            self.assertEqual(len(results), 2)
            self.assertEqual(
                [r.policy_seed for r in results],
                [derive_policy_seed(11, 0), derive_policy_seed(11, 1)],
            )
            for world_seed, rollout_index in specs:
                path = root / f"seed_{world_seed}_r{rollout_index}.jsonl"
                self.assertTrue(path.exists())
                self.assertTrue(path.with_suffix(".meta.json").exists())
                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["world_seed"], world_seed)
                self.assertEqual(records[0]["rollout_index"], rollout_index)
                self.assertTrue(records[0]["action_executed"])

    def test_batched_retry_succeeds_before_exhaustion(self) -> None:
        agent = RetryBatchAgent()
        results = run_parallel_rollouts(
            [(7, 0)],
            lambda ws: FakeParallelEnv(world_seed=ws, decisions=2),
            agent,
            batch_size=1,
            max_decisions=4,
            max_retries=1,
        )

        self.assertEqual(results[0].stopped_reason, "terminal")
        self.assertEqual([d.agent["retries"] for d in results[0].decisions], [1, 1])
        self.assertTrue(all(d.action_executed for d in results[0].decisions))
        self.assertEqual(agent.retry_flags_seen[:2], [[False], [True]])

    def test_batched_retry_exhaustion_records_invalid_and_stops(self) -> None:
        agent = RetryBatchAgent(always_invalid=True)
        results = run_parallel_rollouts(
            [(7, 0)],
            lambda ws: FakeParallelEnv(world_seed=ws, decisions=2),
            agent,
            batch_size=1,
            max_decisions=4,
            max_retries=1,
        )

        self.assertEqual(results[0].stopped_reason, "agent_invalid")
        self.assertEqual(len(results[0].decisions), 1)
        record = results[0].decisions[0]
        self.assertEqual(record.agent["retries"], 1)
        self.assertFalse(record.agent["valid"])
        self.assertFalse(record.action_executed)
        self.assertEqual(record.selected_action, {})


if __name__ == "__main__":
    unittest.main()
