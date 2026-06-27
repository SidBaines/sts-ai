"""Unit tests for RolloutSession + SessionRegistry with a fake env + fake model
agent (no simulator, no MLX). Exercises step/commit/history, sampling, branching
(replay), and persistence."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sts_ai.agent_factory import build_agent
from sts_ai.interactive.session import SessionRegistry
from sts_ai.schemas import AgentDecision, LegalAction


class FakeEnv:
    """A tiny deterministic linear game: at decision d the legal actions are
    `d# action {0,1}`; either advances to d+1 (action 1 costs 2 HP). Terminal
    after n decisions. Deterministic, so replay reproduces it exactly."""

    def __init__(self, *, world_seed, ascension=0, combat_control="llm", max_act=3,
                 battle_simulations=2000, n_decisions=6):
        self.world_seed = world_seed
        self.ascension = ascension
        self.combat_control = combat_control
        self.max_act = max_act
        self.battle_simulations = battle_simulations
        self.n = n_decisions
        self.pos = 0
        self.max_hp = 80
        self.cur_hp = 80

    def advance_to_decision(self):
        return 0

    def is_terminal(self):
        return self.pos >= self.n

    def phase(self):
        return "out_of_combat"

    def _actions(self):
        d = self.pos
        return [
            LegalAction(index=0, bits=d * 10 + 0, description=f"d{d} action 0"),
            LegalAction(index=1, bits=d * 10 + 1, description=f"d{d} action 1"),
        ]

    def legal_actions(self):
        return [] if self.is_terminal() else self._actions()

    def describe_state(self):
        return f"Your HP: {self.cur_hp}/{self.max_hp}\nDeck: {{Strike,}}\nPotions: none"

    def map_graph(self):
        return None

    def step(self, idx):
        actions = self._actions()
        selected = actions[idx]
        if idx == 1:
            self.cur_hp -= 2
        self.pos += 1
        return selected

    def summary(self):
        return {
            "world_seed": self.world_seed, "ascension": self.ascension, "act": 1,
            "floor": self.pos, "screen_state": "REWARDS", "room": "MONSTER",
            "outcome": "UNDECIDED", "cur_hp": self.cur_hp, "max_hp": self.max_hp,
            "gold": 0, "phase": "out_of_combat", "undefined_behavior_evoked": False,
            "done": self.is_terminal(),
        }

    @staticmethod
    def action_dict(action):
        return asdict(action)


class FakeModelAgent:
    name = "mlx"

    def __init__(self):
        self.framing = ""
        self.last_override = None
        self.draws = 0

    def reseed(self, seed):
        pass

    @property
    def config(self):
        return {"model_id": "fake", "framing": self.framing, "temperature": 0.2,
                "max_tokens": 10, "thinking": False}

    def choose_action(self, state_text, legal_actions, prompt_override=None):
        self.last_override = prompt_override
        self.draws += 1
        return AgentDecision(action_index=1, raw_response="m", reasoning="rz",
                             thinking="th", valid=True, completion_tokens=7, thinking_tokens=3)

    def stream_choose_action(self, state_text, legal_actions, prompt_override=None):
        self.last_override = prompt_override
        self.draws += 1
        for seg in ['{"reasoning"', ': "go", ', '"action_index": 1}']:
            yield seg
        return AgentDecision(action_index=1, raw_response='{"reasoning":"go","action_index":1}',
                             reasoning="go", valid=True)


def _fake_env_factory(**kw):
    return FakeEnv(**kw)


def _fake_agent_builder(name, **kwargs):
    if name in ("mlx", "vllm"):
        return FakeModelAgent()
    return build_agent(name)  # real scripted agents (first/random/heuristic), no model


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.reg = SessionRegistry(
            cache_dir=str(Path(self._tmp.name) / "interactive"),
            env_factory=_fake_env_factory,
            agent_builder=_fake_agent_builder,
        )

    def tearDown(self):
        self._tmp.cleanup()


class StepCommitTest(SessionTestBase):
    def test_user_step_commits_chosen_action(self):
        s = self.reg.create(world_seed=3, framing="F0")
        view = s.current_view()
        self.assertEqual(view["status"], "ok")
        self.assertEqual(view["decision_index"], 0)
        result = s.step("user", action_index=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decision"]["selected"]["description"], "d0 action 1")
        self.assertEqual(s.next_index, 1)
        self.assertEqual(s.stored.methods, ["user"])

    def test_heuristic_step_records_method(self):
        s = self.reg.create(world_seed=3)
        s.step("heuristic")
        self.assertEqual(s.stored.methods, ["heuristic"])
        self.assertEqual(s.history[0].agent["action_index"], 0)  # heuristic fallback picks 0

    def test_model_step_uses_model_decision(self):
        s = self.reg.create(world_seed=3)
        result = s.step("model")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(s.history[0].agent["action_index"], 1)  # FakeModelAgent picks 1
        self.assertEqual(s.history[0].agent["reasoning"], "rz")

    def test_sample_n_stops_at_terminal(self):
        s = self.reg.create(world_seed=3)  # FakeEnv default n=6
        out = s.sample_n("first", n=100)
        self.assertEqual(out["status"], "terminal")
        self.assertEqual(len(out["committed"]), 6)
        self.assertEqual(out["view"]["done"], True)

    def test_invalid_model_index_stops_without_executing(self):
        s = self.reg.create(world_seed=3)
        # force an out-of-range index via the user path validation
        result = s.step("user", action_index=99)
        self.assertEqual(result["status"], "agent_invalid")
        self.assertFalse(s.history[-1].action_executed)
        self.assertEqual(s.stored.stopped_reason, "agent_invalid")


class StreamStepTest(SessionTestBase):
    def test_stream_step_yields_tokens_then_commits(self):
        s = self.reg.create(world_seed=3)
        events = list(s.stream_step("model"))
        tokens = [e["text"] for e in events if e["type"] == "token"]
        done = [e for e in events if e["type"] == "done"]
        self.assertEqual("".join(tokens), '{"reasoning": "go", "action_index": 1}')
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["status"], "ok")
        self.assertEqual(s.next_index, 1)
        self.assertEqual(s.history[0].agent["action_index"], 1)

    def test_stream_step_falls_back_for_scripted_method(self):
        s = self.reg.create(world_seed=3)
        events = list(s.stream_step("heuristic"))
        self.assertEqual([e["type"] for e in events], ["done"])
        self.assertEqual(s.next_index, 1)


class PromptEditTest(SessionTestBase):
    def test_advanced_template_passes_prompt_override(self):
        s = self.reg.create(world_seed=3, framing="RISKY", use_advanced_template=True,
                             prompt_template="ADV {framing}\n{state_text}\n{action_lines}")
        s.step("model")
        agent = s._agents["model"]
        self.assertIsNotNone(agent.last_override)
        self.assertIn("ADV RISKY", agent.last_override)
        self.assertIn("d0 action 0", agent.last_override)

    def test_framing_only_path_no_override(self):
        s = self.reg.create(world_seed=3, framing="NEUTRALISH")
        s.step("model")
        self.assertIsNone(s._agents["model"].last_override)

    def test_preview_prompt_does_not_commit(self):
        s = self.reg.create(world_seed=3, framing="F")
        out = s.preview_prompt(framing="DIFFERENT")
        self.assertEqual(out["status"], "ok")
        self.assertIn("DIFFERENT", out["prompt"])
        self.assertEqual(s.stored.framing, "F")  # unchanged
        self.assertEqual(s.next_index, 0)  # nothing committed


class SampleCandidatesTest(SessionTestBase):
    def test_user_candidates_are_legal_actions(self):
        s = self.reg.create(world_seed=3)
        out = s.sample_candidates("user")
        self.assertEqual([c["description"] for c in out["candidates"]], ["d0 action 0", "d0 action 1"])
        self.assertEqual(s.next_index, 0)  # not committed

    def test_model_candidates_draw_k(self):
        s = self.reg.create(world_seed=3)
        out = s.sample_candidates("model", k=3)
        self.assertEqual(len(out["candidates"]), 3)
        self.assertEqual(s._agents["model"].draws, 3)
        self.assertEqual(s.next_index, 0)  # not committed


class BranchTest(SessionTestBase):
    def test_branch_replays_prefix_and_is_independent(self):
        parent = self.reg.create(world_seed=3, label="parent")
        parent.step("user", action_index=0)
        parent.step("user", action_index=0)
        parent.step("user", action_index=0)
        self.assertEqual(parent.next_index, 3)

        child = self.reg.branch(parent.session_id, 2, label="fork")
        # child env replayed 2 actions -> frontier is decision 2
        self.assertEqual(child.next_index, 2)
        self.assertEqual(child.env.pos, 2)
        self.assertEqual(child.stored.parent_id, parent.session_id)
        self.assertEqual(child.stored.branch_point, 2)

        # diverge in the child; parent untouched
        child.step("user", action_index=1)
        self.assertEqual(child.history[2].selected_action["description"], "d2 action 1")
        self.assertEqual(parent.next_index, 3)
        self.assertEqual(parent.history[2].selected_action["description"], "d2 action 0")

    def test_branch_at_frontier_clones(self):
        parent = self.reg.create(world_seed=3)
        parent.step("first")
        parent.step("first")
        child = parent.branch_at(parent.next_index)
        self.assertEqual(child.next_index, 2)
        self.assertEqual(child.env.pos, 2)


class PersistenceTest(SessionTestBase):
    def test_load_rehydrates_via_replay(self):
        s = self.reg.create(world_seed=3, framing="KEEPME")
        s.step("user", action_index=1)
        s.step("user", action_index=0)
        sid = s.session_id

        # drop from memory, reload from disk
        self.reg.evict(sid)
        reloaded = self.reg.load(sid)
        self.assertEqual(reloaded.stored.framing, "KEEPME")
        self.assertEqual(reloaded.next_index, 2)
        self.assertEqual(reloaded.env.pos, 2)  # replayed both actions
        # continuing works and matches the original next decision index
        view = reloaded.current_view()
        self.assertEqual(view["decision_index"], 2)

    def test_list_sessions_includes_lineage(self):
        parent = self.reg.create(world_seed=3, label="p")
        parent.step("first")
        self.reg.branch(parent.session_id, 1, label="c")
        summaries = {d["session_id"]: d for d in self.reg.list_sessions()}
        self.assertEqual(len(summaries), 2)
        child = [d for d in summaries.values() if d["parent_id"] == parent.session_id]
        self.assertEqual(len(child), 1)


if __name__ == "__main__":
    unittest.main()
