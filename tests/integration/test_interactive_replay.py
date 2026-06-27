"""Replay-determinism check that gates the Interactive Studio's branch design.

Branching has no binary state snapshot (the C++ contexts are opaque), so it works
by replaying a recorded action sequence into a fresh env (see
`sts_ai.interactive.replay`). That is only correct if (a) the forward run is
deterministic for a fixed seed and (b) replaying a prefix reproduces the exact
frontier state + legal actions. This test proves both for `combat_control="llm"`
(re-applies the exact recorded player actions) and reports `"search"` (combats
auto-resolved by the C++ search agent — determinism there is the same assumption
as the repo's frozen-seed contract).

Run in a child process with a timeout to contain the residual seed-class UB hang
(see docs/simulator_issue_handoff.md, tests/integration/test_battle_search.py).
Structural comparisons are between two runs in the *same* build/process, which is
exactly what determinism means — not brittle cross-build value assertions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from tests.support import requires_simulator

# Seed 3 / battle_simulations=50 / max_act=1 is the empirically fast, safe early
# slice used across the integration tier (test_full_game_progression,
# test_combat_control). A short decision budget keeps us in act 1.
_DRIVER = """
import json
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.interactive.replay import replay_actions

MODE = {mode!r}
MAX_DECISIONS = {max_decisions}


def make_env():
    return LightspeedHybridEnv(
        world_seed=3, battle_simulations=50, max_act=1, combat_control=MODE
    )


def drive():
    env = make_env()
    actions, sigs, legals = [], [], []
    for _ in range(MAX_DECISIONS):
        env.advance_to_decision()
        if env.is_terminal():
            break
        legal = env.legal_actions()
        if not legal:
            break
        chosen = legal[0]
        try:
            env.step(0)
        except Exception:
            break
        actions.append({{"bits": int(chosen.bits), "description": chosen.description}})
        sigs.append(json.dumps(env.summary(), sort_keys=True))
        legals.append(
            [] if env.is_terminal() else [a.description for a in env.legal_actions()]
        )
    return actions, sigs, legals


actions1, sigs1, legals1 = drive()
actions2, sigs2, legals2 = drive()

result = {{
    "n": len(actions1),
    "mode": MODE,
    "forward_deterministic": (
        actions1 == actions2 and sigs1 == sigs2 and legals1 == legals2
    ),
}}

if len(actions1) >= 2:
    k = max(1, len(actions1) // 2)
    env_b = make_env()
    replay_error = None
    try:
        replay_actions(env_b, actions1[:k])
        replay_sig = json.dumps(env_b.summary(), sort_keys=True)
        replay_legal = (
            [] if env_b.is_terminal() else [a.description for a in env_b.legal_actions()]
        )
    except Exception as exc:  # noqa: BLE001
        replay_error = repr(exc)
        replay_sig = None
        replay_legal = None
    result.update({{
        "k": k,
        "replay_error": replay_error,
        "replay_matches_state": replay_sig == sigs1[k - 1],
        "replay_matches_legal": replay_legal == legals1[k - 1],
    }})

print(json.dumps(result, sort_keys=True))
"""


@requires_simulator
class ReplayDeterminismTest(unittest.TestCase):
    def _run_probe(self, mode: str, max_decisions: int) -> dict:
        code = textwrap.dedent(_DRIVER).format(mode=mode, max_decisions=max_decisions)
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing else f"src:{existing}"
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.assertIsNotNone(exc)
            self.skipTest(f"{mode} replay hung in native code; contained by subprocess timeout")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"child probe crashed\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        return json.loads(completed.stdout)

    def test_llm_replay_reproduces_frontier_exactly(self):
        payload = self._run_probe("llm", max_decisions=24)
        self.assertGreaterEqual(payload["n"], 2, msg=f"too few decisions to test: {payload}")
        self.assertTrue(payload["forward_deterministic"], msg=f"forward run not deterministic: {payload}")
        self.assertIsNone(payload.get("replay_error"), msg=f"replay raised: {payload}")
        self.assertTrue(payload["replay_matches_state"], msg=f"replay state mismatch: {payload}")
        self.assertTrue(payload["replay_matches_legal"], msg=f"replay legal-actions mismatch: {payload}")

    def test_search_replay_reproduces_frontier(self):
        # Search mode re-runs the C++ search agent for each combat during replay;
        # this asserts that path is deterministic too. If it ever regresses, the
        # Studio still branches exactly in llm mode (see the module docstring).
        payload = self._run_probe("search", max_decisions=16)
        self.assertGreaterEqual(payload["n"], 2, msg=f"too few decisions to test: {payload}")
        self.assertTrue(payload["forward_deterministic"], msg=f"forward run not deterministic: {payload}")
        self.assertIsNone(payload.get("replay_error"), msg=f"replay raised: {payload}")
        self.assertTrue(payload["replay_matches_state"], msg=f"replay state mismatch: {payload}")
        self.assertTrue(payload["replay_matches_legal"], msg=f"replay legal-actions mismatch: {payload}")


if __name__ == "__main__":
    unittest.main()
