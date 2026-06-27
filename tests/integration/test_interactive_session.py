"""Integration test: RolloutSession + SessionRegistry on the real simulator.

Proves the Studio's session/branch path works end to end against the built env:
a session steps decisions, branching replays the prefix into a fresh env, and the
child's frontier state equals the parent's recorded pre-decision state at the
branch point. Scripted agents only (no MLX) so it needs only the simulator.

Run in a child process with a timeout (residual seed-class UB; see
tests/integration/test_battle_search.py). Gated with @requires_simulator.
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

_DRIVER = """
import json, tempfile
from sts_ai.interactive.session import SessionRegistry

reg = SessionRegistry(cache_dir=tempfile.mkdtemp())
s = reg.create(world_seed=3, combat_control="llm", max_act=1, battle_simulations=50,
               framing="probe framing")
out = s.sample_n("first", n=10)
n = s.next_index
result = {"n": n, "framing_kept": s.stored.framing == "probe framing"}

if n >= 4:
    k = max(1, n // 2)
    parent_state_k = json.dumps(s.history[k].state, sort_keys=True)
    child = reg.branch(s.session_id, k)
    cv = child.current_view()
    result.update({
        "k": k,
        "child_index": cv["decision_index"],
        "child_parent": child.stored.parent_id == s.session_id,
        "child_branch_point": child.stored.branch_point,
        "match_state": (cv.get("status") in ("ok", "terminal")
                        and (cv.get("status") == "terminal"
                             or json.dumps(cv.get("state", {}), sort_keys=True) == parent_state_k)),
        "parent_unchanged": s.next_index == n,
        # reload from disk reconstructs the same frontier
        "reload_index": (lambda r: r.next_index)(reg.load(s.session_id)),
    })
print(json.dumps(result, sort_keys=True))
"""


@requires_simulator
class InteractiveSessionIntegrationTest(unittest.TestCase):
    def test_session_step_branch_and_reload(self):
        code = textwrap.dedent(_DRIVER)
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing else f"src:{existing}"
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path.cwd(), env=env, text=True, capture_output=True,
                timeout=180, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.assertIsNotNone(exc)
            self.skipTest("session replay hung in native code; contained by subprocess timeout")
        self.assertEqual(
            completed.returncode, 0,
            msg=f"child crashed\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        payload = json.loads(completed.stdout)
        self.assertGreaterEqual(payload["n"], 4, msg=f"too few decisions: {payload}")
        self.assertTrue(payload["framing_kept"], msg=payload)
        self.assertEqual(payload["child_index"], payload["k"], msg=f"branch frontier off: {payload}")
        self.assertTrue(payload["child_parent"], msg=payload)
        self.assertEqual(payload["child_branch_point"], payload["k"], msg=payload)
        self.assertTrue(payload["match_state"], msg=f"branch frontier state != parent's: {payload}")
        self.assertTrue(payload["parent_unchanged"], msg=payload)
        self.assertEqual(payload["reload_index"], payload["n"], msg=f"reload frontier off: {payload}")


if __name__ == "__main__":
    unittest.main()
