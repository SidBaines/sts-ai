"""Integration test that the default full-game cap can progress past Act 1.

Gated with @requires_simulator and run in a child process with a timeout so a
native hybrid-search hang cannot wedge the parent suite.
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


@requires_simulator
class FullGameProgressionTest(unittest.TestCase):
    def test_heuristic_hybrid_rollout_reaches_act_2(self):
        code = textwrap.dedent(
            """
            import json
            from sts_ai.agent_factory import build_agent
            from sts_ai.lightspeed import LightspeedHybridEnv
            from sts_ai.rollout import run_rollout

            # Seed 3 was chosen empirically from the frozen smoke seeds; with the
            # heuristic agent and hybrid combat it reliably clears Act 1.
            env = LightspeedHybridEnv(
                world_seed=3,
                battle_simulations=100,
                max_act=3,
                combat_control="search",
            )
            result = run_rollout(env, build_agent("heuristic"), max_decisions=1500)
            print(json.dumps({
                "decisions": len(result.decisions),
                "stopped_reason": result.stopped_reason,
                "terminal_state": result.terminal_state,
                "error": result.error,
            }, sort_keys=True))
            """
        )
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"

        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                "heuristic full-game progression child timed out "
                f"stdout={exc.stdout} stderr={exc.stderr}"
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"child rollout failed\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertIsNone(payload["error"], msg=f"rollout hit an error: {payload}")
        self.assertNotEqual(payload["stopped_reason"], "max_decisions", msg=f"rollout truncated: {payload}")

        terminal = payload["terminal_state"]
        self.assertGreaterEqual(terminal["act"], 2, msg=f"did not progress past Act 1: {payload}")


if __name__ == "__main__":
    unittest.main()
