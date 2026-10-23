"""Optional Gymnasium vectorization retains native isolation and cleanup."""
from __future__ import annotations

import os
import unittest

from tests.support import requires_simulator

try:
    import gymnasium as gym
    import numpy as np
    from sts_env.gymnasium import GymnasiumEnv
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    gym = None


if gym is not None:
    class DiagnosticGym(GymnasiumEnv):
        def native_pid(self):
            return self.env._process.pid


def make_vector_env():
    return DiagnosticGym(mode="combat", max_act=1, max_steps=8,
                         battle_simulations=50, step_timeout=10)


@unittest.skipIf(gym is None, "install the optional [gym] extra")
@requires_simulator
class AsyncGymTests(unittest.TestCase):
    def test_non_daemon_vector_workers_step_and_reap_native_children(self):
        vector = gym.vector.AsyncVectorEnv([make_vector_env, make_vector_env],
                                          daemon=False, context="spawn", shared_memory=False)
        outer = list(vector.processes)
        native = []
        try:
            observations, _ = vector.reset(seed=[3, 4])
            native = list(vector.call("native_pid"))
            self.assertEqual(len(set(native)), 2)
            for _ in range(2):
                actions = np.asarray([np.flatnonzero(mask)[0]
                                      for mask in observations["action_mask"]])
                observations, rewards, terminated, truncated, _ = vector.step(actions)
                self.assertEqual(rewards.shape, (2,))
                self.assertEqual(terminated.shape, (2,))
                self.assertEqual(truncated.shape, (2,))
        finally:
            vector.close(timeout=10)
        self.assertTrue(all(not process.is_alive() for process in outer))
        for pid in native:
            with self.subTest(native_pid=pid), self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)  # Signal zero only checks existence; sends no signal.


if __name__ == "__main__":
    unittest.main()
