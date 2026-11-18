from __future__ import annotations

import importlib.util
import unittest

from tests.support import requires_simulator


@requires_simulator
@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium extra absent")
class GymCheckerTests(unittest.TestCase):
    def test_numeric_features_include_the_selected_ascension(self):
        from sts_env.gymnasium import GymnasiumEnv
        self.assertEqual(GymnasiumEnv.FEATURE_NAMES[-1], "ascension")
        for ascension in (0, 20):
            with self.subTest(ascension=ascension):
                env = GymnasiumEnv(ascension=ascension)
                try:
                    observation, info = env.reset(seed=3)
                    self.assertEqual(observation["features"][-1], ascension)
                    self.assertEqual(observation["features"].shape, (16,))
                    self.assertEqual(info["observation"]["state"]["ascension"], ascension)
                finally:
                    env.close()

    def test_gymnasium_checker_accepts_fixed_space_with_masked_noops(self):
        from gymnasium.utils.env_checker import check_env
        from sts_env.gymnasium import GymnasiumEnv
        env = GymnasiumEnv(mode="full", max_steps=10000, step_timeout=30)
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
