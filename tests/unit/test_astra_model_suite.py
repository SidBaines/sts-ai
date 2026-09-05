from __future__ import annotations

from pathlib import Path
import unittest

from scripts.experiments.astra_model_suite import arm_log_path


class ModelSuiteLogPathTests(unittest.TestCase):
    def test_decimal_temperature_arms_have_distinct_logs(self):
        first = arm_log_path(Path("results/base_temp0.0"))
        second = arm_log_path(Path("results/base_temp0.7"))
        self.assertEqual(first, Path("results/base_temp0.0.log"))
        self.assertEqual(second, Path("results/base_temp0.7.log"))
        self.assertNotEqual(first, second)
