import importlib.util
import json
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None
spec = importlib.util.spec_from_file_location(
    "critic_diagnosis",
    Path(__file__).resolve().parents[2] / "scripts/rl/joint_critic_diagnosis.py",
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@unittest.skipIf(np is None, "Optional NumPy dependency absent")
class CriticDiagnosticMetricTests(unittest.TestCase):
    def test_weighted_scores_match_known_cases_and_serialize_numpy_inputs(self):
        y = np.array([0, 1, 1], np.float32)
        w = np.array([2, 1, 1], np.float32)
        result = diagnosis.scores(y, y.copy(), w)
        self.assertEqual(result["auc"], 1)
        self.assertEqual(result["brier"], 0)
        self.assertEqual(result["return_mean"], 0.5)
        self.assertEqual(result["explained_variance"], 1)
        json.dumps(result, allow_nan=False)
        tied = diagnosis.scores(y, np.full(3, 0.5, np.float32), w)
        self.assertEqual(tied["auc"], 0.5)
        self.assertEqual(tied["brier"], 0.25)
        self.assertEqual(tied["explained_variance"], 0)

    def test_single_class_uncertainty_metrics_are_null(self):
        result = diagnosis.scores(
            np.zeros(3, np.float32),
            np.full(3, 0.03, np.float32),
            np.ones(3, np.float32),
        )
        self.assertIsNone(result["auc"])
        self.assertIsNone(result["explained_variance"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
