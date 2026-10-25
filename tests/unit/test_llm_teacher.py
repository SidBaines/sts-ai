"""The teacher subprocess with a stand-in teacher package; no PyTorch or simulator needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from sts_llm.teacher import NonLLMTeacher

# A minimal stand-in for the non-LLM workstream's sts_rl.fresh_study. Importing
# it fails if sts_llm/statistics.py shadows the standard-library module.
FAKE_STUDY = '''
import statistics
if "sts_llm" in str(getattr(statistics, "__file__", "")):
    raise ImportError("standard-library statistics is shadowed by " + statistics.__file__)
FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID = "filter", "dead", "context"

class StudyPolicy:
    def __init__(self, path):
        self.path = path

    def distribution(self, observation):
        indices = [action["index"] for action in observation["legal_actions"]]
        return indices, [statistics.fmean([1.0]) / len(indices)] * len(indices)
'''


def observation(number):
    return {"phase": "combat", "text": f"state {number}",
            "legal_actions": [{"index": 0, "description": "a"}, {"index": 4, "description": "b"}]}


class TeacherWorkerTests(unittest.TestCase):
    def test_worker_imports_stdlib_statistics_and_answers_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "sts_rl").mkdir(parents=True)
            (root / "src" / "sts_rl" / "__init__.py").write_text("")
            (root / "src" / "sts_rl" / "fresh_study.py").write_text(FAKE_STUDY)
            checkpoint = root / "champion.pt"
            checkpoint.write_bytes(b"not a torch checkpoint")
            with NonLLMTeacher(teacher_root=str(root), checkpoint=str(checkpoint), python=sys.executable,
                               timeout=60., batch_size=2) as teacher:
                results = teacher.distributions([observation(n) for n in range(5)])
                requests = teacher._counter
                metadata = teacher.metadata
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["legal_indices"], [0, 4])
        self.assertEqual(results[0]["probabilities"], [.5, .5])
        self.assertEqual(requests, 3)  # 2 + 2 + 1 observations, one deadline each.
        # The stand-in checkpoint records no versions, so the historical constants are used and labelled.
        self.assertEqual((metadata["encoder_version"], metadata["checkpoint_format_version"]),
                         ("public_candidate_v2", 1))
        self.assertEqual(set(metadata["version_source"].values()), {"fallback_constant"})
        self.assertIn("version_source_error", metadata)
        self.assertFalse(metadata["native_simulator_loaded"])
        json.dumps(metadata)


if __name__ == "__main__":
    unittest.main()
