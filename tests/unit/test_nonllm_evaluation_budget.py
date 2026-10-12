from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch
from scripts import train_nonllm
from sts_policy.curriculum import TaskSpec


class EvaluationBudgetTests(unittest.TestCase):
    def test_cli_evaluation_uses_declared_budget_while_ppo_keeps_short_rounds(self):
        class Model(torch.nn.Linear):
            def metadata(self):
                return {"size": "small", "context": "none"}
        observed = {}
        for command in ("evaluate", "ppo"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                model = Model(1, 1)
                worker = MagicMock()
                argv = ["train_nonllm.py", command, "--output", str(Path(directory)/'out.pt'),
                        "--checkpoint", str(Path(directory)/'fake.pt'), "--seconds", "600",
                        "--updates", "1", "--workers", "1", "--episodes", "1", "--device", "cpu"]
                result = {"transitions": [], "teacher_records": [], "episodes": [], "faults": []}
                with patch.object(sys, "argv", argv), \
                     patch.object(train_nonllm, "source_model", return_value=(model,None)), \
                     patch.object(train_nonllm, "tasks_for", return_value=[TaskSpec(seed=3)]), \
                     patch.object(train_nonllm, "CurriculumWorker", return_value=worker), \
                     patch.object(train_nonllm, "collect_rollouts", return_value=result) as collect, \
                     patch.object(train_nonllm, "update_ppo", return_value={"steps":0}), \
                     patch.object(train_nonllm, "save_checkpoint"), patch("builtins.print"):
                    train_nonllm.main()
                observed[command] = collect.call_args.kwargs["max_seconds"]
                worker.close.assert_called_once()
        self.assertGreater(observed["evaluate"], 120.)
        self.assertLessEqual(observed["evaluate"], 600.)
        self.assertEqual(observed["ppo"], 120.)
