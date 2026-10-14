from __future__ import annotations
import random
import json
from pathlib import Path
import unittest
from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec
from sts_policy.workers import CurriculumWorker
from tests.support import requires_simulator


@requires_simulator
class FastReplayTests(unittest.TestCase):
    def paired(self, spec):
        with CurriculumWorker({"fast_observations": True}) as slow, CurriculumWorker({"fast_observations": True, "fast_replay": True}) as fast:
            obs_s, info_s = slow.reset(spec)
            obs_f, info_f = fast.reset(spec)
            self.assertEqual(obs_s, obs_f)
            self.assertEqual(info_s, info_f)
            action = slow.teacher_action(spec.battle_simulations)
            self.assertEqual(action, fast.teacher_action(spec.battle_simulations))
            result_s = slow.step(action)
            result_f = fast.step(action)
            self.assertEqual(result_s, result_f)

    def test_generated_mixed_trajectories_preserve_reset_and_step(self):
        specs = []
        for seed in range(1000, 1004):
            with CurriculumWorker({"fast_observations": True}) as worker:
                rng = random.Random(seed)
                obs, info = worker.reset(TaskSpec(seed=seed, kind="run"))
                for step in range(1500):
                    if step in (8, 31, 62, 95, 127):
                        specs.append(TaskSpec(seed=seed, kind="run", prefix=tuple(info["action_prefix"]),
                                              ooc_previews=tuple(info["ooc_previews"])))
                    action = worker.teacher_action(50)
                    if rng.random() < (0.05 if obs["phase"] == "out_of_combat" else 0.02):
                        action = rng.randrange(len(obs["legal_actions"]))
                    obs, _, terminated, truncated, info = worker.step(action)
                    if terminated or truncated:
                        break
        self.assertGreaterEqual(len(specs), 12)
        for spec in specs:
            self.paired(spec)

    def test_invalid_replay_prefix_fails_in_both_paths(self):
        # The old 2005 fixture played Discovery and is intentionally healthy
        # after 0.1.2. Preserve fail-closed replay coverage with an illegal index.
        spec = TaskSpec(seed=2005, kind="run", prefix=(999999,), ooc_previews=(False,))
        messages = []
        for fast in (False, True):
            with CurriculumWorker({"fast_observations": True, "fast_replay": fast}) as worker:
                with self.assertRaises(SimulatorError) as raised:
                    worker.reset(spec)
                messages.append(str(raised.exception))
                self.assertIsNone(worker._process)
        self.assertEqual(messages[0], messages[1])
        self.assertIn('IndexError: list index out of range', messages[0])

    def test_recorded_random_first_bank_and_twenty_later_act_starts(self):
        root = Path(__file__).resolve().parents[2] / "data/nonllm_curriculum"
        # Persisted fixtures must match the active release; never remove their
        # provenance to replay an older native bank under a corrected backend.
        revised = root / "teacher_act1_200_native011_seed10000/battles.jsonl"
        first = revised if revised.exists() else root / "teacher_act1_200_seed10000/battles.jsonl"
        later = root / "panels_native011" if revised.exists() else root / "teacher_act3_200_seed11000/stage_subsets"
        if not first.exists() or not later.exists():
            self.skipTest("Generated curriculum banks absent; synthetic native paired cases still run")
        first_rows = [json.loads(line) for line in first.read_text().splitlines()]
        for row in random.Random(18001).sample(first_rows, 10):
            self.paired(TaskSpec.from_dict(row))
        for act in (2, 3):
            filename = f"train_act{act}.jsonl" if revised.exists() else f"battles_act{act}.jsonl"
            rows = [json.loads(line) for line in (later / filename).read_text().splitlines()]
            for index in list(range(0, len(rows), max(1, len(rows)//10)))[:10]:
                self.paired(TaskSpec.from_dict(rows[index]))
