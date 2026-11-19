from pathlib import Path
import tempfile
import unittest

import torch

from sts_policy.encoding import ENCODER_VERSION
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import Act3BossActorCritic, policy_from_payload
from sts_policy.training import CHECKPOINT_VERSION
from sts_rl.boss_refits import admits_candidate, confirmed_schedule_advantage, package, primary_nominee


class RefitTests(unittest.TestCase):
    def test_strict_gates_and_conservative_selection(self):
        for fight, bounds, expected in (
            (0, None, False),
            (-0.1, [1, 1], False),
            (0.1, [-0.2, 0.3], False),
            (0.1, [0, 0.2], False),
            (0.1, [0.01, 0.02], True),
        ):
            self.assertEqual(admits_candidate(fight, bounds), expected)
        p = dict(a="early", b="spaced", full_cohort_gain_bounds_pp=[-0.1, 0.1])
        self.assertEqual(primary_nominee(p, "early", "spaced"), "early")
        self.assertFalse(
            confirmed_schedule_advantage(p, dict(a="spaced", b="early", worst_case_ci95_pp=[0.5, 1]))
        )
        for low, high, nominee in ((0.1, 0.2, "spaced"), (-0.4, -0.3, "early")):
            p["full_cohort_gain_bounds_pp"] = [low, high]
            self.assertEqual(primary_nominee(p, "early", "spaced"), nominee)
            other = p["a"] if nominee == p["b"] else p["b"]
            repeat = dict(a=other, b=nominee, worst_case_ci95_pp=[0.01, 0.5])
            self.assertTrue(confirmed_schedule_advantage(p, repeat))
            repeat["worst_case_ci95_pp"] = [0, 0.5]
            self.assertFalse(confirmed_schedule_advantage(p, repeat))

    def test_replace_existing_specialist_keeps_strategy_and_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heads = [CandidateActorCritic(size="small") for _ in range(4)]
            active = Act3BossActorCritic(*heads[:3])

            def payload(model):
                return dict(
                    format_version=CHECKPOINT_VERSION,
                    model_config=model.metadata(),
                    metadata={"encoder_version": ENCODER_VERSION},
                    model_state=model.state_dict(),
                )

            original = payload(active)
            original.update(
                iteration=8,
                steps_total=128,
                optimizer_state={"step": 8},
                rng_state={"marker": 5},
                trainer_state={"return_scale": 1.4},
            )
            student = payload(heads[3])
            student["metadata"]["phase"] = "combat"
            torch.save(original, root / "active.pt")
            torch.save(student, root / "head.pt")
            package(root / "active.pt", root / "head.pt", root / "new.pt")
            result = torch.load(root / "new.pt", weights_only=False)
            restored = policy_from_payload(result)
            for key in ("iteration", "steps_total", "optimizer_state", "rng_state", "trainer_state"):
                self.assertEqual(original[key], result[key])
            for name, expected in (
                ("combat", heads[0]),
                ("ooc", heads[1]),
                ("boss_specialist", heads[3]),
            ):
                actual = getattr(restored, name)
                self.assertTrue(
                    all(torch.equal(actual.state_dict()[k], v) for k, v in expected.state_dict().items())
                )
