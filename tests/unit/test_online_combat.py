from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sts_policy.encoding import ACTION_DIM, STATE_DIM, EncodedObservation
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import Act3BossActorCritic, CompositeActorCritic, route_batch
from sts_rl.common import phase_forward
from sts_rl.online_combat import (
    ROLES,
    action_diagnostics,
    center_combat_returns,
    confirmed_combat_challenger,
    nominate_combat_challenger,
    samples_from_game,
    select_exploration,
    update_fighters,
    combat_optimizers,
    viable_combat_fit,
    temperature_for_exploration,
)


def row(seed, act, room, reward, finished=False, truncated=False, phase="combat"):
    menu = [{"index": 9}, {"index": 4}]
    g = np.random.default_rng(seed + act)
    x = g.normal(size=STATE_DIM).astype("float32")
    x[7] = int(phase == "combat")
    feat = EncodedObservation(
        x, g.normal(size=(2, ACTION_DIM)).astype("float32"), np.array([9, 4]), {}
    )
    return SimpleNamespace(
        observation={
            "phase": phase,
            "state": {"act": act, "room": room, "cur_hp": 80, "max_hp": 80},
            "legal_actions": menu,
        },
        encoded_observation=feat,
        world_seed=seed,
        action_position=0,
        action_index=9,
        behavior_log_prob=-0.7,
        behavior_mask=[True, True],
        reward=reward,
        terminated=finished,
        truncated=truncated,
    )


class OnlineCombatTests(unittest.TestCase):
    def test_matched_temperature_excludes_padding_and_equal_moves(self):
        logits = torch.tensor([[1., 0., 99.], [1., 1., 0.], [2., 2., 2.]])
        mask = torch.tensor([[True, True, False], [True]*3, [True]*3])
        result = temperature_for_exploration(logits, mask, .12)
        self.assertEqual(result["meaningful_menus"], 2)
        self.assertAlmostEqual(result["expected_rate"], .12, places=7)
        with self.assertRaises(ValueError):
            temperature_for_exploration(logits[-1:], mask[-1:], .12)

    def test_single_actor_stratifies_training_without_adding_a_deployment_head(self):
        from sts_policy.encoding import collate_observations
        from sts_rl.ooc_trainer import ooc_forward
        torch.manual_seed(4)
        model = CompositeActorCritic(*(CandidateActorCritic(size="small") for _ in range(2)))
        rows = {role: [] for role in ROLES}
        for seed in range(12):
            ep = [row(seed,1,"MONSTER",.1),row(seed,3,"BOSS",1 if seed%2 else -1,True)]
            parts=samples_from_game(ep,specialist=True)
            for role in ROLES:
                rows[role].extend(parts[role])
        samples=rows["combat"]+rows["boss_specialist"]
        batch=collate_observations([t.encoded_observation for t in samples])
        observations=[t.observation for t in samples]
        forward=ooc_forward(phase_forward(combat_temperature=.2,boss_temperature=.5))
        routed=route_batch(model,batch,observations,forward=forward)
        self.assertEqual(len(route_batch(model,batch,observations)),3)
        self.assertEqual(routed[-1].tolist(),[False]*12+[True]*12)
        raw=model(*batch)[0]
        logits=forward(model,routed)[0]
        self.assertTrue(torch.allclose(logits[:12],raw[:12]/.2,atol=1e-6))
        self.assertTrue(torch.allclose(logits[12:],raw[12:]/.5,atol=1e-6))
        for i,t in enumerate(samples):
            t.behavior_log_prob=float(logits[i].detach().log_softmax(0)[t.action_position])
        before={k:v.clone() for k,v in model.state_dict().items()}
        out=update_fighters(model,rows,combat_optimizers(model,3e-5),dict(combat=.2,boss_specialist=.5),
                            rate=3e-5,epochs=1,device="cpu",seed=11,minibatch=32)
        self.assertEqual(out["combat"]["samples"],24)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith("combat.")))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith("ooc.")))

    def test_returns_censoring_and_routing(self):
        ep = [
            row(3, 1, "BOSS", 0.05),
            row(3, 3, "Rooms.BOSS", 0),
            row(3, 3, "BOSS", 1, True, phase="out_of_combat"),
        ]
        data = samples_from_game(ep)
        self.assertEqual([v.returns for v in data["combat"]], [1.05])
        self.assertEqual([v.returns for v in data["boss_specialist"]], [1])
        self.assertEqual(data["boss_specialist"][0].observation["state"]["act"], 3)
        self.assertEqual(
            samples_from_game([row(3, 3, "BOSS", 0, True, True)]),
            dict.fromkeys(ROLES, []),
        )

    def test_single_fighter_learns_from_bosses_too(self):
        data = samples_from_game([row(43, 3, "BOSS", 1, True)], specialist=False)
        self.assertEqual(len(data["combat"]), 1)
        self.assertFalse(data["boss_specialist"])

    def test_baseline_never_uses_the_same_world(self):
        rows = []
        for seed, val in ((12, 1), (45, -1)):
            for _ in range(10):
                rows += samples_from_game([row(seed, 3, "BOSS", val, True)])[
                    "boss_specialist"
                ]
        center_combat_returns(rows)
        self.assertEqual({t.advantage for t in rows if t.world_seed == 12}, {2})
        self.assertEqual({t.advantage for t in rows if t.world_seed == 45}, {-2})

    def test_temperatures_cover_three_roles_and_only_legal_moves(self):
        torch.manual_seed(22)
        model = Act3BossActorCritic(
            *(CandidateActorCritic(size="small") for _ in range(3))
        )
        state = torch.randn(3, STATE_DIM)
        state[:, 7] = torch.tensor([1.0, 1.0, 0.0])
        actions = torch.randn(3, 3, ACTION_DIM)
        mask = torch.tensor([[1, 1, 0]] * 3, dtype=torch.bool)
        obs = [
            {"phase": phase, "state": {"act": act, "room": "BOSS"}}
            for phase, act in (("combat", 2), ("combat", 3), ("ooc", 3))
        ]
        batch = route_batch(model, (state, actions, mask), obs)
        raw = model(*batch)[0]
        got = phase_forward(combat_temperature=0.2, boss_temperature=0.4)(model, batch)[
            0
        ]
        for i, t in enumerate((0.2, 0.4, 1.0)):
            torch.testing.assert_close(got[i, :2], raw[i, :2] / t)
        self.assertTrue((got[:, 2] == -1e9).all())

    def test_updates_both_fighters_only(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        model = Act3BossActorCritic(
            *(CandidateActorCritic(size="small") for _ in range(3))
        )
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rows = {r: [] for r in ROLES}
        for i in range(32):
            sample = samples_from_game(
                [
                    row(i, 2, "BOSS", 0.05),
                    row(i, 3, "BOSS", 1.0 if i % 3 else -1.0, True),
                ]
            )
            for r in ROLES:
                rows[r].extend(sample[r])
        opt = combat_optimizers(model, 3e-5)
        stats = update_fighters(
            model,
            rows,
            opt,
            dict.fromkeys(ROLES, 1.0),
            rate=3e-5,
            epochs=1,
            device="cpu",
            seed=4,
            minibatch=32,
        )
        for r in ROLES:
            self.assertGreater(stats[r]["steps"], 0)
            self.assertTrue(
                any(
                    not torch.equal(v, before[k])
                    for k, v in model.state_dict().items()
                    if k.startswith(r + ".")
                )
            )
            diag = action_diagnostics(getattr(model, r), rows[r], [0.05, 0.5])
            self.assertLessEqual(diag["expected_rates"][0], diag["expected_rates"][1])
        self.assertTrue(
            all(
                torch.equal(v, before[k])
                for k, v in model.state_dict().items()
                if k.startswith("ooc.") or ".critic." in k
            )
        )

    def test_selection_checks_drops_unknowns_and_both_fighters(self):
        base = {s: {"win": bool(s % 2), "act3": True} for s in range(100)}
        other = {s: dict(x) for s, x in base.items()}
        for s in (1, 3, 5, 7, 9, 11):
            other[s]["win"] = None
        choice, stats = select_exploration(
            base, {"gentle": base, "moderate": other}, ["moderate", "gentle"]
        )
        self.assertEqual(choice, "gentle")
        self.assertEqual(stats["moderate"]["win"]["bounds"], [-0.06, 0.0])
        self.assertRaises(ValueError, select_exploration, base, {"bad": {}}, ["bad"])
        move = {r: dict(changed_rate=0.01, mean_kl=0.02) for r in ROLES}
        self.assertTrue(viable_combat_fit(move))
        move["boss_specialist"]["changed_rate"] = 0
        self.assertFalse(viable_combat_fit(move))

    def test_expanded_nomination_requires_a_distinct_eligible_fit_and_uses_full_cohort(
        self,
    ):
        def row(gain, epochs=2, rate=0.00003, target=0.12, eligible=True):
            return dict(
                setting=dict(epochs=epochs, rate=rate, target=target),
                eligible=eligible,
                comparison={
                    "difference_pp": 99,
                    "full_cohort_gain_bounds_pp": [gain, 99],
                },
            )

        results = {
            "original": row(0),
            "ineligible": row(9, eligible=False),
            "small": row(0.1),
            "four": row(0.2, epochs=4),
            "high_rate": row(0.2, rate=0.00006),
            "distant": row(0.2, target=0.22),
            "chosen": row(0.2, target=0.05),
        }
        self.assertEqual(nominate_combat_challenger(results, "original"), "chosen")
        self.assertIsNone(
            nominate_combat_challenger(
                {"original": row(0), "bad": row(50, eligible=False)}, "original"
            )
        )

    def test_expanded_confirmation_requires_both_conservative_directions(self):
        for screen, lower, expected in (
            (1, 0.1, True),
            (0, 1, False),
            (1, 0, False),
            (-1, 0.1, False),
            (1, -0.1, False),
        ):
            self.assertEqual(
                confirmed_combat_challenger(
                    {"full_cohort_gain_bounds_pp": [screen, 3]},
                    {"ci95_pp": [1, 5], "worst_case_ci95_pp": [lower, 5]},
                ),
                expected,
            )
