import copy
import json
import random
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch

    from sts_policy.model import CandidateActorCritic
    from sts_policy.model_agent import CompositeActorCritic
    from sts_rl.joint_sparse import (
        BinaryFuture,
        JointConfig,
        WinCritics,
        actor_update,
        classify_episode,
        compact_chunk,
        critic_update,
        freeze_legacy_critics,
        identity_state,
        likelihood_error,
        mixture_logits,
        phase_distribution,
        validate_resume_identity,
    )
except ModuleNotFoundError as error:
    if error.name not in {"torch", "numpy"}:
        raise
    torch = None


@unittest.skipIf(torch is None, "Optional RL dependencies absent")
class JointSparseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.cfg = JointConfig(
            out="/tmp/test", checkpoint="/tmp/unused", microbatch=4, minibatch=8
        )
        self.actor = CompositeActorCritic(
            CandidateActorCritic("small", "none"), CandidateActorCritic("small", "none")
        )
        freeze_legacy_critics(self.actor)
        self.ref = copy.deepcopy(self.actor)
        self.critics = WinCritics()
        self.rng = random.Random(5)
        self.rows = []
        for phase in ("combat", "ooc"):
            for i in range(12):
                state = np.zeros(256, np.float32)
                state[7] = phase == "combat"
                state[1] = i / 12
                actions = np.zeros((3, 128), np.float32)
                actions[:, 0] = [0, 1, -1]
                r = dict(
                    state=state,
                    actions=actions,
                    position=i % 3,
                    phase=phase,
                    act=i % 3 + 1,
                    seed=100 + i,
                    return_=float(i % 2),
                )
                r["return"] = r.pop("return_")
                r["value"] = 0.03
                r["advantage"] = r["return"] - 0.03
                dist, _, _ = phase_distribution(
                    getattr(self.actor, phase), [r], self.cfg, phase, "cpu"
                )
                r["logp"] = float(
                    dist.log_prob(torch.tensor([r["position"]])).detach()[0]
                )
                self.rows.append(r)
        self.opts = {
            p: torch.optim.AdamW(
                [v for v in getattr(self.actor, p).parameters() if v.requires_grad],
                lr=1e-5,
                weight_decay=0,
            )
            for p in ("combat", "ooc")
        }

    def test_binary_terminal_only_and_quarantine(self):
        for terminated, truncated, outcome, want in [
            (True, False, "victory", 1),
            (True, False, "defeat", 0),
            (False, False, "victory", 0),
            (True, True, "victory", 0),
        ]:
            f = Future()
            f.set_result(({}, -1, terminated, truncated, {"task_outcome": outcome}))
            self.assertEqual(BinaryFuture(f, "step").result()[1], want)
        self.assertEqual(
            classify_episode(
                dict(terminated=True, truncated=True, task_outcome="defeat")
            ),
            "censored",
        )
        self.assertEqual(
            classify_episode(dict(terminated=True, task_outcome="fault")), "invalid"
        )

    def test_mixture_mask_and_likelihood(self):
        logits = torch.tensor([[100.0, 0.0, 50.0]])
        mask = torch.tensor([[True, True, False]])
        q = mixture_logits(logits, mask, 0.5, 0.02).softmax(-1)
        self.assertAlmostEqual(q[0, 1].item(), 0.01, places=6)
        self.assertEqual(q[0, 2].item(), 0)
        with self.assertRaises(ValueError):
            mixture_logits(logits, mask & False, 1, 0)
        self.assertLess(likelihood_error(self.actor, self.rows, self.cfg), 1e-5)

    def test_mc_target_and_portal_boss(self):
        def transition(seed):
            return SimpleNamespace(
                world_seed=seed,
                observation={
                    "phase": "combat",
                    "state": {"act": 3, "floor": 40, "room": "BOSS", "cur_hp": 22},
                    "legal_actions": [
                        {"description": "Play Strike"},
                        {"description": "End turn"},
                    ],
                },
                behavior_mask=[True, True],
                action_position=0,
                action_index=9,
                encoded_observation=SimpleNamespace(
                    action_indices=[9, 10],
                    state_features=np.zeros(256),
                    action_features=np.zeros((2, 128)),
                ),
                behavior_log_prob=-0.2,
                behavior_value=0.3,
            )

        eps = [
            dict(
                world_seed=1,
                terminated=True,
                truncated=False,
                task_outcome="victory",
                reward=1,
            ),
            dict(
                world_seed=2,
                terminated=False,
                truncated=True,
                task_outcome="",
                reward=0,
            ),
        ]
        rows, episodes, *_ = compact_chunk(
            dict(transitions=[transition(1), transition(2)], episodes=eps, faults=[])
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["return"], 1)
        self.assertAlmostEqual(rows[0]["advantage"], 0.7)
        self.assertEqual(episodes[0]["boss_entries"]["3"]["floor"], 40)

    def test_critic_cannot_change_actor(self):
        before = identity_state(self.actor)
        co = torch.optim.AdamW(self.critics.parameters(), lr=0.001)
        self.assertFalse(
            {id(p) for p in self.actor.parameters()}
            & {id(p) for p in self.critics.parameters()}
        )
        critic_update(self.critics, co, self.rows, self.cfg)
        self.assertEqual(before, identity_state(self.actor))
        self.assertTrue(all(p.grad is None for p in self.actor.parameters()))

    def test_real_two_head_update_and_optimizer_rollback(self):
        before = {p: identity_state(getattr(self.actor, p)) for p in ("combat", "ooc")}
        legacy = {p: identity_state(getattr(self.actor, p).critic) for p in before}
        update = actor_update(
            self.actor,
            self.ref,
            self.opts,
            self.rows,
            self.rows,
            self.cfg,
            self.rng,
            0.1,
        )
        self.assertTrue(update["accepted"], update)
        self.assertTrue(
            all(before[p] != identity_state(getattr(self.actor, p)) for p in before)
        )
        self.assertTrue(
            all(
                legacy[p] == identity_state(getattr(self.actor, p).critic)
                for p in before
            )
        )
        actor_before = identity_state(self.actor)
        states = copy.deepcopy({p: o.state_dict() for p, o in self.opts.items()})
        rng = self.rng.getstate()
        bad = JointConfig(**{**self.cfg.__dict__, "local_kl_limit": -1})
        rejected = actor_update(
            self.actor, self.ref, self.opts, self.rows, self.rows, bad, self.rng, 0.1
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["retained_steps"], 0)
        self.assertGreater(rejected["attempted_steps"], 0)
        self.assertEqual(actor_before, identity_state(self.actor))
        self.assertEqual(rng, self.rng.getstate())
        for phase in states:
            for key, state in states[phase]["state"].items():
                for name, v in state.items():
                    self.assertTrue(
                        torch.equal(
                            v, self.opts[phase].state_dict()["state"][key][name]
                        )
                    )

    def test_resume_rejects_changed_scientific_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "record.json"
            record = {
                "runtime": {"python": "fixed"},
                "source_archive": {"sha256": "source"},
                "parent_checkpoint": {"sha256": "parent"},
                "native": {"module_sha256": "native"},
            }
            path.write_text(json.dumps(record))
            payload = {
                "metadata": {
                    "config": self.cfg.__dict__,
                    "run_record": {"path": str(path)},
                }
            }
            validate_resume_identity(payload, self.cfg, record)
            for field, key in [
                ("source_archive", "sha256"),
                ("parent_checkpoint", "sha256"),
                ("native", "module_sha256"),
                ("runtime", "python"),
            ]:
                changed = copy.deepcopy(record)
                changed[field][key] = "changed"
                with self.assertRaises(ValueError):
                    validate_resume_identity(payload, self.cfg, changed)

    def test_seed_spacing_guard(self):
        with self.assertRaises(ValueError):
            JointConfig(out="x", checkpoint="x", max_games=10000).validate()


if __name__ == "__main__":
    unittest.main()
