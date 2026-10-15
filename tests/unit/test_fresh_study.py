"""Pure public-policy and operational-control contracts; no native simulator."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sts_rl.fresh_study import (DEAD_FILTER_ID, FILTER_ID, INFERENCE, Interrupted,
    MIN_DISK, PUBLIC_CONTEXT_ID, RunControl, StudyPolicy, atomic_json, filter_observation,
    policy_observation, prepare_run, validate_world_range)


def obs(phase="out_of_combat"):
    descriptions = ["drink potion Blood Potion (slot 0): heal 16 HP",
                    "buy potion Blood Potion", "discard potion Weak Potion (slot 1)",
                    "take potion Fire Potion", "drink potions", "Drink potion X", "recall", "take key"]
    return {"phase": phase, "state": {"act": 2}, "text": "Full glossary and visible route text",
            "legal_actions": [{"index": i*3+1, "description": d} for i, d in enumerate(descriptions)]}


class Model:
    def __init__(self, logits):
        self.logits, self.calls = logits, 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def actor_logits(self, *batch, **kwargs):
        self.calls += 1
        return torch.tensor([self.logits], dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        raise AssertionError("Critic/full forward must not run")


class StudyTests(unittest.TestCase):
    def policy(self, model):
        with patch("sts_rl.common.load_policy", return_value=(model, {})), \
             patch("sts_rl.fresh_study.file_identity", return_value={"path": "fake", "sha256": "abc"}):
            return StudyPolicy("fake")

    def test_precise_filter_indices_entries_order_immutable(self):
        value = obs()
        original = deepcopy(value)
        filtered = filter_observation(value)
        self.assertEqual([a["index"] for a in filtered["legal_actions"]], [4, 10, 13, 16, 19, 22])
        self.assertTrue(all(a is value["legal_actions"][i] for a, i in zip(filtered["legal_actions"], [1, 3, 4, 5, 6, 7])))
        self.assertEqual(value, original)
        self.assertIs(filtered["state"], value["state"])
        self.assertEqual(filtered["text"], value["text"])
        self.assertEqual([a["index"] for a in policy_observation(value)["legal_actions"]], [4, 10, 13, 16])

    def test_champion_only_inference_does_not_import_entity_dependencies(self):
        import builtins
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(("sts_policy.entity_model", "sts_policy.entity_encoding")):
                raise AssertionError("Champion-only path imported entity dependencies")
            return original_import(name, *args, **kwargs)
        policy = self.policy(Model([0.] * 4))
        with patch("builtins.__import__", side_effect=guarded_import):
            self.assertEqual(policy.choose_action(obs()), 4)

    def test_combat_unchanged_and_no_empty_filter_fallback(self):
        value = obs("combat")
        self.assertIs(filter_observation(value), value)
        self.assertIs(policy_observation(value), value)
        with self.assertRaisesRegex(ValueError, "emptied"):
            filter_observation({**value, "phase": "out_of_combat", "legal_actions": value["legal_actions"][:1]})

    def test_exact_greedy_tie_original_indices_and_full_public_input(self):
        model = Model([1., 2., 2., 0.])
        policy = self.policy(model)
        captured = []
        def collate(rows, device):
            captured.extend(rows)
            return ()
        with patch("sts_policy.model_agent.collate_public", side_effect=collate):
            self.assertEqual(policy.choose_action(obs()), 10)
            indices, probabilities = policy.distribution(obs())
        self.assertEqual(indices, [4, 10, 13, 16])
        self.assertAlmostEqual(sum(probabilities), 1., places=6)
        self.assertEqual(model.calls, 2)
        self.assertEqual(captured[0]["text"], obs()["text"])
        self.assertEqual(set(captured[0]), {"state", "phase", "text", "legal_actions",
                                         "adapter_public_ooc_potions"})
        self.assertEqual(policy.identity["filter_ids"], [FILTER_ID, DEAD_FILTER_ID])
        self.assertEqual(policy.identity["inference"], INFERENCE)
        self.assertEqual(policy.identity["public_context_id"], PUBLIC_CONTEXT_ID)

    def test_argmax_does_not_use_rounded_softmax(self):
        # Softmax underflow of losers does not change actual logit argmax.
        model = Model([-10000., -10001., 10000., -10002.])
        policy = self.policy(model)
        with patch("sts_policy.model_agent.collate_public", return_value=()):
            self.assertEqual(policy.choose_action(obs()), 13)

    def test_single_choice_filters_first_and_skips_model(self):
        policy = self.policy(Model([float("nan")]))
        value = obs()
        value["legal_actions"] = value["legal_actions"][:2]
        self.assertEqual(policy.choose_action(value), 4)
        self.assertEqual(policy.distribution(value), ([4], [1.]))
        self.assertEqual(policy.champion.calls, 0)

    def test_nonfinite_and_shape_fail(self):
        for logits in ([float("nan"), 1., 2., 3.], [float("inf"), 1., 2., 3.], [1.]):
            policy = self.policy(Model(logits))
            with patch("sts_policy.model_agent.collate_public", return_value=()):
                with self.assertRaises(ValueError):
                    policy.choose_action(obs())

    def test_entity_head_dispatch_lazy_encoder_and_no_diagnostic_features(self):
        from sts_policy.entity_model import EntityActorCritic
        champion = Model([0.]*8)
        policy = self.policy(champion)
        entity = EntityActorCritic(width=8, layers=1, heads=2, phase="ooc")
        policy.replacements["ooc"] = entity
        seen = []
        def encode(value):
            seen.append(value)
            return value
        with patch("sts_policy.entity_encoding.encode_observation", side_effect=encode), \
             patch("sts_policy.entity_encoding.collate_observations",
                   side_effect=lambda values, device: {"public": values}), \
             patch.object(entity, "actor_logits", return_value=torch.tensor([[0., 1., 0., 0.]])) as actor, \
             patch("sts_policy.model_agent.collate_public", return_value=()):
            self.assertEqual(policy.choose_action(obs()), 10)
            self.assertEqual(policy.choose_action(obs("combat")), 1)
        actor.assert_called_once_with({"public": seen})
        self.assertEqual(champion.calls, 1)
        self.assertEqual(seen[0]["text"], obs()["text"])
        self.assertNotIn("world_seed", seen[0]["state"])

    def test_public_inventory_survives_hidden_choices_and_preserves_raw_objects_hash(self):
        from sts_policy.curriculum import public_hash
        value = obs()
        value["legal_actions"][0]["diagnostics"] = {"bits": 991}
        original, digest = deepcopy(value), public_hash(value)
        legal, entries, state = value["legal_actions"], tuple(value["legal_actions"]), value["state"]
        filtered = policy_observation(value)
        self.assertEqual(filtered["adapter_public_ooc_potions"], [
            {"slot": 0, "name": "Blood Potion"}, {"slot": 1, "name": "Weak Potion"}])
        self.assertFalse(any(a["description"].startswith(("drink potion ", "discard potion "))
                             for a in filtered["legal_actions"]))
        self.assertEqual(value, original)
        self.assertEqual(public_hash(value), digest)
        self.assertIs(value["legal_actions"], legal)
        self.assertIs(filtered["state"], state)
        self.assertTrue(all(a is b for a, b in zip(value["legal_actions"], entries)))
        self.assertTrue(all(any(a is b for b in entries) for a in filtered["legal_actions"]))

    def test_native_gap_empty_sentinel_and_higher_public_menu_slot(self):
        value = {"phase": "out_of_combat", "state": {}, "text": "Potions: EMPTY_POTION_SLOT",
                 "legal_actions": [{"index": 4, "description": "discard potion Fire Potion (slot 1)"},
                                   {"index": 9, "description": "leave"}]}
        filtered = policy_observation(value)
        self.assertEqual(filtered["adapter_public_ooc_potions"], [{"slot": 1, "name": "Fire Potion"}])
        self.assertEqual(filtered["legal_actions"], [value["legal_actions"][1]])
        self.assertEqual(value["text"], "Potions: EMPTY_POTION_SLOT")

    def test_duplicate_names_deduplicate_by_slot_sort_and_reject_conflicts(self):
        value = obs()
        labels = ["discard potion Fire Potion (slot 2)", "drink potion Fire Potion (slot 0): deal 20 damage",
                  "discard potion Fire Potion (slot 0)", "leave"]
        value["legal_actions"] = [{"index": i*7, "description": d} for i, d in enumerate(labels)]
        self.assertEqual(policy_observation(value)["adapter_public_ooc_potions"], [
            {"slot": 0, "name": "Fire Potion"}, {"slot": 2, "name": "Fire Potion"}])
        value["legal_actions"][2] = {"index": 14, "description": "discard potion Weak Potion (slot 0)"}
        original = deepcopy(value)
        with self.assertRaisesRegex(ValueError, "Conflicting.*slot 0"):
            policy_observation(value)
        self.assertEqual(value, original)

    def test_injected_private_inventory_overwritten_even_without_visible_slots(self):
        value = obs()
        injected = [{"slot": 99, "name": "private", "bits": 123, "index": 9}]
        value["adapter_public_ooc_potions"] = injected
        self.assertEqual(policy_observation(value)["adapter_public_ooc_potions"], [
            {"slot": 0, "name": "Blood Potion"}, {"slot": 1, "name": "Weak Potion"}])
        value["legal_actions"] = [{"index": 8, "description": "leave"}]
        self.assertEqual(policy_observation(value)["adapter_public_ooc_potions"], [])
        self.assertIs(value["adapter_public_ooc_potions"], injected)
        value["phase"] = "combat"
        self.assertIs(policy_observation(value), value)

    def test_we_meet_again_does_not_invent_unavailable_slots_from_text_or_event(self):
        value = {"phase": "out_of_combat", "text": "Potions: EMPTY_POTION_SLOT",
                 "state": {"event": {"event": "WE_MEET_AGAIN", "potion": "Fire Potion"}},
                 "legal_actions": [{"index": 4, "description": "give potion Fire Potion"},
                                   {"index": 19, "description": "leave"}],
                 "adapter_public_ooc_potions": [{"slot": 1, "name": "Fire Potion"}]}
        filtered = policy_observation(value)
        self.assertEqual(filtered["adapter_public_ooc_potions"], [])
        self.assertEqual([a["index"] for a in filtered["legal_actions"]], [4, 19])
        self.assertIs(filtered["state"], value["state"])

    def test_real_legacy_actor_avoids_critic(self):
        from sts_policy.model import CandidateActorCritic
        model = CandidateActorCritic()
        policy = self.policy(model)
        value = {"phase": "combat", "state": {}, "text": "",
                 "legal_actions": [{"index": 3, "description": "end turn"},
                                   {"index": 9, "description": "play Strike"}]}
        with patch.object(model.critic, "forward", side_effect=AssertionError("critic called")):
            self.assertIn(policy.choose_action(value), [3, 9])


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.popen = self.enterContext(patch("sts_rl.fresh_study.subprocess.Popen"))
        self.popen.return_value.poll.return_value = None

    def test_stop_disk_and_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = {"ac": lambda: True, "disk": lambda _: SimpleNamespace(free=MIN_DISK)}
            control = RunControl(path, "2026-10-16T08:00:00Z", now=lambda: 0, **args)
            self.addCleanup(control.close)
            control.check()
            (path/"STOP").touch()
            with self.assertRaisesRegex(Interrupted, "STOP"):
                control.check()
            (path/"STOP").unlink()
            control.deadline = 0
            with self.assertRaisesRegex(Interrupted, "deadline"):
                control.check()
            control.deadline = None
            control.disk = lambda _: SimpleNamespace(free=MIN_DISK-1)
            with self.assertRaisesRegex(Interrupted, "disk"):
                control.check()
            with self.assertRaisesRegex(ValueError, "UTC"):
                RunControl(path, "2026-10-16T08:00:00", **args)

    def test_battery_pause_and_active_ac_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {"time": 0., "ac": True, "sleeps": 0}
            def sleep(seconds):
                state["sleeps"] += 1
                state["time"] += seconds
                state["ac"] = True
            control = RunControl(directory, ac=lambda: state["ac"], monotonic=lambda: state["time"],
                sleep=sleep, disk=lambda _: SimpleNamespace(free=MIN_DISK))
            self.addCleanup(control.close)
            state["time"] = 2.
            self.assertEqual(control.stats()["active_ac_seconds"], 2.)
            state["ac"] = False
            state["time"] = 4.
            control.check()
            self.assertEqual(state["sleeps"], 2)
            self.assertEqual(control.stats()["active_ac_seconds"], 2.)
            state["time"] += 3.
            self.assertEqual(control.stats()["active_ac_seconds"], 5.)

    def test_cached_power_polls_bounded_under_concurrent_stats_and_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {"time": 0.}
            ac = Mock(return_value=True)
            disk = Mock(return_value=SimpleNamespace(free=MIN_DISK))
            control = RunControl(directory, ac=ac, monotonic=lambda: state["time"], disk=disk)
            self.addCleanup(control.close)
            def burst(_):
                for _ in range(100):
                    control.stats()
                    control.check()
            state["time"] = 1.999
            burst(0)
            self.assertEqual(ac.call_count, 1)
            state["time"] = 2.
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(burst, range(4)))
            self.assertEqual(ac.call_count, 2)
            self.assertEqual(control.stats()["active_ac_seconds"], 2.)
            self.assertGreaterEqual(disk.call_count, 1000)
            disk.return_value = SimpleNamespace(free=MIN_DISK-1)
            with self.assertRaisesRegex(Interrupted, "disk"):
                control.check()
            (Path(directory)/"STOP").touch()
            with self.assertRaisesRegex(Interrupted, "STOP"):
                control.check()
            (Path(directory)/"STOP").unlink()
            control.deadline, control.now = 0., lambda: 0.
            with self.assertRaisesRegex(Interrupted, "deadline"):
                control.check()
            self.assertEqual(ac.call_count, 2)

    def test_stop_and_deadline_remain_prompt_during_cached_battery_pause(self):
        for reason in ("STOP", "deadline"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                state = {"time": 0., "sleeps": 0}
                def sleep(seconds):
                    state["time"] += seconds
                    state["sleeps"] += 1
                    if reason == "STOP":
                        (Path(directory)/"STOP").touch()
                ac = Mock(return_value=False)
                control = RunControl(directory, ac=ac, monotonic=lambda: state["time"],
                    now=lambda: state["time"], sleep=sleep, disk=lambda _: SimpleNamespace(free=MIN_DISK))
                if reason == "deadline":
                    control.deadline = 1.
                with self.assertRaisesRegex(Interrupted, reason):
                    control.check()
                self.assertEqual(state["sleeps"], 1)
                self.assertEqual(ac.call_count, 1)
                self.assertEqual(control.stats()["active_ac_seconds"], 0.)
                control.close()

    def test_owned_caffeinate_ac_battery_resume_and_idempotent_close(self):
        with tempfile.TemporaryDirectory() as directory, patch("sts_rl.fresh_study.sys.platform", "darwin"):
            first, second = Mock(), Mock()
            first.poll.return_value = second.poll.return_value = None
            self.popen.side_effect = [first, second]
            state = {"time": 0., "ac": True}
            ac = Mock(side_effect=lambda: state["ac"])
            control = RunControl(directory, ac=ac, monotonic=lambda: state["time"],
                                 disk=lambda _: SimpleNamespace(free=MIN_DISK))
            self.assertEqual(self.popen.call_args.args[0], ["caffeinate", "-s", "-w", str(os.getpid())])
            state.update(time=2., ac=False)
            self.assertFalse(control.stats()["on_ac"])
            first.terminate.assert_called_once_with()
            first.wait.assert_called_once_with(timeout=1)
            first.kill.assert_not_called()
            state.update(time=4., ac=True)
            control.check()
            self.assertEqual(self.popen.call_count, 2)
            state["time"] = 5.
            control.close()
            second.terminate.assert_called_once_with()
            second.kill.assert_not_called()
            calls = ac.call_count
            state["time"] = 20.
            control.close()
            self.assertEqual(control.stats()["active_ac_seconds"], 1.)
            self.assertEqual(ac.call_count, calls)

    def test_caffeinate_not_started_on_battery_or_other_platforms(self):
        for platform, powered in (("darwin", False), ("linux", True)):
            with self.subTest(platform=platform, powered=powered), tempfile.TemporaryDirectory() as directory, \
                    patch("sts_rl.fresh_study.sys.platform", platform):
                control = RunControl(directory, ac=lambda: powered, disk=lambda _: SimpleNamespace(free=MIN_DISK))
                control.stats()
                control.close()
        self.popen.assert_not_called()

    def test_close_releases_owned_assertion_even_if_final_power_poll_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch("sts_rl.fresh_study.sys.platform", "darwin"):
            control = RunControl(directory, ac=Mock(side_effect=[True, RuntimeError("power unavailable")]))
            with self.assertRaisesRegex(RuntimeError, "power unavailable"):
                control.close()
            self.popen.return_value.terminate.assert_called_once_with()
            control.close()

    def test_fresh_only_atomic_raw_json_and_seed_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"out"
            policy = SimpleNamespace(identity={"champion": {"path": "fake"}})
            with patch("sts_rl.fresh_study.begin_run", return_value={"sha256": "launch"}), \
                 patch("sts_rl.fresh_study.file_identity", return_value={"path": "fake"}):
                prepare_run(path, {"seeds": [17, 18]}, policy)
                with self.assertRaises(FileExistsError):
                    prepare_run(path, {}, policy)
            atomic_json(path/"data.json", obs())
            self.assertEqual(json.loads((path/"data.json").read_text()), obs())
            self.assertEqual(list(path.glob("*.tmp-*")), [])
        validate_world_range(90_500_000_000, 5)
        # Validate the embargo boundary arithmetically; never run/access a world.
        with self.assertRaisesRegex(ValueError, "embargoed"):
            validate_world_range(91_199_999_999, 2)


if __name__ == "__main__":
    unittest.main()
