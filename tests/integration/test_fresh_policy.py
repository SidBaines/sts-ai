"""Actual CPU models and generated temporary checkpoints; no native simulator."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from sts_policy.encoding import ENCODER_VERSION
from sts_policy.curriculum import public_hash
from sts_policy.entity_encoding import collate_observations, encode_observation
from sts_policy.entity_model import EntityActorCritic
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, collate_public
from sts_policy.training import save_checkpoint
from sts_rl.fresh_study import (PUBLIC_CONTEXT_ID, StudyPolicy, policy_observation,
                               prepare_run)
from sts_rl.recording import file_identity


def observation(phase):
    labels = (["end turn", "play Strike", "discard potion Fire Potion (slot 1)"]
              if phase == "combat" else ["leave", "buy potion Fire Potion for 50g", "take potion Fire Potion"])
    state = {"act": 2, "cur_hp": 34, "max_hp": 80}
    if phase == "combat":
        # Native combat observations always expose all live potion slots. The
        # discard menu below must agree with that public inventory.
        state["combat"] = {"potions": {"capacity": 3, "occupied": 1, "slots": [
            {"index": 0, "empty": True},
            {"index": 1, "name": "Fire Potion", "empty": False},
            {"index": 2, "empty": True}]}}
    return {"phase": phase, "state": state,
            "text": "Potions: none" if phase != "combat" else "Potions: Fire Potion (slot 1)",
            "legal_actions": [
                {"index": index, "description": label} for index, label in zip((3, 17, 64), labels)]}


class FreshPolicyIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = self.enterContext(tempfile.TemporaryDirectory(
            prefix=".fresh-policy-test-", dir=Path(__file__).resolve().parents[2]))
        self.root = Path(self.directory)
        with torch.random.fork_rng():
            torch.manual_seed(843)
            self.champion_model = CompositeActorCritic(CandidateActorCritic(), CandidateActorCritic())
        self.champion = self.root / "champion.pt"
        self.save_legacy(self.champion, self.champion_model)

    def save_legacy(self, path, model, phase=None):
        metadata = {"encoder_version": ENCODER_VERSION}
        if phase is not None:
            metadata["phase"] = phase
        save_checkpoint(path, model, model_config=model.metadata(), metadata=metadata)

    def replacement(self, kind, phase):
        with torch.random.fork_rng():
            torch.manual_seed(522 if phase == "combat" else 711)
            model = (EntityActorCritic(width=8, layers=1, heads=2, phase=phase) if kind == "entity"
                     else CandidateActorCritic(context="none"))
        path = self.root / f"{kind}-{phase}.pt"
        if kind == "entity":
            torch.save({"model_state": model.state_dict(), "metadata": model.metadata()}, path)
        else:
            self.save_legacy(path, model, phase)
        return path

    def assert_singleton(self, policy, model, raw):
        value = policy_observation(raw)
        original = deepcopy(raw)
        entity = isinstance(model, EntityActorCritic)
        batch = (collate_observations([encode_observation(value)], device="cpu") if entity
                 else collate_public([value], "cpu"))
        with torch.inference_mode():
            expected = (model.actor_logits(batch) if entity else model.actor_logits(*batch))[0].clone()
        actor = model.actor_logits

        def checked(*args, **kwargs):
            self.assertFalse(kwargs)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            if entity:
                self.assertEqual(len(args), 1)
                self.assertIsInstance(args[0], dict)
                tensors = args[0].values()
                self.assertEqual(tuple(args[0]["action_mask"].shape), (1, len(value["legal_actions"])))
                self.assertNotIn("action_indices", args[0])
                self.assertNotIn("coverage", args[0])
            else:
                self.assertEqual(len(args), 3)
                tensors = args
                self.assertEqual(tuple(args[2].shape), (1, len(value["legal_actions"])))
            self.assertTrue(all(t.device.type == "cpu" and t.shape[0] == 1 for t in tensors))
            return actor(*args)

        with ExitStack() as stack:
            stack.enter_context(patch.object(model, "actor_logits", side_effect=checked))
            stack.enter_context(patch.object(model, "forward", side_effect=AssertionError("full forward called")))
            critics = ([model.combat.critic, model.ooc.critic] if isinstance(model, CompositeActorCritic)
                       else [model.critic])
            for critic in critics:
                stack.enter_context(patch.object(critic, "forward", side_effect=AssertionError("critic called")))
            actual = policy._logits(value)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(policy.choose_action(raw), value["legal_actions"][int(expected.argmax())]["index"])
            indices, probabilities = policy.distribution(raw)
            self.assertEqual(indices, [a["index"] for a in value["legal_actions"]])
            self.assertEqual(probabilities, expected.softmax(-1).tolist())
        self.assertEqual(raw, original)
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))

    def test_actual_legacy_composite_champion_both_phases_actor_only(self):
        policy = StudyPolicy(str(self.champion))
        self.assertIsInstance(policy.champion, CompositeActorCritic)
        for name, tensor in self.champion_model.state_dict().items():
            self.assertTrue(torch.equal(tensor, policy.champion.state_dict()[name]))
        for phase in ("combat", "out_of_combat"):
            self.assert_singleton(policy, policy.champion, observation(phase))

    def test_actual_entity_and_legacy_replacements_roundtrip_exact_singleton_protocol(self):
        for kind, cls in (("entity", EntityActorCritic), ("legacy", CandidateActorCritic)):
            with self.subTest(kind=kind):
                paths = {phase: self.replacement(kind, phase) for phase in ("combat", "ooc")}
                policy = StudyPolicy(str(self.champion), **{k: str(v) for k, v in paths.items()})
                with patch.object(policy.champion, "actor_logits", side_effect=AssertionError("fallback")):
                    for phase, model in policy.replacements.items():
                        self.assertIsInstance(model, cls)
                        raw = observation("combat" if phase == "combat" else "out_of_combat")
                        self.assert_singleton(policy, model, raw)
                        # Serialize the actually loaded weights and load them through StudyPolicy again.
                        reserialized = self.root / f"{kind}-{phase}-roundtrip.pt"
                        payload = torch.load(paths[phase], map_location="cpu", weights_only=False)
                        for name, tensor in payload["model_state"].items():
                            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))
                        payload["model_state"] = model.state_dict()
                        torch.save(payload, reserialized)
                        loaded = StudyPolicy(str(self.champion), **{phase: str(reserialized)})
                        self.assertTrue(torch.equal(policy._logits(policy_observation(raw)),
                                                    loaded._logits(policy_observation(raw))))
                        for name, tensor in model.state_dict().items():
                            self.assertTrue(torch.equal(tensor, loaded.replacements[phase].state_dict()[name]))

    def test_actual_entity_encoder_receives_inventory_after_choices_are_hidden(self):
        policy = StudyPolicy(str(self.champion), ooc=str(self.replacement("entity", "ooc")))
        raw = observation("out_of_combat")
        raw["text"] = "Potions: EMPTY_POTION_SLOT"
        raw["legal_actions"] += [
            {"index": 81, "description": "discard potion Fire Potion (slot 1)", "bits": 998},
            {"index": 99, "description": "drink potion Fire Potion (slot 1): deal 20 damage"}]
        original, digest = deepcopy(raw), public_hash(raw)
        with patch("sts_policy.entity_encoding.encode_observation", wraps=encode_observation) as encode:
            self.assertIn(policy.choose_action(raw), [3, 17, 64])
        model_input = encode.call_args.args[0]
        self.assertEqual(model_input["adapter_public_ooc_potions"], [{"slot": 1, "name": "Fire Potion"}])
        self.assertEqual([a["index"] for a in model_input["legal_actions"]], [3, 17, 64])
        self.assertEqual(raw, original)
        self.assertEqual(public_hash(raw), digest)
        self.assertEqual(encode_observation(model_input).coverage["potion_count"], 1)

    def test_combat_structured_inventory_ignores_injected_adapter_context(self):
        raw = observation("combat")
        original = collate_observations([encode_observation(raw)], device="cpu")
        raw["adapter_public_ooc_potions"] = [{"slot": 99, "name": "injected private potion"}]
        self.assertIs(policy_observation(raw), raw)
        injected = collate_observations([encode_observation(raw)], device="cpu")
        self.assertEqual(set(original), set(injected))
        self.assertTrue(all(torch.equal(original[key], injected[key]) for key in original))

    def test_actual_models_nonfinite_runtime_logits_reject_without_champion_fallback(self):
        for kind in ("entity", "legacy"):
            with self.subTest(kind=kind):
                policy = StudyPolicy(str(self.champion), ooc=str(self.replacement(kind, "ooc")))
                with torch.no_grad():
                    next(policy.replacements["ooc"].parameters()).fill_(float("nan"))
                with patch.object(policy.champion, "actor_logits", side_effect=AssertionError("fallback")):
                    with self.assertRaisesRegex(ValueError, "Nonfinite policy logits"):
                        policy.choose_action(observation("out_of_combat"))

    def test_replacement_metadata_weights_and_phase_rejected_without_fallback(self):
        for kind in ("entity", "legacy"):
            path = self.replacement(kind, "ooc")
            base = torch.load(path, map_location="cpu", weights_only=False)
            changes = {
                "unknown kind": lambda p: p["metadata"].update(kind="unknown"),
                "encoder": lambda p: p["metadata"].update(encoder_version="changed"),
                "missing metadata": lambda p: p.pop("metadata"),
                "metadata not mapping": lambda p: p.update(metadata=[]),
                "missing weight": lambda p: p["model_state"].pop(next(iter(p["model_state"]))),
            }
            if kind == "entity":
                changes.update({
                    "vocab hash": lambda p: p["metadata"].update(vocabulary_sha256="changed"),
                    "vocab size": lambda p: p["metadata"].update(vocabulary_size=-1),
                    "misphased": lambda p: p["metadata"]["model_config"].update(phase="combat"),
                    "unknown phase": lambda p: p["metadata"]["model_config"].update(phase="unknown"),
                    "missing phase": lambda p: p["metadata"]["model_config"].pop("phase"),
                    "contradictory phase": lambda p: p["metadata"].update(phase="combat"),
                })
            else:
                changes.update({
                    "misphased": lambda p: p["metadata"].update(phase="combat"),
                    "unknown phase": lambda p: p["metadata"].update(phase="unknown"),
                    "missing phase": lambda p: p["metadata"].pop("phase"),
                    "model kind": lambda p: p["metadata"].update(model_kind="unknown"),
                    "config kind": lambda p: p["model_config"].update(kind="unknown"),
                    "version": lambda p: p.update(format_version=-1),
                })
            for label, change in changes.items():
                with self.subTest(kind=kind, error=label):
                    payload = deepcopy(base)
                    change(payload)
                    torch.save(payload, path)
                    with self.assertRaises((ValueError, RuntimeError, KeyError)):
                        StudyPolicy(str(self.champion), ooc=str(path))
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(kind=kind, nonfinite=nonfinite):
                    payload = deepcopy(base)
                    next(iter(payload["model_state"].values())).reshape(-1)[0] = nonfinite
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "Nonfinite"):
                        StudyPolicy(str(self.champion), ooc=str(path))

    def test_composite_cannot_be_a_phase_replacement_and_corrupt_load_never_falls_back(self):
        payload = torch.load(self.champion, map_location="cpu", weights_only=False)
        payload["metadata"]["phase"] = "ooc"
        replacement = self.root / "bad.pt"
        torch.save(payload, replacement)
        with self.assertRaisesRegex(ValueError, "single candidate"):
            StudyPolicy(str(self.champion), ooc=str(replacement))
        replacement.write_bytes(b"not a checkpoint")
        with self.assertRaises(Exception):
            StudyPolicy(str(self.champion), ooc=str(replacement))

    def test_changed_champion_and_replacement_inputs_fail_closed(self):
        replacement = self.replacement("entity", "ooc")
        original_load = torch.load
        for changing in (self.champion, replacement):
            with self.subTest(changing=changing.name):
                def changed_load(path, *args, **kwargs):
                    payload = original_load(path, *args, **kwargs)
                    if Path(path) == changing:
                        with changing.open("ab") as output:
                            output.write(b"changed")
                    return payload
                with patch("torch.load", side_effect=changed_load):
                    with self.assertRaisesRegex(RuntimeError, "changed during load"):
                        StudyPolicy(str(self.champion), ooc=str(replacement))

    def test_nonfinite_champion_rejected_at_load(self):
        payload = torch.load(self.champion, map_location="cpu", weights_only=False)
        payload["model_state"]["ooc.critic.0.weight"][0, 0] = float("nan")
        torch.save(payload, self.champion)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            StudyPolicy(str(self.champion))

    def test_receipt_records_context_identity_and_detects_changed_checkpoint(self):
        replacement = self.replacement("entity", "ooc")
        policy = StudyPolicy(str(self.champion), ooc=str(replacement))
        with patch("sts_rl.fresh_study.begin_run", return_value={"sha256": "fixture"}):
            receipt = prepare_run(self.root / "run", {"fixture": True}, policy)
            stored = json.loads((self.root / "run" / "receipt.json").read_text())
        self.assertEqual(receipt, stored)
        self.assertEqual(stored["policy"]["public_context_id"], PUBLIC_CONTEXT_ID)
        self.assertEqual(stored["policy"]["champion"], file_identity(self.champion))
        self.assertEqual(stored["policy"]["ooc"], file_identity(replacement))
        with replacement.open("ab") as output:
            output.write(b"changed")
        with patch("sts_rl.fresh_study.begin_run", side_effect=AssertionError("launch after changed input")):
            with self.assertRaisesRegex(RuntimeError, "ooc changed"):
                prepare_run(self.root / "changed-run", {}, policy)


if __name__ == "__main__":
    unittest.main()
