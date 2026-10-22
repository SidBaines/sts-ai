from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import torch

from sts_policy.model_agent import CompositeActorCritic
from sts_policy.training import load_checkpoint
from sts_rl import ooc_trainer, parallel_collect
from sts_rl.ooc_trainer import OOCTrainConfig


def small_model():
    return CompositeActorCritic(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))


def collected_head(model, *args, **kwargs):
    return {"kept": [], "game_episodes": [{"floor": 5, "task_outcome": "defeat"}],
            "faults": 0, "fault_records": [],
            "stats": {"combat_weight": [(model.combat.weight[0, 0].detach().item(), 1)]}}


def fake_collector_main(conn, checkpoint, workers, worker_options, threads):
    """Exercise the real spawned request loop without a simulator or game workers."""
    with patch.object(parallel_collect, "load_policy", return_value=(small_model(), {})), \
            patch.object(parallel_collect, "make_workers", return_value=[]), \
            patch.object(ooc_trainer, "collect_ooc_batch", side_effect=collected_head):
        parallel_collect._collector_main(conn, checkpoint, workers, worker_options, threads)


class OOCCombatFollowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = small_model()
        self.initial = deepcopy(self.model.state_dict())
        self.checkpoint = self.root / "start.pt"
        self.follow = self.root / "combat.pt"
        head_config = {"size": "small", "context": "none"}
        self.template = {"format_version": 1, "model_state": self.initial,
                         "metadata": {"encoder_version": "public_candidate_v2"},
                         "model_config": {"kind": "composite", "combat": head_config, "ooc": head_config}}
        torch.save(self.template, self.checkpoint)
        self.cfg = OOCTrainConfig(out=str(self.root / "run"), checkpoint=str(self.checkpoint),
                                  combat_follow=str(self.follow), games=2, iterations=1,
                                  eval_every=1, health_every=0, exploration_stats=False)

    def write_head(self, value, *, bare=False, path=None):
        state = {k: torch.full_like(v, value) for k, v in self.initial.items()}
        if bare:
            state = {k[len("combat."):]: v for k, v in state.items() if k.startswith("combat.")}
        destination = path or self.follow
        config = self.template["model_config"]["combat"] if bare else self.template["model_config"]
        torch.save({"format_version": 1, "model_state": state, "model_config": config,
                    "metadata": self.template["metadata"]}, destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    def train(self, cfg=None, *, resume=False, collect=None, update=None, evaluate=None):
        def noop_update(*args, **kwargs):
            return {"optimizer": kwargs["optimizer"], "steps": 0, "early_stop": False}

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(ooc_trainer, "load_policy", return_value=(self.model, self.template)), \
                patch.object(ooc_trainer, "begin_run", return_value={}), \
                patch.object(ooc_trainer, "make_workers", return_value=[]), \
                patch.object(ooc_trainer, "evaluate", side_effect=evaluate,
                             return_value={"summary": {}, "records": []}), \
                patch.object(ooc_trainer, "collect_ooc_batch", side_effect=collect or collected_head), \
                patch.object(ooc_trainer, "write_collection"), \
                patch.object(ooc_trainer, "update_ppo", side_effect=update or noop_update):
            ooc_trainer.train(cfg or self.cfg, resume=resume)

    def records(self, event="train"):
        rows = [json.loads(line) for line in (Path(self.cfg.out) / "log.jsonl").read_text().splitlines()]
        return [row for row in rows if row["event"] == event]

    def latest(self):
        return torch.load(Path(self.cfg.out) / "latest.pt", weights_only=False)

    def assert_combat(self, state, expected):
        for key in self.initial:
            if key.startswith("combat."):
                value = expected[key] if isinstance(expected, dict) else torch.full_like(state[key], expected)
                self.assertTrue(torch.equal(state[key], value), key)

    def test_cli_generates_combat_follow_flag(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "rl" / "train_ooc.py"
        with patch.object(sys, "argv", [str(script), "--out", self.cfg.out, "--combat-follow", str(self.follow)]), \
                patch.object(ooc_trainer, "train") as train:
            runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(train.call_args.args[0].combat_follow, str(self.follow))

    def test_collected_head_detaches_parameters_before_reading_them(self):
        self.assertTrue(self.model.combat.weight.requires_grad)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            batch = collected_head(self.model)
        self.assertEqual(batch["stats"]["combat_weight"],
                         [(self.model.combat.weight[0, 0].detach().item(), 1)])

    def test_train_combat_with_follow_is_rejected_before_any_setup(self):
        with patch.object(ooc_trainer, "load_policy") as load:
            with self.assertRaisesRegex(ValueError, "train_combat and combat_follow"):
                ooc_trainer.train(replace(self.cfg, train_combat=True))
        load.assert_not_called()
        self.assertFalse(Path(self.cfg.out).exists())

    def test_unset_does_not_follow_or_add_record_fields(self):
        self.write_head(4.)
        with patch.object(ooc_trainer, "load_checkpoint") as load, \
                patch.object(ooc_trainer, "_combat_weights_sha256") as identity, \
                patch.object(ooc_trainer, "_follow_combat") as follow:
            self.train(replace(self.cfg, combat_follow=None))
        load.assert_not_called()
        identity.assert_not_called()
        follow.assert_not_called()
        self.assert_combat(self.model.state_dict(), self.initial)
        self.assertEqual(self.records("combat_swap"), [])
        for row in self.records() + self.records("eval"):
            self.assertNotIn("combat_sha256", row)
            self.assertNotIn("combat_hash_kind", row)
            self.assertNotIn("combat_swapped", row)
        self.assertEqual(self.latest()["trainer_state"], {})
        for iteration in (0, 1):
            result = json.loads((Path(self.cfg.out) / f"eval_{iteration:04d}.json").read_text())
            self.assertNotIn("combat_sha256", result)
            self.assertNotIn("combat_hash_kind", result)

    def test_missing_file_keeps_starting_weights_and_records_their_hash(self):
        self.train()
        initial_hash = ooc_trainer._combat_weights_sha256(self.initial)
        self.assertEqual(self.records()[0]["combat_sha256"], initial_hash)
        self.assertEqual(self.records()[0]["combat_hash_kind"], "weights")
        self.assertFalse(self.records()[0]["combat_swapped"])
        self.assertEqual(self.records("combat_swap"), [])
        self.assertEqual(self.latest()["trainer_state"],
                         {"combat_sha256": initial_hash, "combat_hash_kind": "weights",
                          "combat_weights_sha256": initial_hash, "combat_path": str(self.checkpoint)})
        for row in self.records("eval"):
            self.assertEqual(row["combat_sha256"], initial_hash)
            self.assertEqual(row["combat_hash_kind"], "weights")

    def test_fresh_start_ignores_inherited_combat_identity_with_no_follow_file(self):
        self.template["trainer_state"] = {
            "combat_sha256": "stale file hash", "combat_hash_kind": "file",
            "combat_weights_sha256": ooc_trainer._combat_weights_sha256(self.initial),
            "combat_path": "inherited.pt"}
        self.train()
        saved = self.latest()["trainer_state"]
        self.assertEqual(saved["combat_sha256"], ooc_trainer._combat_weights_sha256(self.initial))
        self.assertEqual(saved["combat_hash_kind"], "weights")
        self.assertEqual(saved["combat_path"], str(self.checkpoint))

    def test_fresh_spliced_start_adopts_file_even_if_inherited_hash_matches_it(self):
        sha = self.write_head(4.)
        self.template["trainer_state"] = {
            "combat_sha256": sha, "combat_hash_kind": "file",
            "combat_weights_sha256": ooc_trainer._combat_weights_sha256(self.initial),
            "combat_path": str(self.follow)}
        self.train()
        self.assert_combat(self.latest()["model_state"], 4.)
        self.assertTrue(self.records()[0]["combat_swapped"])
        self.assertEqual(self.records("combat_swap")[0]["old_sha256"],
                         ooc_trainer._combat_weights_sha256(self.initial))
        self.assertEqual(self.records()[0]["combat_sha256"], sha)

    def test_changed_file_swaps_before_collection_and_unchanged_hash_is_not_loaded(self):
        first_hash = self.write_head(4.)
        hashes, seen = [first_hash], []

        def collect(model, *args, **kwargs):
            seen.append(model.combat.weight[0, 0].detach().item())
            if len(seen) == 1:
                hashes.append(self.write_head(6.))
            return collected_head(model)

        with patch.object(ooc_trainer, "load_checkpoint", wraps=load_checkpoint) as load:
            self.train(replace(self.cfg, iterations=3), collect=collect)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(seen, [4., 6., 6.])
        rows = self.records()
        self.assertEqual([r["combat_swapped"] for r in rows], [True, True, False])
        self.assertEqual([r["combat_sha256"] for r in rows], [*hashes, hashes[-1]])
        self.assertEqual([r["combat_hash_kind"] for r in rows], ["file"] * 3)
        swaps = self.records("combat_swap")
        self.assertEqual([r["iteration"] for r in swaps], [1, 2])
        self.assertEqual([r["old_sha256"] for r in swaps],
                         [ooc_trainer._combat_weights_sha256(self.initial), hashes[0]])
        self.assertEqual([r["new_sha256"] for r in swaps], hashes)
        self.assertEqual([r["path"] for r in swaps], [str(self.follow)] * 2)
        self.assert_combat(self.latest()["model_state"], 6.)
        for iteration, sha, value in ((1, hashes[0], 4.), (3, hashes[1], 6.)):
            saved = torch.load(Path(self.cfg.out) / f"iter_{iteration:04d}.pt", weights_only=False)
            self.assert_combat(saved["model_state"], value)
            self.assertEqual(saved["metadata"]["combat_sha256"], sha)
            self.assertEqual(saved["metadata"]["combat_hash_kind"], "file")
            self.assertEqual(saved["trainer_state"]["combat_weights_sha256"],
                             ooc_trainer._combat_weights_sha256(saved["model_state"]))
        for key in self.initial:
            if key.startswith("ooc."):
                self.assertTrue(torch.equal(self.model.state_dict()[key], self.initial[key]))

    def test_bare_and_prefixed_heads_are_accepted(self):
        for bare in (False, True):
            with self.subTest(bare=bare):
                sha = self.write_head(3., bare=bare)
                self.model.load_state_dict(self.initial)
                cfg = replace(self.cfg, out=str(self.root / f"run_{bare}"))
                self.train(cfg)
                self.assert_combat(self.model.state_dict(), 3.)
                saved = torch.load(Path(cfg.out) / "latest.pt", weights_only=False)
                self.assertEqual(saved["trainer_state"], {
                    "combat_sha256": sha, "combat_hash_kind": "file", "combat_path": str(self.follow),
                    "combat_weights_sha256": ooc_trainer._combat_weights_sha256(saved["model_state"])})

    def test_missing_misshapen_or_wrong_dtype_key_raises_before_modifying_any_weights(self):
        for bare in (False, True):
            for malformed in ("missing", "shape", "dtype"):
                with self.subTest(bare=bare, malformed=malformed):
                    self.write_head(5., bare=bare)
                    payload = load_checkpoint(self.follow)
                    key = "bias" if bare else "combat.bias"
                    if malformed == "missing":
                        del payload["model_state"][key]
                        message = "missing key combat.bias"
                    elif malformed == "shape":
                        payload["model_state"][key] = torch.zeros(3)
                        message = "shape mismatch for combat.bias"
                    else:
                        payload["model_state"][key] = payload["model_state"][key].double()
                        message = "dtype mismatch for combat.bias"
                    torch.save(payload, self.follow)
                    collect = Mock(side_effect=collected_head)
                    with self.assertRaisesRegex(ValueError, message):
                        self.train(replace(self.cfg, out=str(self.root / f"bad_{bare}_{malformed}")),
                                   collect=collect)
                    collect.assert_not_called()
                    self.assert_combat(self.model.state_dict(), self.initial)

    def test_incompatible_encoder_or_config_is_rejected_before_copying(self):
        for bare in (False, True):
            for malformed in ("encoder", "missing_encoder", "config", "missing_config"):
                if bare and malformed == "missing_config":
                    continue  # Bare legacy heads may omit model_config.
                with self.subTest(bare=bare, malformed=malformed):
                    self.write_head(5., bare=bare)
                    payload = load_checkpoint(self.follow)
                    if malformed == "encoder":
                        payload["metadata"]["encoder_version"] = "incompatible_encoder"
                    elif malformed == "missing_encoder":
                        del payload["metadata"]["encoder_version"]
                    elif malformed == "missing_config":
                        del payload["model_config"]
                    else:
                        config = payload["model_config"] if bare else payload["model_config"]["combat"]
                        config["context"] = "mean"  # Same tensor shapes can have different semantics.
                    torch.save(payload, self.follow)
                    message = "encoder version mismatch" if "encoder" in malformed else "model configuration mismatch"
                    with self.assertRaisesRegex(ValueError, message):
                        ooc_trainer._follow_combat(self.model, str(self.follow), "initial", self.template)
                    self.assert_combat(self.model.state_dict(), self.initial)

    def test_bare_head_without_model_config_still_checks_encoder(self):
        self.write_head(3., bare=True)
        payload = load_checkpoint(self.follow)
        del payload["model_config"]
        torch.save(payload, self.follow)
        self.train()
        self.assert_combat(self.model.state_dict(), 3.)
        payload["metadata"]["encoder_version"] = "incompatible_encoder"
        torch.save(payload, self.follow)
        with self.assertRaisesRegex(ValueError, "encoder version mismatch"):
            ooc_trainer._follow_combat(self.model, str(self.follow), "previous", self.template)
        self.assert_combat(self.model.state_dict(), 3.)

    def test_checkpoint_evaluation_receives_swapped_weights_and_identity(self):
        sha = self.write_head(4.)
        seen = []

        def evaluate(model, *args, **kwargs):
            state = deepcopy(model.state_dict())
            seen.append(state)
            return {"summary": {"combat_weight": state["combat.weight"][0, 0].item()}, "records": []}

        self.train(evaluate=evaluate)
        self.assertEqual(len(seen), 2)
        self.assert_combat(seen[0], self.initial)
        self.assert_combat(seen[1], 4.)
        initial_hash = ooc_trainer._combat_weights_sha256(self.initial)
        for iteration, expected_hash, kind in ((0, initial_hash, "weights"), (1, sha, "file")):
            row = self.records("eval")[iteration]
            self.assertEqual(row["iteration"], iteration)
            self.assertEqual(row["combat_sha256"], expected_hash)
            self.assertEqual(row["combat_hash_kind"], kind)
            self.assertEqual(row["combat_weight"], seen[iteration]["combat.weight"][0, 0].item())
            result = json.loads((Path(self.cfg.out) / f"eval_{iteration:04d}.json").read_text())
            self.assertEqual(result["combat_sha256"], expected_hash)
            self.assertEqual(result["combat_hash_kind"], kind)

    def test_snapshot_bytes_are_used_even_if_file_is_replaced_during_read(self):
        first_hash = self.write_head(4.)
        next_file = self.root / "next.pt"
        second_hash = self.write_head(8., path=next_file)
        read_bytes = Path.read_bytes

        def replace_after_read(path):
            data = read_bytes(path)
            if path == self.follow:
                next_file.replace(self.follow)
            return data

        with patch.object(Path, "read_bytes", replace_after_read):
            sha = ooc_trainer._follow_combat(self.model, str(self.follow), "initial", self.template)
        self.assertEqual(sha, first_hash)
        self.assert_combat(self.model.state_dict(), 4.)
        self.assertEqual(ooc_trainer._follow_combat(self.model, str(self.follow), sha, self.template), second_hash)
        self.assert_combat(self.model.state_dict(), 8.)

    def test_ppo_cannot_change_combat_after_swap_and_optimizer_excludes_it(self):
        self.write_head(4.)

        def corrupt(model, *args, **kwargs):
            optimizer = kwargs["optimizer"]
            optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertTrue(all(not p.requires_grad and id(p) not in optimized for p in model.combat.parameters()))
            with torch.no_grad():
                model.combat.weight.add_(1.)
            return {"optimizer": optimizer, "steps": 1}

        with self.assertRaisesRegex(RuntimeError, "frozen combat head changed"):
            self.train(update=corrupt)

    def test_resume_restores_swapped_head_and_hash_before_following_new_file(self):
        sha = self.write_head(4.)
        self.train()
        self.model.load_state_dict(self.initial)
        with patch.object(ooc_trainer, "load_checkpoint", wraps=load_checkpoint) as load:
            self.train(replace(self.cfg, iterations=2), resume=True)
        load.assert_not_called()
        self.assertEqual([r["combat_swapped"] for r in self.records()], [True, False])
        self.assertEqual(self.records()[-1]["combat_weight"], 4.)
        self.assert_combat(self.model.state_dict(), 4.)
        new_sha = self.write_head(7.)
        self.model.load_state_dict(self.initial)
        self.train(replace(self.cfg, iterations=3), resume=True)
        self.assertEqual(self.records("combat_swap")[-1]["old_sha256"], sha)
        self.assertEqual(self.records()[-1]["combat_sha256"], new_sha)
        self.assert_combat(self.latest()["model_state"], 7.)

    def test_resume_discards_file_identity_without_a_matching_weights_fingerprint(self):
        sha = self.write_head(4.)
        self.train()
        original = self.latest()
        for invalid in ("missing_fingerprint", "changed_weights", "legacy_identity"):
            with self.subTest(invalid=invalid):
                saved = deepcopy(original)
                if invalid == "changed_weights":
                    saved["model_state"]["combat.bias"].fill_(9.)
                else:
                    del saved["trainer_state"]["combat_weights_sha256"]
                    if invalid == "legacy_identity":
                        del saved["trainer_state"]["combat_hash_kind"]
                torch.save(saved, Path(self.cfg.out) / "latest.pt")
                self.model.load_state_dict(self.initial)
                with patch.object(ooc_trainer, "load_checkpoint", wraps=load_checkpoint) as load:
                    self.train(replace(self.cfg, iterations=2), resume=True)
                self.assertEqual(load.call_count, 1)
                self.assertTrue(self.records()[-1]["combat_swapped"])
                self.assertEqual(self.records("combat_swap")[-1]["old_sha256"],
                                 ooc_trainer._combat_weights_sha256(saved["model_state"]))
                self.assertEqual(self.records()[-1]["combat_sha256"], sha)
                self.assertEqual(self.records()[-1]["combat_hash_kind"], "file")
                self.assert_combat(self.latest()["model_state"], 4.)
                self.assertEqual(self.latest()["trainer_state"]["combat_weights_sha256"],
                                 ooc_trainer._combat_weights_sha256(original["model_state"]))

    def test_resume_without_follow_keeps_adopted_combat_and_original_ooc_reference(self):
        self.write_head(4.)
        self.train()
        saved = self.latest()
        saved["model_state"]["ooc.weight"].add_(.5)
        torch.save(saved, Path(self.cfg.out) / "latest.pt")
        self.model.load_state_dict(self.initial)
        self.train(replace(self.cfg, combat_follow=None, iterations=2), resume=True)
        self.assert_combat(self.latest()["model_state"], 4.)
        row = self.records()[-1]
        self.assertNotIn("combat_sha256", row)
        self.assertNotIn("combat_hash_kind", row)
        self.assertNotIn("combat_swapped", row)
        self.assertAlmostEqual(row["disp_ratio_weight"], .5 / self.cfg.lr)
        self.assertEqual(self.latest()["trainer_state"], {})

    def test_old_latest_without_combat_fields_resumes_from_saved_weights(self):
        self.write_head(4.)
        self.train()
        legacy = self.latest()
        legacy.pop("trainer_state")
        legacy["config"].pop("combat_follow")
        torch.save(legacy, Path(self.cfg.out) / "latest.pt")
        self.follow.unlink()
        self.model.load_state_dict(self.initial)
        self.train(replace(self.cfg, iterations=2), resume=True)
        self.assert_combat(self.model.state_dict(), 4.)
        row = self.records()[-1]
        self.assertFalse(row["combat_swapped"])
        self.assertEqual(row["combat_sha256"], ooc_trainer._combat_weights_sha256(legacy["model_state"]))
        self.assertEqual(row["combat_hash_kind"], "weights")
        self.assertEqual(self.latest()["trainer_state"]["combat_path"], str(Path(self.cfg.out) / "latest.pt"))
        self.assertEqual(OOCTrainConfig(**legacy["config"]).combat_follow, None)

    def test_spawned_collectors_receive_full_swapped_state_on_every_request(self):
        self.write_head(4.)
        with patch.object(parallel_collect, "_collector_main", fake_collector_main):
            pool = parallel_collect.CollectorPool(2, workers=2, checkpoint=str(self.checkpoint), in_process=True)
        try:
            ooc_trainer._follow_combat(self.model, str(self.follow), "initial", self.template)
            options = dict(combat="greedy", temperature=1., ppo=ooc_trainer.PPOConfig(),
                           credit="macro", gae_lambda=1., chunk=1, seed=1)
            first = pool.collect_ooc(self.model, [1, 2], **options)
            self.write_head(7.)
            ooc_trainer._follow_combat(self.model, str(self.follow), "previous", self.template)
            second = pool.collect_ooc(self.model, [1, 2], **options)
            self.assertEqual(first["stats"]["combat_weight"], [(4., 1), (4., 1)])
            self.assertEqual(second["stats"]["combat_weight"], [(7., 1), (7., 1)])
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
