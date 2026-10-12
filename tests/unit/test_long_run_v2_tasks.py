from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.long_run_tasks import ContinuationPool, INTERFACE_SHA, NATIVE_SHA, task_identity
from sts_policy.long_run_v2_tasks import (sample_v2, mixture_counts, fixed_v2_dev_panels,
    write_task_batch, TRAIN_WORLD_START)


def provenance():
    return {"environment_version": "0.1.1", "observation_version": "sts_env_public_v1",
            "python_interface_sha256": INTERFACE_SHA, "runtime_python_version": "3.12.13",
            "native_provenance": {"module_sha256": NATIVE_SHA, "patch_sha256": "patched"}}


def source_row(world, act, *, learner=False):
    task = TaskSpec(seed=world, kind="run" if learner else "battle", max_act=3,
        prefix=(0, 1) if act == 2 else (0, 1, 2),
        ooc_previews=(True, False) if act == 2 else (True, False, False),
        expected_public_hash="a" * 64, expected_provenance=provenance())
    row = {**task.to_dict(), "curriculum_metadata": {"act": act, "floor": 18 if act == 2 else 35}}
    if learner:
        row["long_metadata"] = {"start_source": "learner", "start_act": act,
            "start_floor": 17 if act == 2 else 34, "task_id": task_identity(task)}
    return row


def pool():
    worlds = [s for s in range(12000, 12500) if world_split(s) == "train"][:30]
    return ContinuationPool({act: tuple(source_row(s, act) for s in worlds) for act in (2, 3)},
                            provenance(), {"manifest": {"path": "/tmp/bank/manifest.json", "sha256": "source"}})


class V2TasksTests(unittest.TestCase):
    def test_exact_integer_quota_and_full_goal_prefix_identity(self):
        self.assertEqual(mixture_counts(128), {"scratch": 90, "act2": 25, "act3": 13})
        self.assertEqual(mixture_counts(10), {"scratch": 7, "act2": 2, "act3": 1})
        self.assertEqual(sum(mixture_counts(1).values()), 1)
        source = pool(); batch = sample_v2(source, round_index=0)
        self.assertEqual(len(batch["tasks"]), 128)
        self.assertEqual(len({task_identity(t) for t in batch["tasks"]}), 128)
        by_id = {task_identity(TaskSpec.from_dict(row)): row for rows in source.by_act.values() for row in rows}
        for task, metadata in zip(batch["tasks"], batch["metadata"]["tasks"], strict=True):
            self.assertEqual((task.kind, task.max_act, task.horizon, task.ascension), ("run", 3, 4096, 0))
            self.assertEqual(metadata["task_id"], task_identity(task))
            self.assertEqual(metadata["goal_act"], 3)
            self.assertEqual(world_split(task.seed), "train")
            if task.prefix:
                original = TaskSpec.from_dict(by_id[metadata["source_frontier_id"]])
                self.assertEqual(task.prefix, original.prefix)
                self.assertEqual(task.ooc_previews, original.ooc_previews)
                self.assertEqual(task.expected_public_hash, original.expected_public_hash)
                self.assertEqual(task.expected_provenance, original.expected_provenance)
                self.assertEqual(len(metadata["source_prefix_sha256"]), 64)
        inventory = batch["metadata"]["world_inventory"]
        self.assertEqual(inventory["fresh_worlds"], 90)
        self.assertEqual(inventory["fresh_saved_overlap"], 0)
        self.assertLess(inventory["distinct_worlds"], 128)  # Native Act2/3 worlds overlap.
        self.assertEqual(inventory["distinct_worlds"], len({t.seed for t in batch["tasks"]}))

    def test_resume_determinism_and_rotating_disjoint_train_blocks(self):
        first = sample_v2(pool(), round_index=0)
        self.assertEqual(first, sample_v2(pool(), round_index=0))
        second = sample_v2(pool(), round_index=1)
        worlds = [{t.seed for t in b["tasks"] if not t.prefix} for b in (first, second)]
        self.assertFalse(worlds[0] & worlds[1])
        self.assertTrue(all(TRAIN_WORLD_START <= s < TRAIN_WORLD_START + 100000 for s in worlds[0]))
        self.assertTrue(all(TRAIN_WORLD_START + 100000 <= s < TRAIN_WORLD_START + 200000 for s in worlds[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full3_continuous_00000000.jsonl"
            original = write_task_batch(first, path)
            self.assertEqual(original, write_task_batch(sample_v2(pool(), round_index=0), path))
            with self.assertRaisesRegex(ValueError, "Immutable"):
                write_task_batch(sample_v2(pool(), round_index=0, seed=22), path)

    def test_learner_source_and_shared_world_union_preserved(self):
        first = sample_v2(pool(), round_index=0, episodes=20)
        fresh = sorted(t.seed for t in first["tasks"] if not t.prefix)
        source = pool()
        rows2 = tuple(source_row(s, 2, learner=True) for s in fresh[:4])
        rows3 = tuple(source_row(s, 3, learner=True) for s in fresh[:2])
        merged = ContinuationPool({2: rows2, 3: rows3}, source.provenance,
                                  {"learner_files": [{"path": "/tmp/learner.jsonl", "sha256": "file"}]})
        batch = sample_v2(merged, round_index=0, episodes=20)
        saved = [m for m in batch["metadata"]["tasks"] if m["start_source"] == "learner"]
        self.assertEqual(len(saved), 6)
        self.assertEqual({m["start_floor"] for m in saved}, {17, 34})
        self.assertEqual(batch["metadata"]["world_inventory"]["distinct_worlds"], 14)
        self.assertEqual(batch["metadata"]["world_inventory"]["fresh_saved_overlap"], 4)
        self.assertEqual(batch["metadata"]["source"], merged.source_identity)

    def test_invalid_namespace_and_saved_split_provenance_rejected(self):
        for kwargs in ({"world_start": 99999999}, {"round_index": -1}, {"round_index": True}, {"round_index": 999000},
                       {"horizon": 0}, {"simulations": 0}, {"episodes": 0}, {"world_start": 2**63 - 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample_v2(pool(), **{"round_index": 0, **kwargs})
        for mutation in ("split", "hash", "provenance", "goal", "previews"):
            bad = copy.deepcopy(pool())
            for row in bad.by_act[2]:
                if mutation == "split": row["split"] = "dev"
                if mutation == "hash": row["expected_public_hash"] = "not-a-hash"
                if mutation == "provenance": row["expected_provenance"]["python_interface_sha256"] = "wrong"
                if mutation == "goal": row["max_act"] = 2
                if mutation == "previews": row["ooc_previews"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): sample_v2(bad, round_index=0)

    def test_dev_screen_confirmation_disjoint_and_retention_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = fixed_v2_dev_panels(directory)
            self.assertEqual(definition, fixed_v2_dev_panels(directory))
            panels = definition["panels"]
            self.assertEqual({k: v["episodes"] for k, v in panels.items()},
                {"screen_full3": 48, "confirm_full3": 192, "retention_act1": 192})
            screen, confirm, retention = (panels[k] for k in ("screen_full3", "confirm_full3", "retention_act1"))
            self.assertFalse(set(screen["worlds"]) & set(confirm["worlds"]))
            self.assertEqual(confirm["worlds"], retention["worlds"])
            for label, panel in panels.items():
                self.assertEqual(len(set(panel["worlds"])), panel["episodes"])
                self.assertTrue(all(400000 <= w < 500000 and world_split(w) == "dev" for w in panel["worlds"]))
                rows = [json.loads(line) for line in Path(panel["path"]).read_text().splitlines()]
                self.assertEqual([task_identity(TaskSpec.from_dict(row)) for row in rows], panel["task_ids"])
                self.assertEqual({row["max_act"] for row in rows}, {1 if label == "retention_act1" else 3})
                self.assertEqual({row["horizon"] for row in rows}, {2048 if label == "retention_act1" else 4096})
            self.assertFalse(any("final" in path.name.lower() for path in Path(directory).iterdir()))
            with self.assertRaisesRegex(ValueError, "Immutable"):
                fixed_v2_dev_panels(directory, seed=778)
            with self.assertRaisesRegex(ValueError, "pair"):
                fixed_v2_dev_panels(directory, retention_episodes=48)


if __name__ == "__main__":
    unittest.main()
