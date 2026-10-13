from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sts_ai import fight_lab as lab


def record(index=0, *, combat=True, floor=1, hp=40):
    return {"world_seed": 1004, "rollout_index": 0, "policy_seed": 42,
            "decision_index": index, "phase": "combat" if combat else "out_of_combat",
            "state": {"act": 1, "floor": floor, "cur_hp": hp, "max_hp": 80,
                      "combat": {"player_cur_hp": hp, "player_max_hp": 80, "turn": 1,
                                 "enemies": [{"name": "GREMLIN_NOB", "alive": True}]} if combat else None},
            "state_text": "HP: 40/80", "legal_actions": [{"index": 0, "bits": 1, "description": "end turn"}],
            "selected_action": {"index": 0, "bits": 1, "description": "end turn"},
            "after_state": {"cur_hp": 39}, "agent": {"latency_s": 1, "raw_response": "test"},
            "action_executed": True}


class FightLabTest(unittest.TestCase):
    def test_start_is_first_micro_action_not_every_action_in_turn_one(self):
        records = [record(0, combat=False), record(1), record(2),
                   record(3, combat=False), record(4, floor=2)]
        rows = lab.index_records(records, "sources/seed_1004_r0.jsonl")
        self.assertEqual([r["battle_start"] for r in rows], [False, True, False, False, True])
        self.assertEqual(rows[2]["fight_start_index"], 1)
        self.assertEqual({r["split"] for r in rows}, {"eval"})

    def test_filter_intersection_strict_hp_and_enemy_normalization(self):
        rows = lab.index_records([record(0, hp=40), record(1, hp=39)], "seed_1004_r0.jsonl")
        filters = lab.StateFilter(enemy="gremlin nob", hp_below=0.5, act=1, turn=1, split="eval")
        self.assertFalse(filters.matches(rows[0]))
        self.assertTrue(filters.matches(rows[1]))
        self.assertFalse(lab.StateFilter(enemy="sentry").matches(rows[1]))
        self.assertFalse(lab.StateFilter(floor_min=2).matches(rows[1]))
        with self.assertRaises(ValueError):
            lab.StateFilter(hp_below=float("nan"))
        with self.assertRaises(ValueError):
            lab.StateFilter(floor_min=4, floor_max=2)

    def test_selection_is_deterministic_and_rejects_modified_rows(self):
        collection = {"states": lab.index_records([record()], "seed_1004_r0.jsonl")}
        selected = lab.selection(collection, lab.StateFilter())
        self.assertEqual(selected, lab.selection(collection, lab.StateFilter()))
        lab.validate_selection(collection, selected)
        altered = copy.deepcopy(selected)
        altered["states"][0]["index"] = 99
        with self.assertRaises(ValueError):
            lab.validate_selection(collection, altered)
        altered["selection_sha256"] = lab.digest({k: v for k, v in altered.items() if k != "selection_sha256"})
        with self.assertRaises(ValueError):
            lab.validate_selection(collection, altered)

    def test_latency_excluded_but_model_text_and_state_included(self):
        a, b = record(), record()
        b["agent"]["latency_s"] = 999
        self.assertEqual(lab.trace_digest([a]), lab.trace_digest([b]))
        b["agent"]["raw_response"] = "other"
        self.assertNotEqual(lab.trace_digest([a]), lab.trace_digest([b]))

    def test_explicit_selection_freezes_only_the_clicked_state(self):
        collection = {"states": lab.index_records([record(0), record(1)], "seed_1004_r0.jsonl")}
        selected = lab.selection(collection, lab.StateFilter(), ["seed_1004_r0_d1"])
        self.assertEqual([r["index"] for r in selected["states"]], [1])
        lab.validate_selection(collection, selected)
        altered = copy.deepcopy(selected)
        altered["states"] = collection["states"]
        altered["selection_sha256"] = lab.digest({k: v for k, v in altered.items() if k != "selection_sha256"})
        with self.assertRaises(ValueError):
            lab.validate_selection(collection, altered)
        with self.assertRaises(ValueError):
            lab.selection(collection, lab.StateFilter(), ["missing"])

    def test_source_content_and_directory_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "trace.jsonl"
            path.write_text(__import__("json").dumps(record()) + "\n")
            collection = {"sources": [{"path": path.name, "sha256": lab.file_sha256(path)}]}
            self.assertEqual(lab.source_records(root, collection, path.name), [record()])
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                lab.source_records(root, collection, path.name)
            collection["sources"] = [{"path": "../other.jsonl", "sha256": None}]
            with self.assertRaises(ValueError):
                lab.source_records(root, collection, "../other.jsonl")

    def test_partial_or_invalid_fights_are_not_scored_as_losses(self):
        task = lab.SingleFightTask()
        metrics = task.metrics_from_episode([], {"cur_hp": 20}, "agent_invalid", {"entry_hp": 40})
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["label"], "incomplete")
        self.assertEqual(task.completion_reason({"cur_hp": 0}), "player_loss")
        self.assertEqual(task.completion_reason({"cur_hp": 30}), "task_complete")
        self.assertIsNone(task.completion_reason(record()["state"]))

    def test_replay_rejects_public_or_action_drift_before_step(self):
        from unittest.mock import Mock
        env = Mock()
        source = record()
        view = {**source, "legal_action_dicts": source["legal_actions"]}
        with patch("sts_ai.rollout.prepare_decision", return_value=("ok", view)):
            lab.replay_to(env, [source], 0)
            env.step.assert_not_called()
            bad = copy.deepcopy(source)
            bad["state_text"] += " changed"
            with self.assertRaisesRegex(ValueError, "diverged"):
                lab.replay_to(env, [bad], 0)
            with self.assertRaises(ValueError):
                lab.replay_to(env, [source], -1)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "requires app extra")
class FightLabServerTest(unittest.TestCase):
    def test_filter_export_and_unregistered_paths(self):
        from fastapi.testclient import TestClient
        from sts_ai.fight_lab_server import create_app
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab.write_new(root / "collection.json", {"states": lab.index_records([record()], "seed_1004_r0.jsonl"),
                          "sources": [], "reports": [], "config": {}})
            with TestClient(create_app(root)) as client:
                self.assertEqual(client.get("/").status_code, 200)
                result = client.get("/api/states?enemy=gremlin+nob&battle_start=true").json()
                self.assertEqual(len(result["states"]), 1)
                self.assertEqual(client.get("/api/states?hp_below=0.5").json()["states"], [])
                self.assertEqual(client.get("/api/states?hp_below=2").status_code, 400)
                response = client.get("/api/states?download=true")
                self.assertIn("attachment", response.headers["content-disposition"])
                self.assertEqual(client.get("/api/view?source=../../secret").status_code, 400)
                self.assertEqual(client.get("/api/trajectories").json(), [])
