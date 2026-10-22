"""Smoke test for scripts/rl/plot_lineage.py on a tiny synthetic run tree."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plot_lineage", ROOT / "scripts/rl/plot_lineage.py")
plot_lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_lineage)


def write_run(run_dir: Path, events: list[dict], iters: dict[int, bytes]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for iteration, payload in iters.items():
        (run_dir / f"iter_{iteration:04d}.pt").write_bytes(payload)


def ppo_events(checkpoint, iterations, games=10, start=1, record=None, resume_at=None, hide=True):
    config = {"credit": "macro", "checkpoint": str(checkpoint), "games": games, "hide_ooc_potions": hide,
              "eval_world_start": 91}
    events = [{"t": 1.0, "event": "resume", "iteration": resume_at, "config": config}] if resume_at is not None else \
        [{"t": 1.0, "event": "start", "config": config, **({"run_record": {"path": str(record)}} if record else {})}]
    for i in range(start, start + iterations):
        events.append({"t": 2.0 + i, "event": "train", "iteration": i, "games": games})
        events.append({"t": 2.5 + i, "event": "eval", "iteration": i, "wins": i, "n": 100, "mean_floor": 30.0,
                       "world_start": 91})
    return events


class PlotLineageTests(unittest.TestCase):
    def test_reconstructs_positions_and_edge_types(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "data" / "rl_runs"
            root_ckpt = Path(directory) / "root.pt"
            root_ckpt.write_bytes(b"root")
            # A: from the external root, parent hash taken from its launch record
            record = runs / "a" / "records" / "launch.json"
            write_run(runs / "a", ppo_events(root_ckpt, 4, record=record), {2: b"a2", 4: b"a4"})
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({"parent_checkpoint": {"sha256": hashlib.sha256(b"root").hexdigest()}}))
            # B: from a renamed copy of A's iteration 2, matched by hash (no launch record)
            (runs / "inputs").mkdir()
            shutil.copy(runs / "a" / "iter_0002.pt", runs / "inputs" / "champion_x.pt")
            write_run(runs / "b", ppo_events(runs / "inputs" / "champion_x.pt", 3, games=20), {3: b"b3"})
            # splice of A's iteration 4 with a combat student, then C trains from the splice
            (runs / "splice.pt").write_bytes(b"spliced")
            write_run(runs / "c", ppo_events(runs / "splice.pt", 2), {})
            # D: branched by copying A's latest at iteration 3 and resuming
            write_run(runs / "d", ppo_events(root_ckpt, 2, start=4, resume_at=3), {})
            # E: an OOC distillation of A's iteration 4
            distill = {"checkpoint": str(runs / "a" / "iter_0004.pt"), "head": "ooc", "beta": 1.0}
            write_run(runs / "e", [{"t": 9.0, "event": "start", "config": distill}], {})
            (runs / "e" / "distilled.pt").write_bytes(b"distilled")
            extras = {
                "splices": [{"id": "s", "output": str(runs / "splice.pt"), "ooc": str(runs / "a" / "iter_0004.pt"),
                             "combat": "student.pt", "combat_label": "student"}],
                "branches": [{"run": "d", "parent_run": "a", "iteration": 3}],
                "champions": [{"label": "champ", "path": str(runs / "inputs" / "champion_x.pt"), "promoted": "1"}],
            }
            lineage = plot_lineage.reconstruct([runs], extras, [])
            nodes = {n["id"]: n for n in lineage["nodes"]}
            edges = {(e["source"], e["target"]): e for e in lineage["edges"]}

            self.assertEqual(lineage["warnings"], [])
            self.assertEqual(nodes["ppo:a"]["x0"], 0)
            self.assertEqual(nodes["ppo:b"]["x0"], 2)
            self.assertEqual(nodes["ppo:b"]["g0"], 20)  # two A iterations of 10 games
            self.assertEqual(nodes["ppo:b"]["x_end"], 5)
            self.assertEqual(nodes["ppo:b"]["g_end"], 80)
            self.assertEqual(nodes["splice:s"]["x0"], 4)
            self.assertEqual(nodes["ppo:c"]["x0"], 4)
            self.assertEqual(nodes["ppo:c"]["combat"], "student")
            self.assertEqual(nodes["ppo:d"]["x0"], 3)
            self.assertEqual([e["x"] for e in nodes["ppo:d"]["evals"]], [4, 5])
            self.assertEqual(nodes["distill:e"]["x0"], 4)
            self.assertTrue(nodes["ppo:a"]["parent"]["node"].startswith("external:"))
            self.assertEqual(edges[("ppo:a", "ppo:b")]["type"], "ppo")
            self.assertEqual(edges[("ppo:a", "ppo:b")]["source_iteration"], 2)
            self.assertEqual(edges[("ppo:a", "splice:s")]["type"], "splice")
            self.assertEqual(edges[("splice:s", "ppo:c")]["type"], "ppo")
            self.assertEqual(edges[("ppo:a", "ppo:d")]["type"], "resume")
            self.assertEqual(edges[("ppo:a", "distill:e")]["type"], "distill")
            self.assertEqual((lineage["champions"][0]["node"], lineage["champions"][0]["iteration"]), ("ppo:a", 2))
            self.assertTrue(nodes["ppo:a"].get("champion_path"))
            self.assertFalse(nodes["ppo:b"].get("champion_path"))

            out = Path(directory) / "lineage.html"
            plot_lineage.write_html(lineage, out)
            self.assertIn(plot_lineage.PLOTLY_URL, out.read_text())


if __name__ == "__main__":
    unittest.main()
