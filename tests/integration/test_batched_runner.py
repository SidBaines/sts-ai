"""Bounded real-policy trajectory gates; reference artifacts remain read-only."""
import json
from pathlib import Path
import unittest

from sts_rl.batched_runner import compare_records, run_in_process
from sts_rl.controlled_policy import policy_factory
from tests.support import requires_simulator

REFERENCE_ROOT = Path("/Users/sidbaines/.codex/worktrees/representation-controls/SlayTheSpireAI/"
                      "data/rl_runs/representation_controls_20261016/games")
POLICY_ROOT = Path("/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/"
                   "data/rl_runs/fresh_start_20261015")
CHAMPION = POLICY_ROOT / "inputs/champion_3d135d9bd53b.pt"
COMBAT = POLICY_ROOT / "fit_old_combat_lr1e3_v1/best.pt"


@requires_simulator
class ReferenceGamesTests(unittest.TestCase):
    def check_policy(self, condition, combat=None):
        reference = REFERENCE_ROOT / condition / "merged/summary.json"
        paths = [reference, CHAMPION, *([combat] if combat else [])]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            self.skipTest("Reference trajectory gate requires local artifacts: " + ", ".join(missing))
        seeds = list(range(91_001_000_000, 91_001_000_008))
        original = [r for r in json.loads(reference.read_text())["records"] if r["world_seed"] in seeds]
        policy = policy_factory(str(CHAMPION), str(combat) if combat else None)
        single = run_in_process(seeds, policy, 1)
        comparison = compare_records(original, single)
        self.assertEqual(comparison["compared"], 8)
        self.assertEqual(comparison["differences"], [], comparison)
        # Metrics and final observations must also retain the old definitions.
        old = {r["world_seed"]: r for r in original}
        for record in single:
            for key in ("win", "floor", "bosses", "decisions", "behavior", "terminal_hash"):
                self.assertEqual(record[key], old[record["world_seed"]][key], (record["world_seed"], key))
        batched = run_in_process(seeds, policy, 4, reference_records=single)
        comparison = compare_records(single, batched)
        print(f"\n{condition}: K=1 reference {len(single)}/8; K=4 {json.dumps(comparison)}", flush=True)
        for row in comparison["differences"]:
            self.assertIn("decision", row)
            self.assertIsNotNone(row["singleton_logit_gap"])
            self.assertIsNotNone(row["batched_logit_gap"])
            self.assertEqual(row["singleton_action"], row["reference_action"], row)

    def test_champion(self):
        self.check_policy("champion")

    def test_champion_with_replacement_combat(self):
        self.check_policy("old1919_retry1", COMBAT)


if __name__ == "__main__":
    unittest.main()
