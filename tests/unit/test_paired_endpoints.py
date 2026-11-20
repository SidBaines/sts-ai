import json
import tempfile
import unittest
from pathlib import Path

from sts_rl.paired_endpoints import compare, compare_boss_replays, digest, paired_binary_outcomes


class PairedEndpointTests(unittest.TestCase):
    def test_frozen_boss_comparisons_separate_salts_and_require_identical_initial_states(self):
        from sts_rl.fresh_panel_grid import canonical_grid_hash
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); gridpath=root/"grid.json"
            rows=[dict(world_seed=w,start_id=4,salt=s,group="BOSS3") for w in (8,9,10) for s in (-1,0)]
            grid=dict(coverage={"BOSS3":3},grid=rows);grid["grid_hash"]=canonical_grid_hash(grid)
            gridpath.write_text(json.dumps(grid))
            records=[dict(**r,policy=arm,win=(arm=="replacement")==(r["salt"]==0),
                          first_combat_hash=str((r["world_seed"],r["salt"]))) for r in rows for arm in ("champion","replacement")]
            summary=dict(grid_hash=grid["grid_hash"],grid_input={"sha256":digest(gridpath)},
                         source={"policy":{"champion":{"sha256":"A"},"combat":{"sha256":"B"}}},
                         confidence_claims_allowed=True,records=records)
            (root/"summary.json").write_text(json.dumps(summary))
            scores,valid=compare_boss_replays(gridpath,root,"A","B",3)
            self.assertTrue(valid)
            self.assertEqual((scores["original"]["difference_pp"],scores["reseeded"]["difference_pp"]),(-100,100))
            records[1]["first_combat_hash"]="other"
            (root/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"identical"):
                compare_boss_replays(gridpath,root,"A","B",3)

    def test_binary_outcomes_check_roster_and_keep_discordant_arms_straight(self):
        a = {"A": True, "B": False, "C": False, "D": False}
        b = {"A": False, "B": True, "C": True, "D": False}
        result = paired_binary_outcomes(a, b)
        self.assertEqual((result["a_only"], result["b_only"], result["difference_pp"]), (1, 2, 25))
        self.assertEqual(result["ci95_pp"], result["worst_case_ci95_pp"])
        with self.assertRaisesRegex(ValueError, "roster"):
            paired_binary_outcomes(a, {"other": None})
        with self.assertRaisesRegex(ValueError, "boolean"):
            paired_binary_outcomes(a, dict(b, D=0))

    def test_missing_outcomes_keep_known_side_and_reject_wrong_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ck = root / "policy.pt"
            ck.write_bytes(b"policy")
            config = dict(world_start=10, n=6, processes=7, games_per_process=8)
            first, second = root / "first", root / "second"
            for path, outcomes in [(first, [True, False, False, None, True, None]),
                                   (second, [False, True, True, False, None, True])]:
                path.mkdir()
                records = [dict(world_seed=10+i, win=x, status="completed" if x is not None else "censored")
                           for i,x in enumerate(outcomes)]
                (path / "run.json").write_text(json.dumps(dict(config=config, native={"module_sha256":"one"}, checkpoints={"checkpoint":{"sha256":digest(ck)}})))
                summary = dict(records=records, completed=6-outcomes.count(None), wins=outcomes.count(True), faults=0,
                               cuts=0, censored=outcomes.count(None), mean_floor=40, requeues=0)
                (path / "summary.json").write_text(json.dumps(summary))
            result = compare(first, second, 10, "a", "b", ck, ck, 6)
            self.assertEqual(result["paired_completed"], 3)
            self.assertAlmostEqual(result["full_cohort_gain_bounds_pp"][0], -100/6)
            self.assertAlmostEqual(result["full_cohort_gain_bounds_pp"][1], 200/6)
            summary["records"][0]["world_seed"] = 99
            (second / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "roster"):
                compare(first, second, 10, "a", "b", ck, ck, 6)
