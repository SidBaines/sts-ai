from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts_policy.reporting_v3 import build_report, outcome_summary, paired, read_json, finite_document


def episode(seed,family="fresh",win=False,truncated=False):
    return {"task_id":str(seed),"world_seed":seed,"start_family":family,"terminated":not truncated,
        "truncated":truncated,"task_outcome":"victory" if win else "defeat","floor":51 if win else 30,
        "max_act_reached":3 if win else 2,"actor_choice_rows":5,"fights":[]}


class ReportingV3Tests(unittest.TestCase):
    def test_nonfinite_historical_metrics_are_flagged_unavailable(self):
        unavailable=[]
        cleaned=finite_document({"kl":float("nan"),"mse":[float("inf"),.3]},"",unavailable)
        self.assertEqual(cleaned,{"kl":None,"mse":[None,.3]})
        self.assertEqual(unavailable,[".kl",".mse[0]"])
        json.dumps(cleaned,allow_nan=False)

    def test_fresh_conditional_fault_and_cut_denominators(self):
        rows=[episode(1),episode(2,win=True),episode(3,"continuation",True),episode(4,truncated=True)]
        report=outcome_summary({"episodes":rows,"faults":[{}],"requested_tasks":5,"task_ids":[str(i) for i in range(5)]})
        self.assertEqual(report["fresh_completed"],2)
        self.assertEqual(report["fresh_wins"],1)
        self.assertEqual(report["conditional_wins"],1)
        self.assertEqual(report["survival"][-1]["reached"],1)
        self.assertEqual(report["survival"][-1]["denominator"],2)
        rows.append(episode(2,win=True))
        self.assertIsNone(outcome_summary({"episodes":rows})["fresh_interval_95"])

    def test_paired_excludes_faulted_or_censored_tasks(self):
        a={"episodes":[episode(1),episode(2,win=True)]}
        b={"episodes":[episode(1,win=True),episode(2,truncated=True)]}
        metrics=paired(a,b)
        self.assertEqual(metrics["common_valid"],1)
        self.assertEqual(metrics["wins"],{"gains":1,"losses":0,"net":1})

    def test_report_does_not_read_banks_models_or_final_data(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); (root/"evals").mkdir(); (root/"banks").mkdir()
            (root/"evals/final_unopened.json").write_text("invalid")
            (root/"banks/teacher.json").write_text("invalid")
            doc={"results":[{"episodes":[episode(1)],"task_ids":["1"],"requested_tasks":1}]}
            (root/"evals/common_initial_context_v2_dev192.json").write_text(json.dumps(doc))
            report=build_report(root)
            self.assertEqual(len(report["evaluations"]),1)
            self.assertEqual(report["errors"],[])
            self.assertEqual(len(report["sources"]),1)
            with self.assertRaises(ValueError):
                read_json(root.parent/"escape.json",root)


if __name__ == "__main__":
    unittest.main()
