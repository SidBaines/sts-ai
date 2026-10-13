from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts_policy.reporting_v3 import build_report, counterfactual_summary, outcome_summary, paired, read_json, finite_document


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

    def test_counterfactual_reporting_excludes_incomplete_menu_gains_and_checks_hashes(self):
        def document(complete):
            return {"task":{"seed":1},"observation":{"legal_actions":[{},{}]},
                "continuation":{"policy":"privileged native combat"},"usable_complete_menu":complete,
                "panels":[{"reference":{"episodes":[{"task_outcome":"defeat"}]},"branches":[
                    {"valid_completed":True,"paired":{"victory_delta":1,"floor_delta":1}},
                    {"valid_completed":complete,"paired":{"victory_delta":0,"floor_delta":0} if complete else None}]}]}
        summary=counterfactual_summary([document(True),document(False)])
        self.assertEqual(summary["menus_with_win_gain"],1)
        self.assertEqual(summary["source_menus"],2)
        self.assertEqual(summary["source_worlds"],1)
        self.assertEqual(summary["valid_branches"],3)
        with TemporaryDirectory() as folder:
            root=Path(folder); run=root/"counterfactual/probe";run.mkdir(parents=True)
            panel=run/"panel_000.json";panel.write_text(json.dumps(document(True)))
            manifest={"status":"saved","panels":[{"path":panel.name,"sha256":hashlib.sha256(panel.read_bytes()).hexdigest()}]}
            (run/"manifest.json").write_text(json.dumps(manifest))
            report=build_report(root)
            self.assertEqual(report["counterfactuals"][0]["summary"]["usable_source_menus"],1)
            panel.write_text(json.dumps(document(False)))
            report=build_report(root)
            self.assertEqual(report["counterfactuals"][0]["summary"]["source_menus"],0)
            self.assertIn("hash differs",report["errors"][0]["error"])

    def test_report_preserves_actual_act_exposure_and_weighted_loss_mass(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); run=root/"pilots/balanced";run.mkdir(parents=True)
            (run/"status.json").write_text(json.dumps({"status":"saved"}))
            values={"chunk":1,"actual_actor_exposures_by_act":{"1":100,"2":20,"3":5},
                "actual_actor_loss_mass_by_act":{"1":40.,"2":40.,"3":40.}}
            (run/"update_0001.json").write_text(json.dumps(values))
            update=build_report(root)["runs"][0]["updates"][0]
            self.assertEqual(update["actual_actor_exposures_by_act"],values["actual_actor_exposures_by_act"])
            self.assertEqual(update["actual_actor_loss_mass_by_act"],values["actual_actor_loss_mass_by_act"])


if __name__ == "__main__":
    unittest.main()
