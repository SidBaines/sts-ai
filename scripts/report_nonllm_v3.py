"""Export a read-only, self-contained learning report and interactive dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts_policy.reporting_v3 import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data/nonllm_v3"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compare-data",type=Path,action="append",default=[],help="Preserved study roots to include as separate selectable profiles")
    args = parser.parse_args()
    report = build_report(args.data)
    if args.compare_data:
        report = {"bundle_version":"learning_profiles_v1","studies":[
            {"label":p.name,"report":build_report(p)} for p in [*args.compare_data,args.data]]}
    args.output.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(report,sort_keys=True,allow_nan=False)
    (args.output/"report.json").write_text(payload+"\n")
    template = (Path(__file__).resolve().parents[1]/"web/nonllm_v3_dashboard.html").read_text()
    (args.output/"index.html").write_text(template.replace("__REPORT_PAYLOAD__",payload.replace("<","\\u003c")))
    studies = report.get("studies",[{"report":report}])
    print(json.dumps({"profiles":len(studies),"evaluations":sum(len(s["report"]["evaluations"]) for s in studies),
        "runs":sum(len(s["report"]["runs"]) for s in studies),"errors":[e for s in studies for e in s["report"]["errors"]],
        "dashboard":str((args.output/"index.html").resolve())}))


if __name__ == "__main__":
    main()
