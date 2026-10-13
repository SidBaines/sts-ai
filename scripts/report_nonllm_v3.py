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
    args = parser.parse_args()
    report = build_report(args.data)
    args.output.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(report,sort_keys=True,allow_nan=False)
    (args.output/"report.json").write_text(payload+"\n")
    template = (Path(__file__).resolve().parents[1]/"web/nonllm_v3_dashboard.html").read_text()
    (args.output/"index.html").write_text(template.replace("__REPORT_PAYLOAD__",payload.replace("<","\\u003c")))
    print(json.dumps({"evaluations":len(report["evaluations"]),"runs":len(report["runs"]),
                      "errors":report["errors"],"dashboard":str((args.output/"index.html").resolve())}))


if __name__ == "__main__":
    main()
