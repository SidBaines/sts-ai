#!/usr/bin/env python3
"""Re-evaluate a run's saved iter_*.pt checkpoints with the exact-argmax evaluator.

Writes a sibling run directory <run>_argmax/ whose log.jsonl contains eval events
(paired against the argmax baseline), so the dashboard shows clean curves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from sts_rl.common import JsonlLog, close_workers, load_policy, make_workers  # noqa: E402
from sts_rl.evaluate import evaluate, paired  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--baseline", default="data/rl_runs/baselines/orig_argmax_1024.json")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--n", type=int, default=1024)
    args = p.parse_args()
    run = Path(args.run)
    out = run.parent / (run.name + "_argmax")
    out.mkdir(exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    log.emit({"event": "start", "config": {**config, "notes": f"argmax re-score of {run.name}"}})
    baseline = json.loads(Path(args.baseline).read_text())["records"]
    torch.set_num_threads(1)
    workers = make_workers(args.workers, dead_action_filter=True)
    try:
        for ckpt in sorted(run.glob("iter_*.pt")):
            iteration = int(re.findall(r"\d+", ckpt.stem)[0])
            model, _ = load_policy(ckpt)
            result = evaluate(model, workers, n=args.n)
            (out / f"eval_{iteration:04d}.json").write_text(json.dumps(result))
            log.emit({"event": "eval", "iteration": iteration, **result["summary"],
                      "paired": paired(baseline, result["records"])})
    finally:
        close_workers(workers)
    log.emit({"event": "done"})


if __name__ == "__main__":
    main()
