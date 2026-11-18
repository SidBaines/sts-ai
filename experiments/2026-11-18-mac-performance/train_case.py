"""One disposable iteration, keeping the promoted training recipe's learning settings."""

import argparse
import json
from pathlib import Path

from sts_rl.ooc_trainer import OOCTrainConfig, train

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--games", type=int, default=96)
    p.add_argument("--diagnostics-on-update-device", action="store_true")
    p.add_argument("--collectors", type=int, default=2)
    p.add_argument("--production-shape", action="store_true")
    args = p.parse_args()
    settings = json.loads(
        (args.run_root / "reward_ablation_20261023/floors/config.json").read_text()
    )
    settings.update(
        out=str(args.out),
        checkpoint=str(args.run_root / "champions/champion_4058cbf0e271.pt"),
        update_device=args.device,
        iterations=1,
        games=args.games,
        workers=16 * args.collectors,
        collectors=args.collectors,
        collect_chunk=48,
        eval_n=24,
        notes="Disposable fixed-work timing on the Mac.",
    )
    if args.production_shape:
        settings.update(
            games=3072, workers=384, collectors=12, collect_chunk=128, eval_n=1024
        )
    if args.diagnostics_on_update_device:
        settings["diagnostics_on_update_device"] = True
    import random

    import numpy as np

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    train(OOCTrainConfig(**settings))
