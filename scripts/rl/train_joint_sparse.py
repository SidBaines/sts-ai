#!/usr/bin/env python3
"""Launch the versioned joint sparse-win PPO condition."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sts_rl.joint_sparse import JointConfig, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    data = json.loads(Path(a.config).read_text())
    data.update(out=a.out, checkpoint=a.checkpoint)
    run(JointConfig(**data), resume=a.resume)


if __name__ == "__main__":
    main()
