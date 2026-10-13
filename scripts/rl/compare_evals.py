#!/usr/bin/env python3
"""Paired comparison of eval JSONs on common worlds: each B minus A.

Usage: compare_evals.py A.json B.json [B2.json ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.evaluate import paired  # noqa: E402


def main() -> None:
    a = json.loads(Path(sys.argv[1]).read_text())
    for path in sys.argv[2:]:
        b = json.loads(Path(path).read_text())
        p = paired(a["records"], b["records"])
        print(f"{path}: mean {b['summary']['mean_floor']:.2f} vs {a['summary']['mean_floor']:.2f}  "
              f"paired {p['diff']:+.2f} ± {p['se']:.2f}  CI [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]  "
              f"wins {p['wins'][0]} -> {p['wins'][1]}  n={p['n']}")


if __name__ == "__main__":
    main()
