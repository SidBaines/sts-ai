#!/usr/bin/env python
"""Per-model comparison report over a sweep tree.

Reads <sweep-dir>/<label>/seed_*.jsonl (+ .meta.json sidecars) and prints a table
of outcomes, behaviour proxies, reasoning length/latency, and invalid-JSON rate —
one row per model/mode label. Intended to inform RL planning (which models, what
headroom, what the eval suite should measure).

    PYTHONPATH=src .venv/bin/python scripts/compare_models.py data/rollouts/sweep
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sts_ai.affordances import BLOCK_BASE

_PLAY_RE = re.compile(
    r"^play\s+(.*?)\s+\(cost\s+(\S+?)\)(?:\s*->\s*(.*?))?(?:\s*\(deal\s+(\d+)[^)]*\))?\s*$"
)


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _chosen_is_block(desc: str) -> bool:
    m = _PLAY_RE.match(desc)
    if not m:
        return False
    name = m.group(1)
    base = name[:-1] if name.endswith("+") else name
    return base in BLOCK_BASE or base == "Entrench"


def _chosen_lethal(desc: str, enemies: list[dict[str, Any]]) -> bool:
    m = _PLAY_RE.match(desc)
    if not m or m.group(4) is None:
        return False
    dmg, target = int(m.group(4)), (m.group(3) or "").strip()
    for e in enemies:
        if str(e.get("name")) == target and e.get("alive"):
            return dmg >= int(e.get("cur_hp", 0)) + int(e.get("block", 0))
    return False


def summarize_label(label_dir: Path) -> dict[str, Any]:
    metas = [json.loads(p.read_text()) for p in sorted(label_dir.glob("seed_*.meta.json"))]
    agg: dict[str, Any] = {
        "label": label_dir.name,
        "rollouts": len(metas),
        "wins": 0, "floors": [], "hp_frac": [], "decisions": [],
        "invalid_total": 0, "decision_total": 0,
        "completion_tokens": [], "thinking_tokens": [], "latency_s": [],
        "fb_possible": 0, "fb_combat_incoming": 0, "fb_forgone": 0,
        "lethal_avail": 0, "lethal_taken": 0, "combat_decisions": 0,
    }
    for m in metas:
        agg["wins"] += 1 if "VICTORY" in str(m.get("outcome", "")) else 0
        agg["floors"].append(m.get("final_floor", 0))
        if m.get("max_hp"):
            agg["hp_frac"].append(m.get("final_hp", 0) / m["max_hp"])
        agg["decisions"].append(m.get("n_decisions", 0))
        agg["invalid_total"] += m.get("n_invalid", 0)
        agg["decision_total"] += m.get("n_decisions", 0)

    for jsonl in sorted(label_dir.glob("seed_*.jsonl")):
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ag = rec.get("agent", {})
            agg["completion_tokens"].append(ag.get("completion_tokens", 0))
            agg["thinking_tokens"].append(ag.get("thinking_tokens", 0))
            agg["latency_s"].append(ag.get("latency_s", 0.0))
            if not rec.get("action_executed", True):
                continue
            if rec.get("phase") != "combat":
                continue
            agg["combat_decisions"] += 1
            aff = rec.get("affordances", {})
            chosen = rec.get("selected_action", {}).get("description", "")
            enemies = rec.get("state", {}).get("combat", {}).get("enemies", [])
            if aff.get("incoming_damage_total", 0) > 0 and aff.get("full_block_possible"):
                agg["fb_possible"] += 1
                if not _chosen_is_block(chosen):
                    agg["fb_forgone"] += 1
            if aff.get("single_target_lethal_available"):
                agg["lethal_avail"] += 1
                if _chosen_lethal(chosen, enemies):
                    agg["lethal_taken"] += 1
    return agg


def render_row(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": a["label"],
        "n": a["rollouts"],
        "win_rate": _rate(a["wins"], a["rollouts"]),
        "avg_floor": _mean(a["floors"]),
        "avg_hp_frac": _mean([x * 100 for x in a["hp_frac"]]),
        "avg_decisions": _mean([float(x) for x in a["decisions"]]),
        "invalid_rate": _rate(a["invalid_total"], a["decision_total"]),
        "avg_compl_tok": _mean([float(x) for x in a["completion_tokens"]]),
        "avg_think_tok": _mean([float(x) for x in a["thinking_tokens"]]),
        "avg_latency_s": _mean(a["latency_s"]),
        "fullblock_forgone": _rate(a["fb_forgone"], a["fb_possible"]),
        "lethal_taken": _rate(a["lethal_taken"], a["lethal_avail"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-model sweep comparison report.")
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Also dump the raw rows as JSON.")
    args = parser.parse_args()

    label_dirs = sorted(p for p in args.sweep_dir.iterdir() if p.is_dir() and any(p.glob("seed_*.meta.json")))
    rows = [render_row(summarize_label(d)) for d in label_dirs]
    if not rows:
        print(f"no labelled rollouts with meta sidecars under {args.sweep_dir}")
        return

    cols = ["label", "n", "win_rate", "avg_floor", "avg_hp_frac", "avg_decisions",
            "invalid_rate", "avg_compl_tok", "avg_think_tok", "avg_latency_s",
            "fullblock_forgone", "lethal_taken"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if args.json:
        print("\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
