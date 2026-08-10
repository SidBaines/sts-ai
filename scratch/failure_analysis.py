#!/usr/bin/env python3
"""Failure-mode analysis for Gemma 4 E4B thinking rollouts.

Reads completed rollout pairs from:
  data/rollouts/e4b_think_perf/vllm_gemma_4_E4B_it_thinking_8192/

Writes:
  scratch/e4b/failure_analysis.json
  scratch/e4b/hp_erosion_curve.png
  scratch/e4b/largest_final_hp_drop_hist.png
  scratch/e4b/death_final_floor_distribution.png

Usage:
  PYTHONPATH=src .venv/bin/python scratch/failure_analysis.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/rollouts/e4b_think_perf/vllm_gemma_4_E4B_it_thinking_8192"
OUT_DIR = REPO_ROOT / "scratch/e4b"
OUT_JSON = OUT_DIR / "failure_analysis.json"

# Keep Matplotlib's cache inside scratch, then remove it at the end. This avoids
# writing to user-level cache directories on machines where they are read-only.
MPL_CONFIG_DIR = OUT_DIR / ".matplotlib-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
XDG_CACHE_DIR = MPL_CONFIG_DIR / "xdg-cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402


OUTCOME_CATEGORIES = ["death", "victory", "budget_truncated", "agent_invalid", "other"]
OUTCOME_REPORT_LABELS = {
    "death": "death",
    "victory": "victory",
    "budget_truncated": "budget_truncated[stopped_reason=max_decisions]",
    "agent_invalid": "agent_invalid",
    "other": "other",
}
DANGER_KEYWORDS = [
    "lethal",
    "die",
    "dies",
    "dying",
    "death",
    "low hp",
    "low health",
    "danger",
    "dangerous",
    "risk",
    "risky",
    "survive",
    "survival",
    "block",
    "incoming",
    "fatal",
]


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def pct(num: int | float, den: int | float) -> float:
    return (100.0 * num / den) if den else 0.0


def percentile(values: list[float] | list[int], q: float) -> float | None:
    """Linear percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stats(values: list[int] | list[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if safe_float(v) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p25": percentile(vals, 25),
        "p75": percentile(vals, 75),
        "min": min(vals),
        "max": max(vals),
    }


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(num: int | float, den: int | float, digits: int = 1) -> str:
    return f"{pct(num, den):.{digits}f}%"


def seed_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"seed_(\d+)_r0", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def categorize_outcome(meta: dict[str, Any]) -> str:
    outcome = str(meta.get("outcome", ""))
    stopped_reason = str(meta.get("stopped_reason", ""))
    if outcome.endswith("PLAYER_LOSS"):
        return "death"
    if outcome.endswith("PLAYER_VICTORY"):
        return "victory"
    if stopped_reason == "max_decisions":
        return "budget_truncated"
    if stopped_reason == "agent_invalid":
        return "agent_invalid"
    return "other"


def compact_counter(counter: Counter[Any], max_items: int = 12) -> str:
    if not counter:
        return "(none)"
    items = counter.most_common(max_items)
    text = ", ".join(f"{k}:{v}" for k, v in items)
    remaining = len(counter) - len(items)
    if remaining > 0:
        text += f", ... (+{remaining} more)"
    return text


def binned_hist(values: list[int], bins: list[tuple[str, int, int]], width: int = 28) -> list[str]:
    if not values:
        return []
    counts = []
    for label, lo, hi in bins:
        counts.append((label, sum(1 for v in values if lo <= v <= hi)))
    max_count = max((c for _, c in counts), default=0)
    lines = []
    for label, count in counts:
        bar_len = int(round(width * count / max_count)) if max_count else 0
        lines.append(f"  {label:>7}: {count:>3} {'#' * bar_len}")
    return lines


def exact_hist_lines(counter: Counter[int], width: int = 28) -> list[str]:
    if not counter:
        return []
    max_count = max(counter.values())
    lines = []
    for key in range(min(counter), max(counter) + 1):
        count = counter.get(key, 0)
        bar_len = int(round(width * count / max_count)) if max_count else 0
        lines.append(f"  {key:>2}: {count:>3} {'#' * bar_len}")
    return lines


def find_pairs() -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    jsonl_files = sorted(DATA_DIR.glob("seed_*_r0.jsonl"), key=seed_sort_key)
    meta_files = sorted(DATA_DIR.glob("seed_*_r0.meta.json"), key=seed_sort_key)
    meta_by_stem = {p.name.replace(".meta.json", ""): p for p in meta_files}
    jsonl_by_stem = {p.name.replace(".jsonl", ""): p for p in jsonl_files}

    pairs = []
    for stem, jsonl_path in jsonl_by_stem.items():
        meta_path = meta_by_stem.get(stem)
        if meta_path is not None:
            pairs.append((jsonl_path, meta_path))

    skips = {
        "jsonl_files": len(jsonl_files),
        "meta_files": len(meta_files),
        "jsonl_missing_meta": sum(1 for stem in jsonl_by_stem if stem not in meta_by_stem),
        "meta_missing_jsonl": sum(1 for stem in meta_by_stem if stem not in jsonl_by_stem),
        "bad_meta_files": 0,
        "bad_jsonl_lines": 0,
        "rollouts_with_no_records": 0,
        "records_missing_floor_or_hp": 0,
    }
    return sorted(pairs, key=lambda pair: seed_sort_key(pair[0])), skips


def classify_campfire_action(description: str) -> str:
    desc = description.strip().lower()
    if desc.startswith("rest"):
        return "rest"
    if desc.startswith("smith"):
        return "smith"
    if desc.startswith("recall"):
        return "recall"
    return "other"


def enemy_details_from_record(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    combat = ((record.get("state") or {}).get("combat") or {})
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return [], [], []

    parsed = [e for e in enemies if isinstance(e, dict)]
    living = [e for e in parsed if e.get("alive") is not False]
    selected = living or parsed
    names = [str(e.get("name") or "UNKNOWN") for e in selected]
    intents = [str(e.get("intent") or "UNKNOWN") for e in selected]
    contexts = []
    for e in selected:
        name = str(e.get("name") or "UNKNOWN")
        intent = str(e.get("intent") or "UNKNOWN")
        damage = safe_int(e.get("intent_damage"), 0)
        hits = safe_int(e.get("intent_hits"), 0)
        contexts.append(f"{name}|{intent}|dmg={damage}|hits={hits}")
    return names, intents, contexts


def hp_death_dynamics(meta: dict[str, Any]) -> dict[str, Any]:
    hp_raw = meta.get("hp_trajectory") or []
    hp = [safe_int(v) for v in hp_raw]
    hp = [v for v in hp if v is not None]
    max_hp = safe_int(meta.get("max_hp"))
    if not max_hp or max_hp <= 0:
        max_hp = max(hp) if hp else 1

    start = max(1, len(hp) - 15)
    drops = [max(0, hp[i - 1] - hp[i]) for i in range(start, len(hp))]
    largest_drop = max(drops, default=0)
    largest_drop_frac = largest_drop / max_hp if max_hp else 0.0
    death_type = "burst" if largest_drop_frac >= 0.40 else "attrition"

    hp_before_death = None
    if len(hp) >= 2:
        hp_before_death = hp[-2]
    elif len(hp) == 1:
        hp_before_death = hp[0]

    return {
        "death_type": death_type,
        "largest_final_drop": largest_drop,
        "largest_final_drop_frac": largest_drop_frac,
        "largest_final_drop_pct": largest_drop_frac * 100.0,
        "hp_before_death": hp_before_death,
        "max_hp": max_hp,
        "hp_trajectory_len": len(hp),
    }


def danger_awareness(agent: dict[str, Any]) -> tuple[bool, list[str]]:
    text = f"{agent.get('reasoning') or ''}\n{agent.get('thinking') or ''}".lower()
    hits = [kw for kw in DANGER_KEYWORDS if kw in text]
    return bool(hits), hits


def process_jsonl(
    jsonl_path: Path,
    global_counts: dict[str, Any],
) -> dict[str, Any]:
    phase_counts: Counter[str] = Counter()
    last_records: deque[dict[str, Any]] = deque(maxlen=5)
    per_floor_hp: dict[int, int] = {}
    n_records = 0
    n_invalid = 0

    try:
        handle = jsonl_path.open("r", encoding="utf-8")
    except OSError:
        global_counts["bad_jsonl_files"] += 1
        return {
            "record_count": 0,
            "phase_counts": {},
            "invalid_count": 0,
            "last_records": [],
            "per_floor_hp": {},
        }

    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                global_counts["bad_jsonl_lines"] += 1
                global_counts["bad_jsonl_line_examples"].append(f"{jsonl_path.name}:{line_no}")
                continue

            n_records += 1
            last_records.append(record)

            phase = str(record.get("phase") or "unknown")
            phase_counts[phase] += 1
            global_counts["phase_counts"][phase] += 1

            state = record.get("state") or {}
            floor = safe_int(state.get("floor"))
            cur_hp = safe_int(state.get("cur_hp"))
            if floor is not None and cur_hp is not None:
                per_floor_hp[floor] = cur_hp
            else:
                global_counts["records_missing_floor_or_hp"] += 1

            agent = record.get("agent") or {}
            if agent.get("valid") is False:
                n_invalid += 1
                global_counts["invalid_decisions_jsonl"] += 1

            for key in ("thinking_tokens", "completion_tokens", "prompt_tokens"):
                value = safe_int(agent.get(key))
                if value is not None:
                    global_counts["token_values"][key].append(value)

            room = str(state.get("room") or "")
            screen = str(state.get("screen_state") or "")
            if room == "Room.REST" and screen == "ScreenState.REST_ROOM":
                desc = str((record.get("selected_action") or {}).get("description") or "")
                global_counts["campfire_actions"][classify_campfire_action(desc)] += 1

    if n_records == 0:
        global_counts["rollouts_with_no_records"] += 1

    return {
        "record_count": n_records,
        "phase_counts": dict(phase_counts),
        "invalid_count": n_invalid,
        "last_records": list(last_records),
        "per_floor_hp": per_floor_hp,
    }


def build_analysis() -> tuple[dict[str, Any], list[str]]:
    pairs, skips = find_pairs()
    global_counts: dict[str, Any] = {
        "phase_counts": Counter(),
        "token_values": defaultdict(list),
        "campfire_actions": Counter(),
        "bad_jsonl_files": 0,
        "bad_jsonl_lines": 0,
        "bad_jsonl_line_examples": [],
        "records_missing_floor_or_hp": 0,
        "rollouts_with_no_records": 0,
        "invalid_decisions_jsonl": 0,
    }

    rollouts = []
    hp_by_floor: dict[int, list[int]] = defaultdict(list)

    outcome_counts: Counter[str] = Counter()
    stopped_reason_counts: Counter[str] = Counter()
    final_act_counter: Counter[int] = Counter()
    final_floor_counter: Counter[int] = Counter()
    death_floor_counter: Counter[int] = Counter()
    death_band_counts: Counter[str] = Counter()
    death_types: Counter[str] = Counter()
    terminal_phase_counts: Counter[str] = Counter()
    terminal_context_counts: Counter[str] = Counter()
    final_enemy_counts: Counter[str] = Counter()
    final_intent_counts: Counter[str] = Counter()
    final_enemy_context_counts: Counter[str] = Counter()
    final_encounter_counts: Counter[str] = Counter()
    danger_keyword_counts: Counter[str] = Counter()

    final_acts: list[int] = []
    final_floors: list[int] = []
    n_decisions_meta: list[int] = []
    n_combat_meta: list[int] = []
    n_ooc_meta: list[int] = []
    n_invalid_meta: list[int] = []
    n_records_jsonl: list[int] = []
    n_combat_jsonl: list[int] = []
    n_ooc_jsonl: list[int] = []
    largest_drop_pcts: list[float] = []
    largest_drop_pcts_by_type: dict[str, list[float]] = {"burst": [], "attrition": []}
    hp_before_death_values: list[int] = []
    final_death_tokens: dict[str, list[int]] = defaultdict(list)
    deaths_with_danger_awareness = 0
    death_rollout_rows = []

    for jsonl_path, meta_path in pairs:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skips["bad_meta_files"] += 1
            continue

        parsed = process_jsonl(jsonl_path, global_counts)
        for floor, hp in parsed["per_floor_hp"].items():
            hp_by_floor[floor].append(hp)

        category = categorize_outcome(meta)
        outcome_counts[category] += 1
        stopped_reason_counts[str(meta.get("stopped_reason") or "unknown")] += 1

        final_act = safe_int(meta.get("final_act"))
        final_floor = safe_int(meta.get("final_floor"))
        if final_act is not None:
            final_acts.append(final_act)
            final_act_counter[final_act] += 1
        if final_floor is not None:
            final_floors.append(final_floor)
            final_floor_counter[final_floor] += 1

        for key, target in (
            ("n_decisions", n_decisions_meta),
            ("n_combat", n_combat_meta),
            ("n_out_of_combat", n_ooc_meta),
            ("n_invalid", n_invalid_meta),
        ):
            value = safe_int(meta.get(key))
            if value is not None:
                target.append(value)

        n_records_jsonl.append(parsed["record_count"])
        n_combat_jsonl.append(int(parsed["phase_counts"].get("combat", 0)))
        n_ooc_jsonl.append(int(parsed["phase_counts"].get("out_of_combat", 0)))

        row = {
            "seed": safe_int(meta.get("world_seed")),
            "jsonl": jsonl_path.name,
            "meta": meta_path.name,
            "category": category,
            "outcome": meta.get("outcome"),
            "stopped_reason": meta.get("stopped_reason"),
            "final_act": final_act,
            "final_floor": final_floor,
            "final_hp": safe_int(meta.get("final_hp")),
            "max_hp": safe_int(meta.get("max_hp")),
            "n_decisions_meta": safe_int(meta.get("n_decisions")),
            "n_records_jsonl": parsed["record_count"],
            "jsonl_invalid_count": parsed["invalid_count"],
        }
        rollouts.append(row)

        if category != "death":
            continue

        if final_floor is not None:
            death_floor_counter[final_floor] += 1
            if final_floor < 16:
                death_band_counts["before_act1_boss"] += 1
            elif final_floor in (16, 17):
                death_band_counts["at_act1_boss"] += 1
            elif final_floor >= 18:
                death_band_counts["act2_plus"] += 1
            else:
                death_band_counts["other"] += 1

        dynamics = hp_death_dynamics(meta)
        death_types[dynamics["death_type"]] += 1
        largest_drop_pcts.append(float(dynamics["largest_final_drop_pct"]))
        largest_drop_pcts_by_type[dynamics["death_type"]].append(float(dynamics["largest_final_drop_pct"]))
        if dynamics["hp_before_death"] is not None:
            hp_before_death_values.append(int(dynamics["hp_before_death"]))

        last_records = parsed["last_records"]
        terminal = last_records[-1] if last_records else {}
        terminal_state = terminal.get("state") or {}
        terminal_agent = terminal.get("agent") or {}
        terminal_phase = str(terminal.get("phase") or "missing")
        terminal_room = str(terminal_state.get("room") or "missing")
        terminal_screen = str(terminal_state.get("screen_state") or "missing")
        terminal_phase_counts[terminal_phase] += 1
        terminal_context_counts[f"{terminal_phase}|{terminal_room}|{terminal_screen}"] += 1

        for key in ("thinking_tokens", "completion_tokens", "prompt_tokens"):
            value = safe_int(terminal_agent.get(key))
            if value is not None:
                final_death_tokens[key].append(value)

        aware, keyword_hits = danger_awareness(terminal_agent)
        if aware:
            deaths_with_danger_awareness += 1
            danger_keyword_counts.update(keyword_hits)

        enemy_names: list[str] = []
        enemy_intents: list[str] = []
        enemy_contexts: list[str] = []
        encounter = "missing"
        if terminal_phase == "combat":
            enemy_names, enemy_intents, enemy_contexts = enemy_details_from_record(terminal)
            if enemy_names:
                final_enemy_counts.update(set(enemy_names))
                final_intent_counts.update(enemy_intents)
                final_enemy_context_counts.update(enemy_contexts)
                encounter = " + ".join(sorted(enemy_names))
                final_encounter_counts[encounter] += 1
            else:
                final_encounter_counts["combat_unknown_enemies"] += 1

        death_rollout_rows.append(
            {
                **row,
                **dynamics,
                "terminal_phase": terminal_phase,
                "terminal_room": terminal_room,
                "terminal_screen": terminal_screen,
                "terminal_enemies": enemy_names,
                "terminal_enemy_intents": enemy_intents,
                "terminal_encounter": encounter,
                "danger_awareness": aware,
                "danger_keywords": keyword_hits,
            }
        )

    skips.update(
        {
            "bad_jsonl_files": global_counts["bad_jsonl_files"],
            "bad_jsonl_lines": global_counts["bad_jsonl_lines"],
            "bad_jsonl_line_examples": global_counts["bad_jsonl_line_examples"][:10],
            "records_missing_floor_or_hp": global_counts["records_missing_floor_or_hp"],
            "rollouts_with_no_records": global_counts["rollouts_with_no_records"],
        }
    )

    n_rollouts = len(rollouts)
    n_deaths = outcome_counts["death"]
    hp_erosion = []
    for floor in sorted(hp_by_floor):
        values = hp_by_floor[floor]
        hp_erosion.append(
            {
                "floor": floor,
                "n_rollouts": len(values),
                "mean": statistics.fmean(values),
                "p25": percentile(values, 25),
                "p75": percentile(values, 75),
            }
        )

    every_death_classified = sum(death_types.values()) == n_deaths
    assert n_rollouts > 0, "expected at least one completed rollout"
    assert every_death_classified, "every death should be classified burst or attrition"

    analysis = {
        "input_dir": str(DATA_DIR.relative_to(REPO_ROOT)),
        "output_dir": str(OUT_DIR.relative_to(REPO_ROOT)),
        "n_completed_rollouts": n_rollouts,
        "skips": skips,
        "outcome_category_definitions": {
            "death": "outcome ends with PLAYER_LOSS",
            "victory": "outcome ends with PLAYER_VICTORY",
            "budget_truncated": "stopped_reason == max_decisions and not death/victory",
            "agent_invalid": "stopped_reason == agent_invalid and not death/victory/budget_truncated",
            "other": "anything else",
        },
        "outcome_counts": {cat: int(outcome_counts.get(cat, 0)) for cat in OUTCOME_CATEGORIES},
        "stopped_reason_counts": dict(stopped_reason_counts),
        "depth": {
            "final_act_stats": stats(final_acts),
            "final_floor_stats": stats(final_floors),
            "final_act_hist": {str(k): int(v) for k, v in sorted(final_act_counter.items())},
            "final_floor_hist": {str(k): int(v) for k, v in sorted(final_floor_counter.items())},
            "reached_act2": {
                "count": sum(1 for a in final_acts if a >= 2),
                "fraction": (sum(1 for a in final_acts if a >= 2) / n_rollouts) if n_rollouts else 0.0,
            },
            "reached_act3": {
                "count": sum(1 for a in final_acts if a >= 3),
                "fraction": (sum(1 for a in final_acts if a >= 3) / n_rollouts) if n_rollouts else 0.0,
            },
        },
        "death_floor_concentration": {
            "n_deaths": n_deaths,
            "final_floor_hist": {str(k): int(v) for k, v in sorted(death_floor_counter.items())},
            "bands": {
                name: {
                    "count": int(death_band_counts.get(name, 0)),
                    "fraction": (death_band_counts.get(name, 0) / n_deaths) if n_deaths else 0.0,
                }
                for name in ("before_act1_boss", "at_act1_boss", "act2_plus", "other")
            },
        },
        "hp_death_dynamics": {
            "death_type_counts": dict(death_types),
            "largest_final_drop_pct_stats": stats(largest_drop_pcts),
            "largest_final_drop_pct_by_type_stats": {
                key: stats(values) for key, values in largest_drop_pcts_by_type.items()
            },
            "hp_before_death_stats": stats(hp_before_death_values),
            "death_rollouts": death_rollout_rows,
        },
        "hp_erosion_curve": hp_erosion,
        "terminal_death_context": {
            "terminal_phase_counts": dict(terminal_phase_counts),
            "terminal_context_counts": dict(terminal_context_counts),
            "final_enemy_counts_deaths_involving_enemy": dict(final_enemy_counts),
            "final_intent_counts": dict(final_intent_counts),
            "final_enemy_context_counts": dict(final_enemy_context_counts),
            "final_encounter_counts": dict(final_encounter_counts),
        },
        "decision_mix_behavior": {
            "records_total_jsonl": int(sum(n_records_jsonl)),
            "phase_counts_jsonl": dict(global_counts["phase_counts"]),
            "phase_mean_per_rollout_jsonl": {
                "combat": statistics.fmean(n_combat_jsonl) if n_combat_jsonl else 0.0,
                "out_of_combat": statistics.fmean(n_ooc_jsonl) if n_ooc_jsonl else 0.0,
            },
            "decisions_per_rollout_meta_stats": stats(n_decisions_meta),
            "combat_decisions_per_rollout_meta_stats": stats(n_combat_meta),
            "out_of_combat_decisions_per_rollout_meta_stats": stats(n_ooc_meta),
            "invalid_decisions_meta_total": int(sum(n_invalid_meta)),
            "invalid_decisions_jsonl_total": int(global_counts["invalid_decisions_jsonl"]),
            "rollouts_with_invalid_decisions_meta": sum(1 for v in n_invalid_meta if v > 0),
            "campfire_actions": dict(global_counts["campfire_actions"]),
        },
        "reasoning_patterns": {
            "overall_token_stats": {key: stats(values) for key, values in global_counts["token_values"].items()},
            "final_death_token_stats": {key: stats(values) for key, values in final_death_tokens.items()},
            "danger_awareness": {
                "deaths_with_keyword": deaths_with_danger_awareness,
                "fraction": (deaths_with_danger_awareness / n_deaths) if n_deaths else 0.0,
                "keywords": DANGER_KEYWORDS,
                "keyword_hit_counts": dict(danger_keyword_counts),
            },
        },
        "rollouts": rollouts,
    }

    report = format_report(analysis)
    return analysis, report


def format_report(analysis: dict[str, Any]) -> list[str]:
    n = analysis["n_completed_rollouts"]
    skips = analysis["skips"]
    outcomes = analysis["outcome_counts"]
    depth = analysis["depth"]
    deaths = analysis["death_floor_concentration"]
    hp_dyn = analysis["hp_death_dynamics"]
    decision = analysis["decision_mix_behavior"]
    reasoning = analysis["reasoning_patterns"]
    terminal = analysis["terminal_death_context"]

    final_act_stats = depth["final_act_stats"]
    final_floor_stats = depth["final_floor_stats"]
    death_drop_stats = hp_dyn["largest_final_drop_pct_stats"]
    hp_before_stats = hp_dyn["hp_before_death_stats"]
    decisions_stats = decision["decisions_per_rollout_meta_stats"]

    final_floor_values = []
    for floor, count in analysis["depth"]["final_floor_hist"].items():
        final_floor_values.extend([int(floor)] * int(count))

    death_floor_values = []
    for floor, count in deaths["final_floor_hist"].items():
        death_floor_values.extend([int(floor)] * int(count))

    lines = [
        "Failure Analysis: gemma-4-E4B-it thinking",
        f"Input: {analysis['input_dir']}",
        (
            f"Loaded: {n} completed rollouts "
            f"({skips['jsonl_files']} jsonl, {skips['meta_files']} meta; "
            f"skipped jsonl_missing_meta={skips['jsonl_missing_meta']}, "
            f"meta_missing_jsonl={skips['meta_missing_jsonl']}, bad_meta={skips['bad_meta_files']}, "
            f"bad_jsonl_lines={skips['bad_jsonl_lines']})"
        ),
        "",
        "Outcome & Depth",
        "  outcomes: "
        + ", ".join(f"{OUTCOME_REPORT_LABELS[cat]}={outcomes.get(cat, 0)}" for cat in OUTCOME_CATEGORIES),
        (
            "  final_act: "
            f"mean={fmt_num(final_act_stats['mean'])}, median={fmt_num(final_act_stats['median'])}, "
            f"max={fmt_num(final_act_stats['max'], 0)} | hist {depth['final_act_hist']}"
        ),
        (
            "  final_floor: "
            f"mean={fmt_num(final_floor_stats['mean'])}, median={fmt_num(final_floor_stats['median'])}, "
            f"max={fmt_num(final_floor_stats['max'], 0)}"
        ),
    ]
    lines.extend(
        binned_hist(
            final_floor_values,
            [
                ("0-5", 0, 5),
                ("6-10", 6, 10),
                ("11-15", 11, 15),
                ("16-17", 16, 17),
                ("18-22", 18, 22),
                ("23-27", 23, 27),
                ("28-33", 28, 33),
            ],
        )
    )
    act2 = depth["reached_act2"]
    act3 = depth["reached_act3"]
    lines.append(
        f"  reached Act 2: {act2['count']}/{n} ({act2['fraction'] * 100.0:.1f}%); "
        f"Act 3: {act3['count']}/{n} ({act3['fraction'] * 100.0:.1f}%)"
    )

    lines.extend(["", "Death-Floor Concentration"])
    n_deaths = deaths["n_deaths"]
    lines.append(f"  deaths: {n_deaths}")
    lines.append("  death final_floor histogram:")
    lines.extend(exact_hist_lines(Counter(death_floor_values)))
    for label, title in (
        ("before_act1_boss", "before Act-1 boss (<16)"),
        ("at_act1_boss", "at Act-1 boss (16/17)"),
        ("act2_plus", "Act 2+ (>=18)"),
    ):
        band = deaths["bands"][label]
        lines.append(f"  {title}: {band['count']}/{n_deaths} ({band['fraction'] * 100.0:.1f}%)")

    death_type_counts = hp_dyn["death_type_counts"]
    lines.extend(
        [
            "",
            "HP / Death Dynamics",
            (
                "  burst/attrition: "
                f"burst={death_type_counts.get('burst', 0)}, attrition={death_type_counts.get('attrition', 0)} "
                "(burst threshold: largest last-15-decision HP drop >=40% max_hp)"
            ),
            (
                "  largest final HP drop (% max_hp): "
                f"mean={fmt_num(death_drop_stats['mean'])}, median={fmt_num(death_drop_stats['median'])}, "
                f"p25={fmt_num(death_drop_stats['p25'])}, p75={fmt_num(death_drop_stats['p75'])}, "
                f"max={fmt_num(death_drop_stats['max'])}"
            ),
            (
                "  HP just before death: "
                f"mean={fmt_num(hp_before_stats['mean'])}, median={fmt_num(hp_before_stats['median'])}, "
                f"p25={fmt_num(hp_before_stats['p25'])}, p75={fmt_num(hp_before_stats['p75'])}, "
                f"max={fmt_num(hp_before_stats['max'], 0)}"
            ),
            "  HP erosion curve (last observed state.cur_hp per rollout/floor):",
            "    floor   n    mean    p25    p75",
        ]
    )
    for row in analysis["hp_erosion_curve"]:
        lines.append(
            f"    {row['floor']:>5} {row['n_rollouts']:>3} "
            f"{row['mean']:>7.1f} {row['p25']:>6.1f} {row['p75']:>6.1f}"
        )

    lines.extend(
        [
            "",
            "What Precedes Death",
            "  terminal phase: " + compact_counter(Counter(terminal["terminal_phase_counts"])),
            "  terminal room/screen: " + compact_counter(Counter(terminal["terminal_context_counts"]), max_items=8),
            "  final combat enemies (deaths involving enemy): "
            + compact_counter(Counter(terminal["final_enemy_counts_deaths_involving_enemy"]), max_items=10),
            "  final combat encounters: " + compact_counter(Counter(terminal["final_encounter_counts"]), max_items=10),
            "  final combat enemy intent contexts: "
            + compact_counter(Counter(terminal["final_enemy_context_counts"]), max_items=8),
        ]
    )

    phase_counts = Counter(decision["phase_counts_jsonl"])
    total_records = decision["records_total_jsonl"]
    lines.extend(
        [
            "",
            "Decision Mix & Behavior",
            (
                f"  decisions: total_jsonl={total_records}, "
                f"mean_per_rollout={fmt_num(decisions_stats['mean'])}, "
                f"median={fmt_num(decisions_stats['median'])}"
            ),
            (
                f"  combat={phase_counts.get('combat', 0)} "
                f"({fmt_pct(phase_counts.get('combat', 0), total_records)}), "
                f"out_of_combat={phase_counts.get('out_of_combat', 0)} "
                f"({fmt_pct(phase_counts.get('out_of_combat', 0), total_records)})"
            ),
            (
                "  per-rollout mean: "
                f"combat={decision['phase_mean_per_rollout_jsonl']['combat']:.1f}, "
                f"out_of_combat={decision['phase_mean_per_rollout_jsonl']['out_of_combat']:.1f}"
            ),
            (
                "  invalid decisions: "
                f"meta_total={decision['invalid_decisions_meta_total']}, "
                f"jsonl_agent.valid_false={decision['invalid_decisions_jsonl_total']}, "
                f"rollouts_with_invalid={decision['rollouts_with_invalid_decisions_meta']}"
            ),
            "  campfire actions: " + compact_counter(Counter(decision["campfire_actions"])),
        ]
    )

    overall_tokens = reasoning["overall_token_stats"]
    final_death_tokens = reasoning["final_death_token_stats"]
    danger = reasoning["danger_awareness"]
    lines.extend(["", "Reasoning Patterns"])
    for key in ("thinking_tokens", "completion_tokens"):
        st = overall_tokens.get(key, {})
        lines.append(
            f"  overall {key}: n={st.get('n', 0)}, mean={fmt_num(st.get('mean'))}, "
            f"median={fmt_num(st.get('median'))}, p75={fmt_num(st.get('p75'))}, max={fmt_num(st.get('max'), 0)}"
        )
    for key in ("thinking_tokens", "completion_tokens"):
        st = final_death_tokens.get(key, {})
        lines.append(
            f"  final-death {key}: n={st.get('n', 0)}, mean={fmt_num(st.get('mean'))}, "
            f"median={fmt_num(st.get('median'))}, p75={fmt_num(st.get('p75'))}, max={fmt_num(st.get('max'), 0)}"
        )
    lines.append(
        f"  last-reasoning danger awareness: {danger['deaths_with_keyword']}/{n_deaths} "
        f"({danger['fraction'] * 100.0:.1f}%)"
    )
    lines.append("  danger keyword hits: " + compact_counter(Counter(danger["keyword_hit_counts"])))

    lines.extend(
        [
            "",
            "Wrote",
            f"  {OUT_JSON.relative_to(REPO_ROOT)}",
            f"  {(OUT_DIR / 'hp_erosion_curve.png').relative_to(REPO_ROOT)}",
            f"  {(OUT_DIR / 'largest_final_hp_drop_hist.png').relative_to(REPO_ROOT)}",
            f"  {(OUT_DIR / 'death_final_floor_distribution.png').relative_to(REPO_ROOT)}",
        ]
    )
    return lines


def plot_hp_erosion(hp_erosion: list[dict[str, Any]], out_path: Path) -> None:
    floors = [row["floor"] for row in hp_erosion]
    means = [row["mean"] for row in hp_erosion]
    p25 = [row["p25"] for row in hp_erosion]
    p75 = [row["p75"] for row in hp_erosion]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(floors, means, color="#1f77b4", marker="o", linewidth=2, label="mean HP")
    ax.fill_between(floors, p25, p75, color="#1f77b4", alpha=0.20, label="25th-75th percentile")
    ax.axvspan(16, 17, color="#d62728", alpha=0.10, label="Act-1 boss floors")
    ax.axvline(18, color="#666666", linestyle="--", linewidth=1, label="Act 2+")
    ax.set_title("HP erosion by floor")
    ax.set_xlabel("Floor")
    ax.set_ylabel("Player HP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_largest_drop_hist(
    largest_drop_pcts_by_type: dict[str, list[float]],
    out_path: Path,
) -> None:
    attrition = largest_drop_pcts_by_type.get("attrition", [])
    burst = largest_drop_pcts_by_type.get("burst", [])
    bins = list(range(0, 105, 5))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [attrition, burst],
        bins=bins,
        stacked=True,
        color=["#4c78a8", "#e45756"],
        label=["attrition", "burst"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(40, color="#222222", linestyle="--", linewidth=1.5, label="burst threshold")
    ax.set_title("Largest HP drop in final 15 decisions")
    ax.set_xlabel("Largest drop (% of max HP)")
    ax.set_ylabel("Death rollouts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_death_floor_hist(death_floor_hist: dict[str, int], out_path: Path) -> None:
    floors = sorted(int(k) for k in death_floor_hist)
    counts = [death_floor_hist[str(floor)] for floor in floors]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(floors, counts, width=0.85, color="#d95f02", edgecolor="white", linewidth=0.4)
    ax.axvspan(15.5, 17.5, color="#7570b3", alpha=0.18, label="Act-1 boss floors 16-17")
    ax.axvline(18, color="#333333", linestyle="--", linewidth=1, label="Act 2+")
    ax.set_title("Death final-floor distribution")
    ax.set_xlabel("Final floor")
    ax.set_ylabel("Death rollouts")
    ax.set_xticks(range(min(floors), max(floors) + 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_outputs(analysis: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

    plot_hp_erosion(analysis["hp_erosion_curve"], OUT_DIR / "hp_erosion_curve.png")

    largest_by_type = {
        "burst": [],
        "attrition": [],
    }
    for row in analysis["hp_death_dynamics"]["death_rollouts"]:
        largest_by_type[row["death_type"]].append(row["largest_final_drop_pct"])
    plot_largest_drop_hist(largest_by_type, OUT_DIR / "largest_final_hp_drop_hist.png")
    plot_death_floor_hist(
        analysis["death_floor_concentration"]["final_floor_hist"],
        OUT_DIR / "death_final_floor_distribution.png",
    )


def cleanup_matplotlib_cache() -> None:
    try:
        shutil.rmtree(MPL_CONFIG_DIR)
    except OSError:
        pass


def main() -> None:
    analysis, report_lines = build_analysis()
    write_outputs(analysis)
    print("\n".join(report_lines))
    cleanup_matplotlib_cache()


if __name__ == "__main__":
    main()
