"""Matched three-arm fight comparisons, with explicit missingness and provenance."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import statistics
from typing import Any

from sts_ai import fight_lab as lab
from sts_ai.fight_analysis import analyze, cluster_interval
from sts_ai.provenance import file_sha256


ARM_NAMES = ("old_format", "new_no_thinking", "new_thinking")
ARM_LABELS = {
    "old_format": "Old format · no thinking",
    "new_no_thinking": "New format · no thinking",
    "new_thinking": "New format · thinking",
}


def mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def outcome(row: dict | None) -> str:
    if row is None:
        return "Not finished"
    if row["outcome"] == "win":
        return "Win"
    if row["outcome"] == "loss":
        return "Death"
    if row["invalid"]:
        return "Invalid action"
    if row["stopped_reason"] == "timeout":
        return "Timeout"
    return "Other incomplete"


def paired_rows(left: dict[str, dict], right: dict[str, dict],
                exclude_ids: set[str] | None = None) -> list[dict]:
    result = []
    for state_id in sorted((left.keys() & right.keys()) - (exclude_ids or set())):
        a, b = left[state_id], right[state_id]
        if (a["world_seed"], a["hp"], a["policy_seed"]) != (b["world_seed"], b["hp"], b["policy_seed"]):
            raise ValueError("paired fight identity differs")
        both_resolved = a["completed"] and b["completed"]
        result.append({"state_id": state_id, "world_seed": a["world_seed"],
                       "enemies": a["enemies"], "floor": a["floor"],
                       "left_outcome": outcome(a), "right_outcome": outcome(b),
                       "win_difference": int(b["win"]) - int(a["win"]),
                       "both_resolved": both_resolved,
                       "hp_loss_difference": b["hp_loss"] - a["hp_loss"] if both_resolved else None,
                       "seconds_difference": b["seconds"] - a["seconds"]
                           if a["seconds"] is not None and b["seconds"] is not None else None,
                       "left_hp_loss": a["hp_loss"], "right_hp_loss": b["hp_loss"]})
    return result


def infrastructure_annotations(run: Path) -> list[dict]:
    path = run / "infrastructure_annotations.json"
    if not path.exists():
        return []
    saved = lab.read_json(path)
    if saved["schema_version"] != 1:
        raise ValueError("unknown infrastructure annotation version")
    for item in saved["annotations"]:
        sid = item["state_id"]
        result = run / f"{sid}.result.json"
        trace = run / f"{sid}_r0.jsonl"
        if (item["kind"] != "system_sleep_timeout"
                or lab.read_json(result)["status"] != "timeout"
                or file_sha256(result) != item["result_sha256"]
                or file_sha256(trace) != item["trace_sha256"]):
            raise ValueError("infrastructure annotation does not match frozen attempt")
    return saved["annotations"]


def summarize_pairs(rows: list[dict]) -> dict:
    resolved = [r for r in rows if r["both_resolved"]]
    return {"paired_finished": len(rows), "paired_resolved": len(resolved),
            "mean_win_difference": mean(rows, "win_difference"),
            "win_difference_world_bootstrap_95": cluster_interval(rows, "win_difference"),
            "wins_gained": sum(r["win_difference"] == 1 for r in rows),
            "wins_lost": sum(r["win_difference"] == -1 for r in rows),
            "mean_hp_loss_difference": mean(resolved, "hp_loss_difference"),
            "hp_difference_world_bootstrap_95": cluster_interval(resolved, "hp_loss_difference"),
            "higher_exit_hp": sum(r["hp_loss_difference"] < 0 for r in resolved),
            "lower_exit_hp": sum(r["hp_loss_difference"] > 0 for r in resolved),
            "tied_exit_hp": sum(r["hp_loss_difference"] == 0 for r in resolved)}


def compare(runs: dict[str, Path]) -> dict[str, Any]:
    if set(runs) != set(ARM_NAMES):
        raise ValueError("provide all three named arms")
    manifests = {arm: lab.read_json(path / "manifest.json") for arm, path in runs.items()}
    reference = manifests["new_no_thinking"]
    cohort = reference["order"]
    if manifests["new_thinking"]["order"] != cohort:
        raise ValueError("new arms have different scheduled starts/order")
    for manifest in manifests.values():
        if any(manifest[k] != reference[k] for k in ("root", "model_id", "model_revision")):
            raise ValueError("arms have different bank/model identities")
        for state_id in cohort:
            if manifest["sampling_seeds"].get(state_id) != reference["sampling_seeds"][state_id]:
                raise ValueError("arms have different sampling seeds")
    by_arm, summaries, decision_rows, provenance, settings = {}, {}, {}, {}, {}
    starts = None
    for arm in ARM_NAMES:
        _, rows, decisions, source_starts, snapshot = analyze(runs[arm])
        rows = [r for r in rows if r["state_id"] in cohort]
        by_arm[arm] = {r["state_id"]: r for r in rows}
        decision_rows[arm] = [r for r in decisions if r["state_id"] in cohort]
        settings[arm] = snapshot["settings"]
        provenance[arm] = snapshot
        if arm == "new_no_thinking":
            starts = {r["state_id"]: r for r in source_starts}
        resolved = [r for r in rows if r["completed"]]
        summaries[arm] = {"scheduled_comparison": len(cohort), "finished": len(rows),
            "not_finished": len(cohort) - len(rows), "resolved": len(resolved),
            "wins": sum(r["win"] for r in rows), "deaths": sum(r["outcome"] == "loss" for r in rows),
            "invalid": sum(r["invalid"] for r in rows),
            "timeouts": sum(r["stopped_reason"] == "timeout" for r in rows),
            "other_incomplete": sum(outcome(r) == "Other incomplete" for r in rows),
            "win_rate_finished": mean(rows, "win"), "win_rate_resolved": mean(resolved, "win"),
            "mean_net_hp_loss_resolved": mean(resolved, "hp_loss"),
            "mean_seconds": mean(rows, "seconds"), "mean_responses": mean(rows, "responses"),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "thinking_tokens": sum(r["thinking_tokens"] for r in rows),
            "responses": sum(r["responses"] for r in rows), "cap_hits": sum(r["cap_hits"] for r in rows)}
    for arm in ARM_NAMES:
        for key in ("temperature", "top_p", "top_k", "adapter_path", "max_decisions"):
            if settings[arm][key] != settings["new_no_thinking"][key]:
                raise ValueError(f"unplanned setting difference: {key}")
    annotations = {arm: infrastructure_annotations(run) for arm, run in runs.items()}
    excluded = {arm: {r["state_id"] for r in entries} for arm, entries in annotations.items()}
    pairs = {}
    for a, b in combinations(ARM_NAMES, 2):
        omitted = (by_arm[a].keys() & by_arm[b].keys()) & (excluded[a] | excluded[b])
        rows = paired_rows(by_arm[a], by_arm[b], omitted)
        pairs[f"{a}__{b}"] = {"left": a, "right": b, "rows": rows,
                              "excluded_infrastructure_ids": sorted(omitted),
                              "summary": summarize_pairs(rows)}
    all_excluded = set.union(*excluded.values()) & set(cohort)
    common = set.intersection(*(set(v) for v in by_arm.values())) - all_excluded
    common_resolved = {sid for sid in common if all(by_arm[arm][sid]["completed"] for arm in ARM_NAMES)}
    common_summary = {arm: {
        "finished_n": len(common), "wins": sum(by_arm[arm][sid]["win"] for sid in common),
        "win_rate": mean([by_arm[arm][sid] for sid in common], "win"),
        "resolved_n": len(common_resolved),
        "mean_net_hp_loss": mean([by_arm[arm][sid] for sid in common_resolved], "hp_loss"),
    } for arm in ARM_NAMES}
    source_state = {}
    for arm, run in runs.items():
        source_state[arm] = {name: lab.read_json(run / name) for name in
                             ("progress.json", "queue_status.json", "budget_status.json", "budget_history.json")
                             if (run / name).exists()}
        source_state[arm]["infrastructure_recoveries"] = [
            {"path": str(path), "sha256": file_sha256(path), **lab.read_json(path)}
            for path in sorted((run / "infrastructure_interruptions").glob("*/recovery.json"))]
        source_state[arm]["infrastructure_annotations"] = annotations[arm]
        annotation_path = run / "infrastructure_annotations.json"
        if annotation_path.exists():
            source_state[arm]["infrastructure_annotations_sha256"] = file_sha256(annotation_path)
    return {"cohort": cohort, "starts": starts, "summaries": summaries,
            "by_arm": by_arm, "decisions": decision_rows, "pairs": pairs,
            "common_finished_ids": sorted(common), "common_resolved_ids": sorted(common_resolved),
            "common_summary": common_summary, "settings": settings,
            "infrastructure_excluded_ids": sorted(all_excluded),
            "provenance": provenance, "run_status": source_state,
            "analysis_sha256": file_sha256(Path(__file__))}
