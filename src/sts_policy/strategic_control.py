"""Bounded pilot resource and paired retention gates, separate from model selection."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import hashlib
from typing import Any

CONTROL_VERSION = "strategic_pilot_control_v3_v1"


def unchanged_sources(identity: dict[str, str]) -> bool:
    for filename, expected in identity.items():
        try:
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        except FileNotFoundError:
            return False
        if actual != expected:
            return False
    return True


def power_record(text: str) -> dict[str, Any]:
    percent = re.search(r"(\d+)%;", text)
    return {"battery_percent": int(percent.group(1)) if percent else None,
            "ac_present": "AC Power" in text, "discharging": "discharging" in text,
            "charging": "charging" in text and "discharging" not in text and "not charging" not in text,
            "raw": text.strip()[:500]}


def resources(path: Path) -> dict[str, Any]:
    try:
        power = power_record(subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                            text=True, timeout=5, check=True).stdout)
    except (FileNotFoundError, subprocess.SubprocessError):
        power = {"battery_percent": None, "ac_present": None, "discharging": None, "charging": None}
    return {**power, "disk_free_bytes": shutil.disk_usage(path).free,
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def resource_gate(record: dict[str, Any], *, minimum_battery: int = 25,
                  minimum_disk_bytes: int = 8 * 1024**3) -> str | None:
    if record["disk_free_bytes"] < minimum_disk_bytes:
        return "storage_reserve"
    percent = record.get("battery_percent")
    if percent is not None and percent <= minimum_battery and not record.get("charging", False):
        return "battery_reserve"
    return None


def paired_retention(reference: dict, candidate: dict) -> dict:
    old = {e["task_id"]: e for e in reference["episodes"] if e["terminated"] and not e["truncated"]}
    new = {e["task_id"]: e for e in candidate["episodes"] if e["terminated"] and not e["truncated"]}
    common = sorted(set(old) & set(new))
    result = {"common_valid": len(common), "reference_valid": len(old), "candidate_valid": len(new)}
    for name, predicate in (("victory", lambda e: e["task_outcome"] == "victory"),
                            ("act1", lambda e: e["max_act_reached"] >= 2),
                            ("act2", lambda e: e["max_act_reached"] >= 3)):
        gains = sum(not predicate(old[k]) and predicate(new[k]) for k in common)
        losses = sum(predicate(old[k]) and not predicate(new[k]) for k in common)
        result[name] = {"gains": gains, "losses": losses, "net": gains-losses,
                        "reference": sum(predicate(old[k]) for k in common),
                        "candidate": sum(predicate(new[k]) for k in common)}
    deltas = [float(new[k]["floor"])-float(old[k]["floor"]) for k in common]
    result["mean_floor_delta"] = sum(deltas)/len(deltas) if deltas else None
    result["changed_traces"] = sum(old[k]["action_trace_sha256"] != new[k]["action_trace_sha256"] for k in common)
    return result


def retention_gate(paired: dict, expected: int) -> str | None:
    if paired["common_valid"] < expected:
        return "evaluation_incomplete"
    if paired["act1"]["net"] <= -8:
        return "act1_retention_loss"
    if paired["mean_floor_delta"] is not None and paired["mean_floor_delta"] <= -1.5 and paired["victory"]["net"] <= 0:
        return "floor_retention_loss_without_win_gain"
    return None


def drift_gate(metrics: dict) -> str | None:
    drift = metrics["rehearsal_drift"]
    if (drift["original_kl"]["mean"] or 0) >= .03:
        return "original_mean_kl"
    if (drift["original_kl"]["p95"] or 0) >= .2:
        return "original_tail_kl"
    if (drift["original_tv"]["mean"] or 0) >= .08:
        return "original_tv"
    if drift["changed_greedy_fraction"] >= .15:
        return "original_greedy_drift"
    return metrics.get("stop_reason")
