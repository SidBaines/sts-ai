"""Read-only snapshot analysis of saved single-fight attempts (no simulator/MLX)."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def cluster_interval(rows: list[dict], key: str, *, repeats: int = 2000) -> list[float] | None:
    """Percentile bootstrap resampling whole observed worlds, keeping their fights."""
    groups = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            groups[row["world_seed"]].append(float(row[key]))
    if len(groups) < 3:
        return None
    totals = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(20260906)
    means = []
    for _ in range(repeats):
        selected = rng.choices(totals, k=len(totals))
        means.append(sum(s for s, n in selected) / sum(n for s, n in selected))
    return [percentile(means, .025), percentile(means, .975)]


def fight_records(records: list[dict], index: int) -> list[dict]:
    result = []
    first = records[index]
    for record in records[index:]:
        if record["phase"] != "combat" or any(record["state"].get(k) != first["state"].get(k) for k in ("floor", "act")):
            break
        result.append(record)
        after = record["after_state"]
        if after.get("phase") != "combat" or lab.hp(after)[0] <= 0:
            break
    return result


def trace_outcome(records: list[dict]) -> tuple[bool, bool | None, int | None]:
    if not records or not records[-1].get("action_executed", True):
        return False, None, None
    after = records[-1]["after_state"]
    hp = lab.hp(after)[0]
    completed = hp <= 0 or after.get("phase") == "out_of_combat"
    return completed, hp > 0 if completed else None, hp if completed else None


def read_trace(path: Path, *, allow_torn_tail: bool) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    lines = path.read_text().splitlines()
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_torn_tail and i == len(lines) - 1:
                return records, 1
            raise
    return records, 0


def snapshot_inputs(run: Path, manifest: dict, previous: dict | None = None) -> list[dict]:
    inputs = []
    expected = previous["inputs"] if previous is not None else [
        {"state_id": state_id} for state_id in manifest["order"]
        if (run / f"{state_id}.result.json").exists()]
    for item in expected:
        state_id = item["state_id"]
        if state_id not in manifest["order"]:
            raise ValueError("unknown state in analysis snapshot")
        path = run / f"{state_id}.result.json"
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if previous is not None and checksum != item["result_sha256"]:
            raise ValueError("saved result changed since snapshot")
        result = json.loads(raw)
        if result["state_id"] != state_id:
            raise ValueError("result identity differs")
        inputs.append({"state_id": state_id, "result_sha256": checksum, "result": result})
    return inputs


def same_start(actual: dict, expected: dict) -> bool:
    """Allow declared v3→v4/v5 presentation changes at a paired start."""
    version = actual["state"].get("combat_observation")
    if version not in ("combat_public_v4", "combat_public_v5"):
        return lab.public_digest(actual) == lab.public_digest(expected)
    if expected["state"].get("combat_observation") != "combat_public_v3":
        return False
    labels = {"keep current hand (discard no cards)": "{ GAMBLE none"}
    if version == "combat_public_v5":
        from sts_ai.combat_public_v5 import selection_label
        from sts_ai.schemas import LegalAction
        hand = expected["state"].get("combat", {}).get("cards", {}).get("hand", [])
        labels["exhaust no cards"] = "{ EXHAUST_MANY none"
        for action in expected["legal_actions"]:
            labels[selection_label(LegalAction(**action), hand)] = action["description"]
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ("combat_public_v3" if k == "combat_observation" else normalize(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return labels.get(value, value) if isinstance(value, str) else value
    # Text changes are the intervention. Full structured public state plus action
    # indices/bits/descriptions must match after that single documented rename.
    return all(normalize(actual[k]) == expected[k] for k in ("phase", "state", "legal_actions"))


def thinking_truncated(record: dict) -> bool:
    from sts_ai.agents import explicit_thinking_truncated
    agent = record.get("agent", {})
    return (bool(agent.get("metadata", {}).get("thinking_truncated", False))
            or explicit_thinking_truncated(agent.get("raw_response", "")))


def generation_cap_hit(record: dict, limit: int) -> bool:
    agent = record["agent"]
    tokens = agent["completion_tokens"]
    truncated = thinking_truncated(record)
    return truncated or tokens >= limit or (not agent["valid"] and tokens >= limit - 8)


def analyze(run: Path, previous: dict | None = None) -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    manifest = lab.read_json(run / "manifest.json")
    if previous is not None and previous["run_manifest_sha256"] != file_sha256(run / "manifest.json"):
        raise ValueError("run manifest changed since analysis snapshot")
    bank = Path(manifest["root"])
    collection = lab.read_json(bank / "collection.json")
    selected = lab.read_json(run / "selection.json")
    lab.validate_selection(collection, selected)
    if selected["selection_sha256"] != manifest["selection_sha256"]:
        raise ValueError("run selection differs")
    if file_sha256(run / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("run settings changed")
    settings = lab.read_json(run / "settings.json")
    token_limit = int(settings.get("max_tokens", 4096))
    starts = selected["states"]
    by_id = {row["state_id"]: row for row in starts}
    inputs = snapshot_inputs(run, manifest, previous)
    source_cache, source_hashes = {}, {}
    rows, decisions = [], []
    for snapshot in inputs:
        saved = snapshot["result"]
        state_id = snapshot["state_id"]
        start = by_id[state_id]
        source = start["source"]
        if source not in source_cache:
            source_cache[source] = lab.source_records(bank, collection, source)
            source_hashes[source] = file_sha256(bank / source)
        teacher = fight_records(source_cache[source], start["index"])
        teacher_complete, teacher_won, teacher_exit = trace_outcome(teacher)
        if not teacher_complete:
            raise ValueError(f"source fight is incomplete: {state_id}")
        trace = run / f"{state_id}_r0.jsonl"
        if saved.get("trace", trace.name) != trace.name:
            raise ValueError("unexpected trajectory filename")
        records, torn = read_trace(trace, allow_torn_tail=saved["status"] != "ok")
        snapshot["trace_sha256"] = file_sha256(trace) if trace.exists() else None
        if previous is not None:
            old = next(i for i in previous["inputs"] if i["state_id"] == state_id)
            if old["trace_sha256"] != snapshot["trace_sha256"]:
                raise ValueError("trajectory changed since analysis snapshot")
        if records and not same_start(records[0], teacher[0]):
            raise ValueError("model and heuristic starts differ")
        expected_observation = settings.get("combat_observation", teacher[0]["state"].get("combat_observation"))
        if any(r["state"].get("combat_observation") != expected_observation for r in records):
            raise ValueError("record observation version differs from arm")
        if any(r["policy_seed"] != manifest["sampling_seeds"][state_id] for r in records):
            raise ValueError("unexpected sampling seed")
        if saved["status"] == "ok":
            meta_path = trace.with_suffix(".meta.json")
            if file_sha256(trace) != saved["trace_sha256"] or file_sha256(meta_path) != saved["meta_sha256"]:
                raise ValueError("finished trace or metadata changed")
            meta = lab.read_json(meta_path)
            if meta["extra"]["fight_lab"]["state_id"] != state_id:
                raise ValueError("metadata identity differs")
            if int(meta["max_tokens"]) != token_limit or meta["thinking"] != settings["thinking"]:
                raise ValueError("episode generation settings differ from arm")
            for key in ("temperature", "top_p", "top_k", "thinking", "max_tokens"):
                if meta["extra"]["agent_config"][key] != settings[key]:
                    raise ValueError(f"episode agent configuration differs: {key}")
            complete, won, exit_hp = trace_outcome(records)
            if complete != saved["metrics"]["completed"] or (complete and won != saved["metrics"]["won"]):
                raise ValueError("reported outcome differs from trace")
        else:
            complete, won, exit_hp = False, None, None
        compromised = bool(saved.get("undefined_behavior_evoked"))
        complete = complete and not compromised
        outcome = "win" if complete and won else "loss" if complete else "incomplete"
        hp_loss = start["hp"] - exit_hp if complete else None
        teacher_loss = start["hp"] - teacher_exit
        row = {**{k: start[k] for k in ("state_id", "world_seed", "floor", "act", "hp", "max_hp", "hp_fraction", "split")},
            "enemies": sorted(set(start["enemies"])), "encounter": " + ".join(start["enemies"]),
            "outcome": outcome, "completed": complete, "win": outcome == "win",
            "invalid": saved.get("n_invalid", 0) > 0, "stopped_reason": saved.get("stopped_reason", saved["status"]),
            "undefined_behavior": compromised, "exit_hp": exit_hp if complete else None, "hp_loss": hp_loss,
            "teacher_win": teacher_won, "teacher_exit_hp": teacher_exit, "teacher_hp_loss": teacher_loss,
            "hp_loss_gap": hp_loss - teacher_loss if complete else None,
            "model_decisions": sum(r.get("action_executed", True) for r in records),
            "teacher_decisions": len(teacher), "responses": len(records),
            "turns": max(((r["state"].get("combat") or {}).get("turn", 0) for r in records), default=0),
            "teacher_turns": max((r["state"].get("combat") or {}).get("turn", 0) for r in teacher),
            "seconds": saved.get("seconds"), "model_seconds": sum(r["agent"]["latency_s"] for r in records),
            "completion_tokens": sum(r["agent"]["completion_tokens"] for r in records),
            "prompt_tokens": sum(r["agent"]["prompt_tokens"] for r in records),
            "thinking_tokens": sum(r["agent"].get("thinking_tokens", 0) for r in records),
            "max_tokens": token_limit,
            "cap_hits": sum(generation_cap_hit(r, token_limit) for r in records),
            "torn_tail_lines_ignored": torn, "policy_seed": manifest["sampling_seeds"][state_id],
            "source": source, "source_index": start["index"]}
        rows.append(row)
        for i, record in enumerate(records):
            agent = record["agent"]
            action = record.get("selected_action") or {}
            description = action.get("description", "")
            kind = next((name for name in ("Attack", "Skill", "Power", "Status", "Curse") if f"[{name}]" in description),
                        "End turn" if "end turn" in description.lower() else "Potion" if "potion" in description.lower() else "Other")
            decisions.append({"state_id": state_id, "world_seed": start["world_seed"], "encounter": row["encounter"],
                "decision_index": i, "turn": (record["state"].get("combat") or {}).get("turn"),
                "hp": lab.hp(record["state"])[0], "valid": agent["valid"],
                "action_executed": record.get("action_executed", True), "action_kind": kind,
                "action": description, "completion_tokens": agent["completion_tokens"],
                "prompt_tokens": agent["prompt_tokens"], "latency_s": agent["latency_s"],
                "thinking_tokens": agent.get("thinking_tokens", 0), "max_tokens": token_limit,
                "raw_response": agent["raw_response"],
                "reported_thinking_truncated": bool(agent.get("metadata", {}).get("thinking_truncated", False)),
                "thinking_truncated": thinking_truncated(record),
                "cap_hit": generation_cap_hit(record, token_limit)})
    paired = [r for r in rows if r["completed"]]
    counts = Counter(r["outcome"] for r in rows)
    avg = lambda key, data: statistics.mean(r[key] for r in data if r.get(key) is not None) if any(r.get(key) is not None for r in data) else None
    summary = {"scheduled": len(starts), "finished_attempts": len(rows), "worlds_observed": len({r['world_seed'] for r in rows}),
        "completed_fights": len(paired), "wins": counts['win'], "losses": counts['loss'], "incomplete": counts['incomplete'],
        "win_fraction_all_finished": avg("win", rows), "win_fraction_completed": avg("win", paired),
        "win_fraction_world_bootstrap_95": cluster_interval(rows, "win"),
        "invalid_attempts": sum(r['invalid'] for r in rows), "invalid_responses": sum(not d['valid'] for d in decisions),
        "responses": len(decisions), "cap_hits": sum(d['cap_hit'] for d in decisions),
        "mean_hp_loss_model_completed": avg("hp_loss", paired), "mean_hp_loss_teacher_matched": avg("teacher_hp_loss", paired),
        "mean_hp_loss_gap": avg("hp_loss_gap", paired), "median_hp_loss_gap": percentile([r['hp_loss_gap'] for r in paired], .5),
        "hp_loss_gap_world_bootstrap_95": cluster_interval(paired, "hp_loss_gap"),
        "teacher_wins_on_matched_completed": sum(r['teacher_win'] for r in paired),
        "model_higher_exit_hp": sum(r['hp_loss_gap'] < 0 for r in paired),
        "tied_exit_hp": sum(r['hp_loss_gap'] == 0 for r in paired),
        "model_lower_exit_hp": sum(r['hp_loss_gap'] > 0 for r in paired),
        "mean_seconds_per_attempt": avg("seconds", rows), "median_seconds_per_attempt": percentile([r['seconds'] for r in rows if r['seconds'] is not None], .5),
        "total_completion_tokens": sum(r['completion_tokens'] for r in rows),
        "total_prompt_tokens": sum(r['prompt_tokens'] for r in rows),
        "stopped_reasons": dict(Counter(r['stopped_reason'] for r in rows))}
    provenance = {"run": str(run), "run_manifest_sha256": file_sha256(run / 'manifest.json'),
        "selection_sha256": selected['selection_sha256'], "inputs": inputs, "source_hashes": source_hashes,
        "analysis_sha256": file_sha256(Path(__file__)), "bootstrap_seed": 20260906,
        "settings": lab.read_json(run/'settings.json')}
    return summary, rows, decisions, starts, provenance
