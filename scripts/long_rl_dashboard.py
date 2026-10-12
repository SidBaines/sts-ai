#!/usr/bin/env python3
"""Read-only, dependency-free local telemetry for the long curriculum study."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "long_rl_dashboard"
MAX_JSON_BYTES = 64 * 1024 * 1024
METRIC_FIELDS = ("policy_loss", "value_loss", "approx_kl", "entropy", "clip_fraction", "gradient_norm", "ratio_mean", "rejected_kl")


def number(value):
    """JSON-safe finite numeric telemetry; unavailable never becomes zero."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def mean(values):
    valid = [v for v in values if number(v) is not None]
    return statistics.fmean(valid) if valid else None


def quantile(values, q):
    valid = sorted(v for v in values if number(v) is not None)
    if not valid:
        return None
    index = (len(valid) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return valid[lo] + (valid[hi] - valid[lo]) * (index - lo)


@dataclass
class CachedJSON:
    """Mtime/size cache retains only bounded curated documents."""
    limit: int = 1024

    def __post_init__(self):
        self.entries = OrderedDict()
        self.reads = 0

    def read(self, path: Path):
        stat = path.stat()
        if stat.st_size > MAX_JSON_BYTES:
            raise ValueError("telemetry document exceeds size bound")
        stamp = (stat.st_mtime_ns, stat.st_size)
        old = self.entries.get(path)
        if old and old[0] == stamp:
            self.entries.move_to_end(path)
            return old[1]
        data = json.loads(path.read_text())
        self.reads += 1
        self.entries[path] = (stamp, data)
        self.entries.move_to_end(path)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)
        return data


def curated_metric_path(campaign: Path, value):
    """Only campaign-owned named metric JSON; no checkpoint bytes/final cohort."""
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = campaign / path
    resolved = path.resolve()
    if resolved.parent not in {campaign.resolve() / "metrics", campaign.resolve() / "checkpoints"}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.metrics\.json", resolved.name):
        return None
    if "final" in resolved.name.lower():
        return None
    return resolved


def quality_summary(q):
    if not isinstance(q, dict):
        return None
    result = {k: q.get(k) for k in ("clears", "requested", "floor_mean", "faults", "truncated", "complete")}
    n, clears = number(q.get("requested")), number(q.get("clears"))
    result["clear_rate"] = clears / n if n and clears is not None else None
    result["task_count"] = len(q.get("cohort", []))
    result["metrics_sha256"] = q.get("metrics_sha256")
    return result


def normalize_training(metrics, job, index):
    """Reduce logged PPO telemetry; do not substitute optimizer configuration."""
    rounds = metrics.get("rounds", [])
    episodes = [e for r in rounds for e in r.get("episodes", [])]
    updates = [r.get("update", {}) for r in rounds]
    rewards = [e.get("reward") for e in episodes if not e.get("fault")]
    transitions = sum(r.get("transitions", 0) for r in rounds)
    rollout_seconds = sum(r.get("elapsed_seconds", 0) for r in rounds)
    update_seconds = sum(u.get("elapsed_seconds", 0) for u in updates)
    elapsed = number(metrics.get("elapsed_seconds"))
    row = {"job": index, "stage": job.get("optimizer_goal", {}).get("target", {}).get("stage"),
           "session": job.get("session_kind"), "started": job.get("started"), "finished": job.get("finished"),
           "status": job.get("status"), "transitions": transitions,
           "steps": sum(u.get("steps", 0) for u in updates), "episodes": len(episodes),
           "faults": sum(len(r.get("faults", [])) for r in rounds),
           "truncated": sum(bool(e.get("truncated")) for e in episodes),
           "unstarted": sum(r.get("unstarted_tasks", 0) for r in rounds),
           "excluded_transitions": sum(r.get("excluded_transitions", 0) for r in rounds),
           "elapsed_seconds": elapsed, "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
           "transitions_per_second": transitions / elapsed if elapsed else None,
           "rollout_transitions_per_second": transitions / rollout_seconds if rollout_seconds else None,
           "return_mean": mean(rewards), "return_q10": quantile(rewards, .1), "return_q50": quantile(rewards, .5),
           "return_q90": quantile(rewards, .9), "floor_mean": mean([e.get("floor") for e in episodes]),
           "rejected_minibatches": sum(u.get("rejected_minibatches", 0) for u in updates),
           "early_stop_rounds": sum(u.get("early_stop") is True for u in updates),
           "update_preparation_seconds": sum(u.get("preparation_seconds", 0) for u in updates),
           "actor_changed": None, "metrics_file": Path(job.get("metrics", "")).name}
    requested = Counter(task_id for r in rounds for task_id in r.get("task_ids", []))
    observed = Counter(e.get("task_id") for e in episodes)
    observed.update(f.get("task_id") for r in rounds for f in r.get("faults", []) if isinstance(f, dict))
    row["coverage"] = {"requested_ids": sum(requested.values()), "observed_ids": sum(observed.values()),
                       "missing_ids": sum((requested - observed).values()), "unexpected_ids": sum((observed - requested).values()),
                       "exact_match": requested == observed if requested else None,
                       "cohort_completed": all(r.get("cohort_completed") is True for r in rounds) if rounds else None,
                       "collector_budget_episodes": sum(e.get("termination_reason") == "collector_budget" for e in episodes)}
    for field in METRIC_FIELDS:
        row[field] = mean([u.get(field) for u in updates])
    evidence = job.get("actor_evidence", {})
    before, after = evidence.get("before", {}).get("actor_sha256"), evidence.get("after", {}).get("actor_sha256")
    if before and after:
        row["actor_changed"] = before != after
    row["timings"] = {k: sum(r.get("timings", {}).get(k, 0) for r in rounds) for k in
                      ("policy_seconds", "reset_wait_seconds", "step_wait_seconds", "bootstrap_seconds", "fault_wait_seconds")}
    return row


FULL_RUN_ROLES = ("selected_best", "latest_candidate")


def normalize_full_run(telemetry, checkpoints, reference_task_ids=()):
    """Full Act-1-start panels stay separate from curriculum/continuation scores.

    Exact SHA and role identify an evaluated snapshot, never today's live weights.
    Invalid or unmatched cohorts remain visible with a validity warning.
    """
    result = {"monitoring_only": True, "status": "not_yet_evaluated", "history": [],
              "latest": {role: None for role in FULL_RUN_ROLES}, "errors": []}
    if telemetry is None:
        return result
    if not isinstance(telemetry, dict) or telemetry.get("monitoring_only") is not True:
        result["errors"].append("Observer telemetry must be explicitly monitoring_only")
        return result
    if telemetry.get("version", 1) != 1:
        result["errors"].append("Unsupported observer telemetry version")
        return result
    panel = telemetry.get("panel", "dev_act3")
    panel_name = panel.get("name", panel.get("id")) if isinstance(panel, dict) else panel
    if panel_name not in ("dev_act3", None):
        result["errors"].append("Observer panel is not fresh full-start dev_act3")
        return result
    history = telemetry.get("histories", telemetry.get("history", []))
    if not isinstance(history, list) or len(history) > 5000:
        result["errors"].append("Observer history is invalid or exceeds 5000 panels")
        return result
    reference = Counter(reference_task_ids)
    seen = set()
    for record in history:
        if not isinstance(record, dict):
            result["errors"].append("Ignored malformed observer record")
            continue
        sha = record.get("checkpoint_sha256")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            result["errors"].append("Ignored observer record without exact checkpoint SHA256")
            continue
        roles = record.get("roles", [record.get("role")])
        if isinstance(roles, str):
            roles = [roles]
        roles = [role for role in roles if role in FULL_RUN_ROLES]
        if not roles:
            result["errors"].append("Ignored observer record without recognized snapshot role")
            continue
        evaluated_at = number(record.get("evaluated_at"))
        identity = (sha, evaluated_at, tuple(sorted(set(roles))))
        if identity in seen:
            continue
        seen.add(identity)
        row = {"checkpoint_sha256": sha, "roles": sorted(set(roles)),
               "checkpoint_name": record.get("checkpoint_name"),
               "capture_at": number(record.get("capture_at", record.get("captured_at"))),
               "evaluated_at": evaluated_at,
               "captured_global_round": number(record.get("captured_global_round", record.get("global_round"))),
               "requested": number(record.get("requested")), "clears": number(record.get("clears", record.get("wins"))),
               "defeats": number(record.get("defeats")), "faults": number(record.get("faults")),
               "truncated": number(record.get("truncated")), "unstarted": number(record.get("unstarted")),
               "floor_mean": number(record.get("floor_mean")), "reported_complete": record.get("complete") is True}
        ids = record.get("task_ids", [])
        ids_valid = isinstance(ids, list) and all(isinstance(value, str) for value in ids)
        row["task_count"] = len(ids) if ids_valid else None
        row["matched_panel"] = Counter(ids) == reference if ids_valid and reference else None
        row["exact_cohort"] = ids_valid and len(ids) == 48 and len(set(ids)) == 48 and row["matched_panel"] is True
        outcomes = [row[field] for field in ("clears", "defeats", "faults", "truncated", "unstarted")]
        row["outcome_coverage"] = all(v is not None and v >= 0 for v in outcomes) and sum(outcomes) == row["requested"]
        row["complete"] = (row["reported_complete"] and row["requested"] == 48 and row["exact_cohort"]
                           and row["outcome_coverage"] and row["unstarted"] == 0)
        row["clear_rate"] = row["clears"] / row["requested"] if row["requested"] and row["clears"] is not None else None
        row["matches_live_role"] = {role: sha == checkpoints.get("best_checkpoint" if role == "selected_best" else "last_checkpoint", {}).get("sha256") for role in row["roles"]}
        result["history"].append(row)
    result["history"].sort(key=lambda row: row["evaluated_at"] or 0)
    for row in result["history"]:
        for role in row["roles"]:
            result["latest"][role] = row
    # Role aliases may deliberately reuse an older exact-SHA observation after a
    # rollback. The alias, rather than newest eval timestamp, identifies that role.
    for role, alias in (telemetry.get("latest") or {}).items():
        if role not in FULL_RUN_ROLES or not isinstance(alias, dict):
            continue
        row = next((row for row in reversed(result["history"])
                    if row["checkpoint_sha256"] == alias.get("checkpoint_sha256")
                    and row["evaluated_at"] == alias.get("evaluated_at")), None)
        if row is None:
            result["errors"].append(f"Observer {role} alias has no matching measured panel")
            result["latest"][role] = None
            continue
        result["latest"][role] = dict(row,
            role_capture_at=number(alias.get("role_capture_at")),
            role_captured_global_round=number(alias.get("role_captured_global_round")),
            reused_evaluation=alias.get("reused_evaluation") is True)
        result["latest"][role]["matches_live_role"] = {role: row["checkpoint_sha256"] == checkpoints.get("best_checkpoint" if role == "selected_best" else "last_checkpoint", {}).get("sha256")}
    result["status"] = telemetry.get("status", "evaluated" if result["history"] else "not_yet_evaluated")
    result["updated_at"] = number(telemetry.get("updated_at", telemetry.get("heartbeat")))
    result["next_due"] = number(telemetry.get("next_due"))
    result["interval_seconds"] = number(telemetry.get("interval_seconds"))
    result["pending"] = telemetry.get("pending") if isinstance(telemetry.get("pending"), dict) else None
    return result


class TelemetryStore:
    """Journal refresh is rate-limited; each changed metric is reduced once."""
    def __init__(self, campaign: Path, refresh_seconds: float = 8):
        self.campaign = campaign.resolve()
        self.refresh_seconds = refresh_seconds
        self.cache = CachedJSON(limit=16)
        self.metrics_cache = OrderedDict()
        self.last_refresh = -math.inf
        self.snapshot = None
        self.lock = threading.Lock()
        self.errors = []

    def document(self, name, default):
        path = self.campaign / name
        if path.resolve().parent != self.campaign:
            self.errors.append("Rejected document outside campaign")
            return default
        try:
            return self.cache.read(path)
        except (OSError, ValueError) as exc:
            self.errors.append(f"{name}: {type(exc).__name__}")
            return default

    def observer_telemetry(self):
        """Read exactly the owned sibling observer telemetry, never its models."""
        parent = self.campaign.parent
        path = parent / "full_run_eval" / "telemetry.json"
        if path.resolve().parent != parent / "full_run_eval":
            self.errors.append("Rejected observer telemetry symlink outside owned directory")
            return None
        try:
            return self.cache.read(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            self.errors.append(f"full_run_eval/telemetry.json: {type(exc).__name__}")
            return None

    def metric(self, job, index):
        path = curated_metric_path(self.campaign, job.get("metrics"))
        if path is None:
            return None
        try:
            stat = path.stat()
            journal_stamp = json.dumps({k: job.get(k) for k in
                ("actor_evidence", "status", "finished", "started", "optimizer_goal", "session_kind")}, sort_keys=True)
            stamp = (stat.st_mtime_ns, stat.st_size, journal_stamp)
            cached = self.metrics_cache.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
            if stat.st_size > MAX_JSON_BYTES:
                raise ValueError("metric exceeds bound")
            metrics = json.loads(path.read_text())
            row = normalize_training(metrics, job, index)
            self.metrics_cache[path] = (stamp, row)
            while len(self.metrics_cache) > 5000:
                self.metrics_cache.popitem(last=False)
            return row
        except FileNotFoundError:
            return None  # New jobs have no metrics until their first atomic write.
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path.name}: {type(exc).__name__}")
            return None

    def refresh(self):
        self.errors = []
        j = self.document("journal.json", {})
        progress = self.document("progress.json", {})
        config_record = self.document("config.json", {})
        config = config_record.get("config", {})
        rows, families = [], {}
        jobs = j.get("jobs", [])
        for index, job in enumerate(jobs):
            if job.get("kind") != "train":
                continue
            row = self.metric(job, index)
            if row is not None:
                rows.append(row)
            for key, counts in job.get("family_strata", {}).items():
                try:
                    meta = json.loads(key)
                except (ValueError, TypeError):
                    meta = {"start_source": "unknown", "family": key}
                entry = families.setdefault(key, {"metadata": meta, **{k: 0 for k in
                    ("requested", "episodes", "clears", "defeats", "faults", "truncated", "unstarted")}})
                for field in ("requested", "episodes", "clears", "defeats", "faults", "truncated", "unstarted"):
                    entry[field] += counts.get(field, 0)
        for family in families.values():
            family["clear_rate"] = family["clears"] / family["requested"] if family["requested"] else None
            family["conditional"] = family["metadata"].get("start_source") != "fresh_scratch"
        decisions = []
        for event in j.get("events", []):
            if not isinstance(event, dict):
                continue
            decisions.append({"at": event.get("at"), "round": event.get("round"), "stage": event.get("stage"),
                "decision": event.get("decision"), "quality": quality_summary(event.get("quality")),
                "retention": quality_summary(event.get("retention"))})
        selected_dev = None
        for event, decision in zip([e for e in j.get("events", []) if isinstance(e, dict)], decisions):
            metric = (event.get("quality") or {}).get("metrics")
            eval_job = next((job for job in jobs if job.get("kind") == "eval" and job.get("metrics") == metric), {})
            command = eval_job.get("command", [])
            checkpoint = command[command.index("--checkpoint") + 1] if "--checkpoint" in command else None
            goal = command[command.index("--max-act") + 1] if "--max-act" in command else None
            decision["goal_act"] = int(goal) if goal and str(goal).isdigit() else None
            if checkpoint == j.get("best_checkpoint") and event.get("decision", {}).get("action") == "promote":
                selected_dev = decision
        leaders = {key: {"checkpoint": Path(v.get("checkpoint", "")).name,
                        "quality": quality_summary(v.get("quality"))} for key, v in j.get("goal_leaders", {}).items()}
        cps = {}
        for key in ("initial_checkpoint", "current_checkpoint", "best_checkpoint", "last_checkpoint"):
            value = j.get(key, progress.get(key))
            name = Path(value).name if value else None
            artifact_sha = next((job.get("artifacts", {}).get(value) for job in reversed(jobs)
                                 if value in job.get("artifacts", {})), None)
            cps[key] = {"name": name, "sha256": artifact_sha or (config_record.get("initial_checkpoint_sha256") if key == "initial_checkpoint" else None)}
        last_cp = j.get("last_checkpoint")
        cp_job = next((job for job in reversed(jobs) if job.get("checkpoint") == last_cp and job.get("finished")), {})
        active_index = j.get("active_job")
        active_job = jobs[active_index] if isinstance(active_index, int) and 0 <= active_index < len(jobs) else {}
        self.snapshot = {"schema": 1, "sampled_at": time.time(), "status": progress.get("status", j.get("status", "unavailable")),
            "pause_reason": progress.get("pause_reason"), "started": j.get("started"), "heartbeat": progress.get("heartbeat", j.get("heartbeat")),
            "allocation": progress.get("allocation", j.get("allocation", {})), "stage": (j.get("current_goal") or {}).get("stage"),
            "global_round": progress.get("global_round", j.get("global_round")), "active_job": progress.get("active_job"),
            "active_job_kind": active_job.get("kind"), "active_job_status": active_job.get("status"),
            "controller_pid": j.get("controller", {}).get("pid"), "last_checkpoint_at": cp_job.get("finished"),
            "checkpoints": cps, "training": rows, "families": list(families.values()), "decisions": decisions[-300:],
            "dev_history": decisions, "selected_dev": selected_dev, "leaders": leaders, "stages": {k: {"round": v.get("round"), "transitions": v.get("transitions"), "updates": v.get("updates")} for k, v in j.get("stages", {}).items()},
            "totals": {field: sum(r.get(field, 0) for r in rows) for field in ("transitions", "steps", "episodes", "faults", "truncated", "unstarted", "excluded_transitions", "rejected_minibatches")},
            "config": {k: config.get(k) for k in ("device", "workers", "threads", "duty_cycle", "batch_size", "learning_rate", "clip_ratio", "entropy_coef", "gamma", "gae_lambda", "progress_floor_weight", "terminal_hp_weight", "chunk_seconds", "dev_episodes", "dev_interval", "retention_clear_drop")},
            "baseline": {"dev_act1": {"clears": 41, "requested": 48, "floor_mean": 16.75}, "dev_act2": {"clears": 8, "requested": 48, "floor_mean": 27.416666666666668}, "dev_act3": {"clears": 0, "requested": 48, "floor_mean": 28.604166666666668}},
            "committed_totals": {"transitions": sum(job.get("actor_evidence", {}).get("transitions", 0) for job in jobs if job.get("kind") == "train" and job.get("status") == "complete"),
                                 "steps": sum(job.get("actor_evidence", {}).get("updates", 0) for job in jobs if job.get("kind") == "train" and job.get("status") == "complete")},
            "pruning": {"entries": len(j.get("deletions", [])), "deleted": sum(e.get("status") == "deleted" for e in j.get("deletions", [])), "deleted_bytes": sum(e.get("bytes", 0) for e in j.get("deletions", []) if e.get("status") == "deleted")},
            "errors": self.errors.copy(), "history_jobs": len(rows), "cache_refresh_seconds": self.refresh_seconds}
        learner_job = next((job for job in reversed(jobs) if job.get("kind") == "train"), {})
        learner_command = learner_job.get("command", [])
        self.snapshot["config"]["inference_device"] = learner_command[learner_command.index("--inference-device") + 1] if "--inference-device" in learner_command else None
        reference_ids = (j.get("goal_leaders", {}).get("dev_act3", {}).get("quality") or {}).get("cohort", [])
        observer = self.observer_telemetry()
        full_run = normalize_full_run(observer, cps, reference_ids)
        full_run["initial"] = {"checkpoint_sha256": config_record.get("initial_checkpoint_sha256"),
                               "checkpoint_name": cps["initial_checkpoint"]["name"],
                               "clears": 0, "requested": 48, "defeats": 48, "faults": 0, "truncated": 0,
                               "unstarted": 0, "floor_mean": 28.604166666666668, "complete": True,
                               "reference_date": "2026-10-12", "reference_label": "Original pre-update DEV reference"}
        baseline = observer.get("baseline", {}) if isinstance(observer, dict) else {}
        if baseline.get("checkpoint_sha256") == config_record.get("initial_checkpoint_sha256"):
            full_run["initial"]["saved_metrics_at"] = number(baseline.get("captured_at"))
            full_run["initial"]["timestamp_kind"] = baseline.get("timestamp_kind")
        self.snapshot["full_run"] = full_run
        self.snapshot["errors"] = self.errors.copy()
        self.last_refresh = time.monotonic()

    def get(self):
        with self.lock:
            if self.snapshot is None or time.monotonic() - self.last_refresh >= self.refresh_seconds:
                self.refresh()
            result = dict(self.snapshot)
            now = time.time()
            result["served_at"] = now
            for key, stamp in (("heartbeat_age_seconds", result.get("heartbeat")), ("wall_elapsed_seconds", result.get("started")), ("checkpoint_age_seconds", result.get("last_checkpoint_at"))):
                result[key] = max(0, now - stamp) if number(stamp) is not None else None
            # Allocation is the authoritative recorded ledger, never extrapolated through pauses.
            allocation = result["allocation"]
            result["active_used_seconds"] = number(allocation.get("used_seconds"))
            authorized = number(allocation.get("authorized_seconds"))
            used = result["active_used_seconds"]
            result["active_remaining_seconds"] = max(0, authorized - used) if authorized is not None and used is not None else None
            # Ages are computed per response without mutating the cached observation.
            full_run = dict(result["full_run"])
            full_run["latest"] = {role: (dict(row, evaluation_age_seconds=max(0, now-row["evaluated_at"]) if row.get("evaluated_at") is not None else None) if row else None)
                                  for role, row in full_run["latest"].items()}
            full_run["observer_heartbeat_age_seconds"] = max(0, now-full_run["updated_at"]) if number(full_run.get("updated_at")) is not None else None
            result["full_run"] = full_run
            return result


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/telemetry":
                body = json.dumps(store.get(), allow_nan=False, separators=(",", ":")).encode()
                content_type = "application/json"
            elif path in ("/", "/index.html", "/app.js", "/style.css"):
                asset = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                body = (ASSETS / asset).read_bytes()
                content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[asset]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "") != "200":
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "data/long_rl_v1/campaign")
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--refresh-seconds", type=float, default=8)
    args = parser.parse_args()
    store = TelemetryStore(args.campaign, max(5, args.refresh_seconds))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
    print(f"Long RL dashboard: http://127.0.0.1:{server.server_port} (read-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
