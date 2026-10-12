#!/usr/bin/env python3
"""Read-only, dependency-free local telemetry for the long curriculum study."""
from __future__ import annotations

import argparse
import hashlib
import gzip
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
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "long_rl_dashboard"
MAX_JSON_BYTES = 64 * 1024 * 1024
METRIC_FIELDS = ("policy_loss", "value_loss", "approx_kl", "entropy", "clip_fraction", "gradient_norm", "ratio_mean", "rejected_kl")


def number(value):
    """JSON-safe finite numeric telemetry; unavailable never becomes zero."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def accepts_gzip(value):
    for item in (value or '').lower().split(','):
        parts = [part.strip() for part in item.split(';')]
        if parts[0] != 'gzip': continue
        quality = 1.
        for part in parts[1:]:
            if part.startswith('q='):
                try: quality = float(part[2:])
                except ValueError: quality = 0.
        if math.isfinite(quality) and quality > 0: return True
    return False


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


# Episode exploration is lazy and separate from the compact /api/telemetry route.
MAX_EPISODE_ROWS_PER_FILE = 10000
MAX_EPISODE_CACHE_ROWS = 100000
EPISODE_NUMERIC_FIELDS = ('reward', 'floor', 'hp_cost', 'steps', 'native_steps')
EPISODE_METADATA_FIELDS = ('component', 'start_source', 'start_act', 'start_floor', 'goal_act', 'horizon')


def command_option(command, option):
    values = [command[i + 1] for i, value in enumerate(command[:-1]) if value == option]
    return values[-1] if values else None


def input_identity(job, option):
    signature = job.get('signature', {})
    condition = signature.get('condition', signature)
    return condition.get('inputs', {}).get(option, {})


def content_sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(part)
    return checksum.hexdigest()


def bounded_owned_path(parent, value, pattern):
    if not isinstance(value, str): return None
    path = Path(value)
    if not path.is_absolute(): path = parent / path
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not re.fullmatch(pattern, resolved.name): return None
    if 'final' in resolved.name.lower(): return None
    return resolved


def compact_episodes(metrics, job, index, metadata, *, round_index=None, kind=None, roles=()):
    """Join by exact TaskID. Fault and missing-request rows keep numeric nulls."""
    rounds = metrics.get('rounds', [])
    if sum(len(r.get('episodes', [])) + len(r.get('faults', [])) + len(r.get('task_ids', [])) for r in rounds) > MAX_EPISODE_ROWS_PER_FILE:
        raise ValueError('episode document exceeds row bound')
    condition = job.get('signature', {}).get('condition', job.get('signature', {}))
    cfg = metrics.get('config', {})
    base = {'job': index, 'job_id': Path(job.get('metrics', '')).name,
            'round': round_index, 'round_kind': 'observer_capture' if kind == 'full3' else 'training_or_dev',
            'kind': kind or job.get('kind'), 'session': job.get('session_kind'),
            'stage': job.get('optimizer_goal', {}).get('target', {}).get('stage') or job.get('stage'),
            'job_status': job.get('status'), 'condition_id': condition.get('digest'),
            'started': number(job.get('started')), 'finished': number(job.get('finished')),
            'roles': list(roles), 'checkpoint_sha256': job.get('checkpoint_sha256') or job.get('artifacts', {}).get(job.get('checkpoint')),
            'panel_valid': job.get('panel_valid', True),
            'reward_mode': cfg.get('reward_mode'), 'progress_floor_weight': number(cfg.get('progress_floor_weight')),
            'terminal_hp_weight': number(cfg.get('terminal_hp_weight')), 'gamma': number(cfg.get('gamma')),
            'learning_rate': number(cfg.get('learning_rate')), 'clip_ratio': number(cfg.get('clip_ratio')),
            'gae_lambda': number(cfg.get('gae_lambda')), 'reward_version': metrics.get('reward_config', {}).get('version')}
    result = []
    for rollout_index, rollout in enumerate(rounds):
        seen = Counter()
        requested = Counter(rollout.get('task_ids', []))
        rollout_start = len(result)
        for is_fault, observations in ((False, rollout.get('episodes', [])), (True, rollout.get('faults', []))):
            for entry in observations:
                task_id = entry.get('task_id'); seen[task_id] += 1
                meta = metadata.get(task_id, {})
                fault = is_fault or entry.get('fault') is True
                truncated = entry.get('truncated') is True
                censored = truncated or entry.get('termination_reason') == 'collector_budget'
                row = dict(base, rollout=rollout_index, task_id=task_id,
                           world_seed=number(meta.get('world_seed', entry.get('world_seed'))),
                           metadata_joined=bool(meta), requested_occurrence=seen[task_id] <= requested[task_id], **{k: meta.get(k) for k in EPISODE_METADATA_FIELDS},
                           outcome='fault' if fault else entry.get('outcome'), fault=fault,
                           truncated=truncated, censored=censored, unstarted=False,
                           terminated=entry.get('terminated') if isinstance(entry.get('terminated'), bool) else None,
                           termination_reason=entry.get('termination_reason'),
                           end_act=number(entry.get('act')), numeric_eligible=not (fault or censored) and entry.get('terminated') is True,
                           fault_stage=entry.get('fault_stage') if fault else None,
                           fault_error=str(entry.get('error', ''))[:160] if fault else None,
                           **{k: number(entry.get(k)) for k in EPISODE_NUMERIC_FIELDS})
                floor, start = row['floor'], number(meta.get('start_floor'))
                row['floor_progress'] = floor - start if floor is not None and start is not None else None
                result.append(row)
        # Every missing request is visible in denominators. Only mark unstarted
        # when the collector's explicit count supports that classification.
        missing = Counter(rollout.get('task_ids', [])) - seen
        explicit_unstarted = rollout.get('unstarted_tasks', 0)
        wholly_unstarted = sum(missing.values()) == explicit_unstarted
        for task_id in missing.elements():
            meta = metadata.get(task_id, {})
            result.append(dict(base, rollout=rollout_index, task_id=task_id,
                world_seed=number(meta.get('world_seed')), metadata_joined=bool(meta), requested_occurrence=True,
                **{k: meta.get(k) for k in EPISODE_METADATA_FIELDS},
                outcome='unstarted' if wholly_unstarted else 'missing', unstarted=wholly_unstarted,
                fault=False, truncated=False, censored=True, terminated=False,
                termination_reason='unstarted' if wholly_unstarted else 'missing_record',
                numeric_eligible=False, end_act=None, floor_progress=None, fault_stage=None, fault_error=None,
                **{k: None for k in EPISODE_NUMERIC_FIELDS}))
        coverage_valid = seen == requested
        for row in result[rollout_start:]:
            row['coverage_valid'] = coverage_valid
            row['numeric_eligible'] = row['numeric_eligible'] and coverage_valid and row['requested_occurrence']
    return result


def episode_filters(query):
    accepted = {'kind', 'round_min', 'round_max', 'source', 'start_act', 'goal_act', 'component',
                'session', 'outcome', 'stage', 'limit', 'offset', 'max_jobs', 'role', 'horizon', 'start_floor_min', 'start_floor_max'}
    unknown = set(query) - accepted
    if unknown: raise ValueError('Unknown episode filter: ' + ','.join(sorted(unknown)))
    def integer(key, default, lower, upper):
        raw = query.get(key, [str(default)])
        if len(raw) != 1: raise ValueError('Duplicate ' + key)
        value = int(raw[0])
        if not lower <= value <= upper: raise ValueError('Out-of-range ' + key)
        return value
    kind = query.get('kind', ['train'])
    if len(kind) != 1 or kind[0] not in ('train', 'dev', 'full3', 'all'): raise ValueError('Invalid episode kind')
    filters = {'kind': kind[0], 'limit': integer('limit', 2000, 1, 20000), 'offset': integer('offset', 0, 0, 10000000),
               'max_jobs': integer('max_jobs', 400, 1, 1000),
               'round_min': integer('round_min', 0, 0, 10000000), 'round_max': integer('round_max', 10000000, 0, 10000000)}
    if filters['round_min'] > filters['round_max']: raise ValueError('Reversed round range')
    for key in ('source', 'start_act', 'goal_act', 'component', 'session', 'outcome', 'stage', 'horizon', 'role'):
        values = query.get(key, [])
        if len(values) > 16 or any(len(v) > 120 for v in values): raise ValueError('Filter exceeds bound')
        filters[key] = set(part for value in values for part in value.split(',') if part)
    filters['start_floor_min'] = integer('start_floor_min', -1000, -1000, 1000000) if 'start_floor_min' in query else None
    filters['start_floor_max'] = integer('start_floor_max', 1000000, -1000, 1000000) if 'start_floor_max' in query else None
    if filters['start_floor_min'] is not None and filters['start_floor_max'] is not None and filters['start_floor_min'] > filters['start_floor_max']:
        raise ValueError('Reversed starting-floor range')
    return filters


class TelemetryStore:
    """Journal refresh is rate-limited; each changed metric is reduced once."""
    def __init__(self, campaign: Path, refresh_seconds: float = 8):
        self.campaign = campaign.resolve()
        self.refresh_seconds = refresh_seconds
        self.cache = CachedJSON(limit=16)
        self.metrics_cache = OrderedDict()
        self.episode_cache = OrderedDict()
        self.metadata_cache = OrderedDict()
        self.episode_cache_rows = 0
        self.jobs_snapshot = []
        self.journal_snapshot = {}
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
            row["round"] = self.training_round(job)
            self.metrics_cache[path] = (stamp, row)
            while len(self.metrics_cache) > 5000:
                self.metrics_cache.popitem(last=False)
            return row
        except FileNotFoundError:
            return None  # New jobs have no metrics until their first atomic write.
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path.name}: {type(exc).__name__}")
            return None

    def task_metadata(self, job):
        """Verify signed task bytes and use compact metadata, never serve prefixes."""
        value = command_option(job.get('command', []), '--tasks')
        identity = input_identity(job, '--tasks')
        if not value: return {}, None
        train = bounded_owned_path(self.campaign / 'tasks', value, r'[A-Za-z0-9_.-]+\.jsonl')
        dev = bounded_owned_path(self.campaign.parent, value, r'dev_act[123]\.jsonl')
        path = train or dev
        if path is None or not identity.get('sha256') or Path(identity.get('path', '')).resolve() != path:
            return {}, None
        manifest_path = Path(str(path) + '.manifest.json') if train else None
        try:
            stat = path.stat()
            if manifest_path and manifest_path.resolve().parent != self.campaign / 'tasks':
                raise ValueError('task manifest escaped owned directory')
            manifest_stat = manifest_path.stat() if manifest_path else None
            stamp = (stat.st_mtime_ns, stat.st_size, identity['sha256'],
                     (manifest_stat.st_mtime_ns, manifest_stat.st_size) if manifest_stat else None)
            old = self.metadata_cache.get(path)
            if old and old[0] == stamp:
                self.metadata_cache.move_to_end(path); return old[1], old[2]
            if stat.st_size > 16 * 1024**2 or (manifest_stat and manifest_stat.st_size > 4 * 1024**2):
                raise ValueError('task metadata exceeds bound')
            if content_sha(path) != identity['sha256']: raise ValueError('signed task bytes changed')
            manifest = json.loads(manifest_path.read_text()) if manifest_path else None
            if manifest and manifest.get('task_file_sha256') != identity['sha256']:
                raise ValueError('task manifest is not bound to signed bytes')
            by_id = {row['task_id']: row for row in manifest.get('tasks', [])} if manifest else {}
            if len(by_id) > 10000: raise ValueError('too many task metadata rows')
            metadata = {}
            with path.open() as stream:
                for index, line in enumerate(stream):
                    if index >= 10000: raise ValueError('too many task rows')
                    row = json.loads(line)
                    canonical = {k: v for k, v in row.items() if k != 'long_metadata'}
                    task_id = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if train:
                        entry = by_id.get(task_id)
                        embedded = row.get('long_metadata', {})
                        if not entry or any(entry.get(k) != embedded.get(k) for k in EPISODE_METADATA_FIELDS):
                            raise ValueError('TaskID/family metadata do not match immutable task')
                        meta = {k: entry.get(k) for k in EPISODE_METADATA_FIELDS}
                        meta['world_seed'] = entry.get('world_seed')
                        if meta['world_seed'] != row.get('seed') or meta['goal_act'] != row.get('max_act') or meta['horizon'] != row.get('horizon'):
                            raise ValueError('task goal/world metadata mismatch')
                    else:
                        if row.get('split') != 'dev': raise ValueError('not a development task')
                        fresh = row.get('prefix') == []
                        meta = {'world_seed': row.get('seed'), 'start_source': 'fresh_scratch' if fresh else None,
                                'start_act': 1 if fresh else None, 'start_floor': 0 if fresh else None,
                                'goal_act': row.get('max_act'), 'horizon': row.get('horizon'), 'component': 'fixed_dev'}
                    metadata[task_id] = meta
            round_index = manifest.get('round_index') if manifest else None
            self.metadata_cache[path] = (stamp, metadata, round_index)
            self.metadata_cache.move_to_end(path)
            while len(self.metadata_cache) > 1000: self.metadata_cache.popitem(last=False)
            return metadata, round_index
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.errors.append(f'{path.name}: task metadata {type(exc).__name__}')
            return {}, None

    def training_round(self, job):
        # A signed producer task basename gives an inexpensive global round for
        # compact telemetry. Episode queries independently verify its manifest.
        tasks = command_option(job.get('command', []), '--tasks')
        identity = input_identity(job, '--tasks')
        if not tasks or identity.get('path') != tasks or not identity.get('sha256'): return None
        match = re.fullmatch(r'[A-Za-z0-9_]+_(\d{8})\.jsonl', Path(tasks).name)
        return int(match.group(1)) if match else None

    def episode_rows(self, job, index, *, kind, round_index=None, roles=()):
        path = curated_metric_path(self.campaign, job.get('metrics')) if kind != 'full3' else (bounded_owned_path(
            self.campaign.parent / 'full_run_eval' / 'jobs', job.get('metrics'), r'[a-f0-9]{64}_a\d+\.metrics\.json') or
            (bounded_owned_path(self.campaign.parent, job.get('metrics'), r'anchor_dev_act3\.metrics\.json') if 'initial_anchor' in roles else None))
        if path is None: return None
        try:
            stat = path.stat()
            metadata, manifest_round = self.task_metadata(job)
            stamp = (stat.st_mtime_ns, stat.st_size, json.dumps(job, sort_keys=True), round_index, tuple(roles), id(metadata))
            old = self.episode_cache.get(path)
            if old and old[0] == stamp:
                self.episode_cache.move_to_end(path); return old[1]
            if stat.st_size > MAX_JSON_BYTES: raise ValueError('metric exceeds bound')
            expected = job.get('artifacts', {}).get(str(path)) or job.get('metrics_sha256')
            if expected and content_sha(path) != expected: raise ValueError('completed metrics SHA changed')
            metrics = json.loads(path.read_text())
            if kind == 'full3':
                cfg, env = metrics.get('config', {}), metrics.get('training_environment', {})
                expected_ids = job.get('expected_task_ids', [])
                actual_ids = [value for r in metrics.get('rounds', []) for value in r.get('task_ids', [])]
                backend = self.journal_snapshot.get('identity', {}).get('backend', {}).get('provenance', {})
                if (Counter(actual_ids) != Counter(expected_ids) or not expected_ids or
                        cfg.get('checkpoint') != input_identity(job, '--checkpoint').get('path') or
                        cfg.get('tasks') != input_identity(job, '--tasks').get('path') or
                        cfg.get('seed') != 777 or cfg.get('max_act') != 3 or cfg.get('horizon') != 4096 or
                        cfg.get('split') != 'dev' or cfg.get('simulations') != 200 or
                        env.get('native_module_sha256') != backend.get('module_sha256')):
                    raise ValueError('full3 raw metrics provenance/cohort mismatch')
            if kind == 'train': round_index = manifest_round
            rows = compact_episodes(metrics, job, index, metadata, round_index=round_index, kind=kind, roles=roles)
            if old: self.episode_cache_rows -= len(old[1])
            self.episode_cache[path] = (stamp, rows)
            self.episode_cache.move_to_end(path); self.episode_cache_rows += len(rows)
            while self.episode_cache and (self.episode_cache_rows > MAX_EPISODE_CACHE_ROWS or len(self.episode_cache) > 1000):
                _, (_, removed) = self.episode_cache.popitem(last=False); self.episode_cache_rows -= len(removed)
            return rows
        except FileNotFoundError: return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.errors.append(f'{path.name}: episode {type(exc).__name__}')
            return None

    def episodes(self, query):
        filters = episode_filters(query)
        with self.lock:
            if self.snapshot is None or time.monotonic() - self.last_refresh >= self.refresh_seconds:
                self.refresh()
            contexts = {}
            for event in self.journal_snapshot.get('events', []):
                for field in ('quality', 'retention'):
                    quality = event.get(field) or {}
                    if quality.get('metrics'): contexts[quality['metrics']] = (event.get('round'), event.get('stage'))
            entries = []
            for index, job in enumerate(self.jobs_snapshot):
                kind = 'train' if job.get('kind') == 'train' else 'dev' if job.get('kind') == 'eval' else None
                if not kind or filters['kind'] not in ('all', kind): continue
                round_index = self.training_round(job) if kind == 'train' else contexts.get(job.get('metrics'), (None, None))[0]
                if round_index is None:
                    # Phase anchors deliberately have no candidate round.
                    if filters['round_min'] > 0 or filters['round_max'] < 10000000: continue
                elif not filters['round_min'] <= round_index <= filters['round_max']: continue
                copied = dict(job)
                if kind == 'dev': copied['stage'] = contexts.get(job.get('metrics'), (None, None))[1]
                entries.append((number(job.get('finished')) or number(job.get('started')) or 0, copied, index, kind, round_index, ()))
            skipped_panels = 0
            if filters['kind'] in ('all', 'full3'):
                observer = self.observer_telemetry() or {}
                reference = (self.journal_snapshot.get('goal_leaders', {}).get('dev_act3', {}).get('quality') or {}).get('cohort', [])
                producer_valid = observer.get('monitoring_only') is True and observer.get('version') == 1 and observer.get('panel') == 'dev_act3'
                histories = list(observer.get('histories', [])) if producer_valid else []
                if not producer_valid: self.errors.append('No valid monitoring-only full3 producer')
                baseline = observer.get('baseline')
                if producer_valid and isinstance(baseline, dict):
                    histories.insert(0, dict(baseline, roles=['initial_anchor'], captured_global_round=0, evaluated_at=baseline.get('captured_at')))
                for index, row in enumerate(histories):
                    condition = row.get('condition_signature', {})
                    inputs = condition.get('inputs', {})
                    roles = row.get('roles', [])
                    anchor = 'initial_anchor' in roles
                    valid = (Counter(row.get('task_ids', [])) == Counter(reference) and len(reference) == 48 and len(set(reference)) == 48 and
                             row.get('checkpoint_sha256') == inputs.get('--checkpoint', {}).get('sha256') and
                             bool(row.get('metrics_sha256')) and
                             bounded_owned_path(self.campaign.parent, inputs.get('--tasks', {}).get('path'), r'dev_act3\.jsonl') is not None)
                    if anchor: valid = valid and row.get('checkpoint_sha256') == self.journal_snapshot.get('identity', {}).get('initial_checkpoint_sha256')
                    if not valid: skipped_panels += 1; continue
                    round_index = number(row.get('captured_global_round'))
                    if round_index is not None and not filters['round_min'] <= round_index <= filters['round_max']: continue
                    fake = {'metrics': row.get('metrics_path'), 'metrics_sha256': row.get('metrics_sha256'),
                            'signature': condition, 'command': condition.get('arguments', []),
                            'started': row.get('captured_at'), 'finished': row.get('evaluated_at'),
                            'status': 'complete' if row.get('complete') else 'incomplete', 'panel_valid': True,
                            'checkpoint_sha256': row.get('checkpoint_sha256'), 'session_kind': 'observer', 'stage': 'full3_monitor',
                            'expected_task_ids': reference}
                    entries.append((number(row.get('evaluated_at')) or 0, fake, index, 'full3', round_index, roles))
            entries.sort(key=lambda entry: (entry[0], entry[2]))
            eligible = len(entries); entries = entries[-filters['max_jobs']:]
            dataset_entries = []
            def file_stamp(path):
                if path is None: return None
                try:
                    stat = path.stat(); return (str(path), stat.st_mtime_ns, stat.st_size)
                except OSError: return (str(path), None, None)
            for _, job, index, kind, round_index, roles in entries:
                metric_path = curated_metric_path(self.campaign, job.get('metrics')) if kind != 'full3' else (
                    bounded_owned_path(self.campaign.parent / 'full_run_eval' / 'jobs', job.get('metrics'), r'[a-f0-9]{64}_a\d+\.metrics\.json') or
                    (bounded_owned_path(self.campaign.parent, job.get('metrics'), r'anchor_dev_act3\.metrics\.json') if 'initial_anchor' in roles else None))
                task_value = command_option(job.get('command', []), '--tasks')
                task_path = bounded_owned_path(self.campaign / 'tasks', task_value, r'[A-Za-z0-9_.-]+\.jsonl') or bounded_owned_path(self.campaign.parent, task_value, r'dev_act[123]\.jsonl')
                manifest_path = Path(str(task_path) + '.manifest.json') if task_path and task_path.parent == self.campaign / 'tasks' else None
                if manifest_path and manifest_path.resolve().parent != self.campaign / 'tasks': manifest_path = None
                dataset_entries.append((index, kind, round_index, list(roles), job.get('status'), job.get('finished'),
                                        file_stamp(metric_path), file_stamp(task_path), file_stamp(manifest_path), job.get('metrics_sha256')))
            semantic_filters = {k: sorted(v) if isinstance(v, set) else v for k, v in filters.items() if k not in ('offset', 'limit')}
            dataset_id = hashlib.sha256(json.dumps({'jobs': dataset_entries, 'filters': semantic_filters}, sort_keys=True).encode()).hexdigest()
            rows = []; missing = 0; joined = unjoined = unstarted = metadata_missing = 0
            options = {field: set() for field in ('start_source','start_act','goal_act','component','session','outcome','stage')}
            for _, job, index, kind, round_index, roles in entries:
                compact = self.episode_rows(job, index, kind=kind, round_index=round_index, roles=roles)
                if compact is None: missing += 1; continue
                metadata_missing += any(not row['metadata_joined'] for row in compact)
                for row in compact:
                    joined += row['metadata_joined']; unjoined += not row['metadata_joined']; unstarted += row['unstarted']
                    for field in options:
                        if row.get(field) is not None: options[field].add(str(row[field]))
                    matches = all(not filters[key] or str(row.get(field)) in filters[key] for key, field in
                                  (('source','start_source'), ('start_act','start_act'), ('goal_act','goal_act'),
                                   ('component','component'), ('session','session'), ('outcome','outcome'), ('stage','stage'), ('horizon','horizon')))
                    matches = matches and (not filters['role'] or bool(filters['role'].intersection(row['roles'])))
                    for key, compare in (('start_floor_min', lambda value, bound: value >= bound), ('start_floor_max', lambda value, bound: value <= bound)):
                        if filters[key] is not None: matches = matches and row.get('start_floor') is not None and compare(row['start_floor'], filters[key])
                    if matches: rows.append(row)
            offset, limit = filters['offset'], filters['limit']; page = rows[offset:offset + limit]
            availability = {'logged_numeric': list(EPISODE_NUMERIC_FIELDS) + ['floor_progress'],
                            'unavailable': ['start_hp', 'end_hp', 'max_hp', 'hp_percent', 'action_trajectories', 'phase_gradient_norm', 'explained_variance'],
                            'hp_cost_semantics': 'reset/start HP minus end/current HP; net difference, healing may be negative',
                            'numeric_default': 'exclude fault, truncated, censored and unstarted',
                            'filter_options': {field: sorted(values) for field, values in options.items()},
                            'numeric_eligible_rows': sum(row['numeric_eligible'] for row in rows),
                            'numeric_present_rows': {field: sum(row['numeric_eligible'] and row.get(field) is not None for row in rows) for field in EPISODE_NUMERIC_FIELDS},
                            'scope': 'bounded latest matching-job window; not all study episodes if omitted_jobs>0',
                            'rows_returned': len(page), 'rows_matched': len(rows), 'scoped_requested_rows': sum(row['requested_occurrence'] for row in rows),
                            'unexpected_observed_rows': sum(not row['requested_occurrence'] for row in rows),
                            'invalid_coverage_rows': sum(not row['coverage_valid'] for row in rows)}
            return {'schema': 1, 'dataset_id': dataset_id, 'episodes': page, 'total': len(rows), 'offset': offset, 'limit': limit,
                    'has_more': offset + len(page) < len(rows), 'next_offset': offset + len(page) if offset + len(page) < len(rows) else None,
                    'total_pages': math.ceil(len(rows) / limit), 'availability': availability,
                    'coverage': {'eligible_jobs': eligible, 'scanned_jobs': len(entries), 'omitted_jobs': eligible-len(entries),
                                 'missing_metrics': missing, 'metadata_joined_rows': joined, 'unjoined_rows': unjoined,
                                 'missing_task_metadata': metadata_missing,
                                 'unstarted_rows': unstarted, 'skipped_invalid_panels': skipped_panels}, 'errors': self.errors[-20:]}

    def refresh(self):
        self.errors = []
        j = self.document("journal.json", {})
        progress = self.document("progress.json", {})
        config_record = self.document("config.json", {})
        config = config_record.get("config", {})
        rows, families = [], {}
        jobs = j.get("jobs", [])
        self.jobs_snapshot = jobs
        self.journal_snapshot = j
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
            parsed = urlsplit(self.path)
            path = parsed.path
            if path == "/api/episodes":
                try:
                    if len(parsed.query) > 4096: raise ValueError("Query exceeds size bound")
                    response = store.episodes(parse_qs(parsed.query, keep_blank_values=True, max_num_fields=60))
                except (ValueError, TypeError):
                    self.send_error(400, "Invalid bounded episode query")
                    return
                body = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
                content_type = "application/json"
            elif path == "/api/telemetry":
                body = json.dumps(store.get(), allow_nan=False, separators=(",", ":")).encode()
                content_type = "application/json"
            elif path in ("/", "/index.html", "/app.js", "/style.css"):
                asset = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                body = (ASSETS / asset).read_bytes()
                content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[asset]
            else:
                self.send_error(404)
                return
            compressed = content_type == "application/json" and accepts_gzip(self.headers.get("Accept-Encoding"))
            if compressed:
                body = gzip.compress(body, compresslevel=1, mtime=0)
            self.send_response(200)
            if content_type == "application/json": self.send_header("Vary", "Accept-Encoding")
            if compressed: self.send_header("Content-Encoding", "gzip")
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
