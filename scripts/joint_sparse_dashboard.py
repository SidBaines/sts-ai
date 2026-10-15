#!/usr/bin/env python3
"""Bounded read-only telemetry for one joint sparse-win condition. Localhost only."""

import argparse
import gzip
import json
import math
import re
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web/joint_sparse_dashboard"
MAX_BYTES = 32 * 1024 * 1024


def read_json(path, default=None):
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Document bound exceeded")
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        out[prefix] = value
    return out


class Telemetry:
    def __init__(self, out):
        self.out = Path(out)
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def events(self):
        path = self.out / "log.jsonl"
        try:
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Log bound exceeded")
            lines = path.read_text().splitlines()
        except FileNotFoundError:
            return []
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def episodes(self):
        paths = sorted(self.out.glob("collection_*.json.gz")) + sorted(
            self.out.glob("evaluation_*.json.gz")
        )
        if (self.out / "baseline_receipt.json.gz").exists():
            paths.insert(0, self.out / "baseline_receipt.json.gz")
        all_rows = []
        for path in paths[-128:]:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            key = str(path)
            if key in self.cache and self.cache[key][0] == stamp:
                rows = self.cache[key][1]
            else:
                with gzip.open(path, "rb") as f:
                    raw = f.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("Episode receipt bound exceeded")
                payload = json.loads(raw)
                rows = []
                for e in payload.get("episodes", []):
                    # Whitelist public episode summaries; exclude fault paths, trace and payloads.
                    row = {
                        k: e.get(k)
                        for k in (
                            "world_seed",
                            "dataset",
                            "iteration",
                            "classification",
                            "binary_return",
                            "steps",
                            "floor",
                            "act",
                            "max_act_reached",
                            "reward",
                            "terminated",
                            "truncated",
                            "decision_counts",
                            "boss_entries",
                            "end_hp",
                            "end_max_hp",
                            "termination_state",
                            "termination_reason",
                        )
                    }
                    row["dataset"] = (
                        "eval"
                        if path.name.startswith("evaluation_")
                        else (
                            "baseline" if path.name.startswith("baseline") else "train"
                        )
                    )
                    row["receipt"] = path.name
                    rows.append(row)
                for f in payload.get("faults", []):
                    rows.append(
                        {
                            "world_seed": f.get("world_seed"),
                            "dataset": "train",
                            "classification": "fault",
                            "error_type": f.get("error"),
                            "iteration": int(re.search(r"\d+", path.name)[0])
                            if re.search(r"\d+", path.name)
                            else 0,
                            "receipt": path.name,
                        }
                    )
                self.cache[key] = (stamp, rows)
            self.cache.move_to_end(key)
            all_rows.extend(rows)
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return all_rows

    def snapshot(self, query):
        with self.lock:
            events = self.events()
            series = []
            keys = set()
            for e in events:
                if e.get("event") not in (
                    "collection",
                    "update",
                    "evaluation",
                    "critic_health",
                    "critic_validation",
                ):
                    continue
                metrics = flatten(
                    {
                        k: v
                        for k, v in e.items()
                        if k not in ("config", "run_record", "snapshot")
                    }
                )
                keys.update(metrics)
                series.append(
                    {
                        "iteration": e.get("iteration", 0),
                        "time": e.get("t", e.get("time", e.get("timestamp"))),
                        "event": e["event"],
                        "dataset": e.get("dataset", "train"),
                        "metrics": metrics,
                    }
                )
            rows = self.episodes()
            total = len(rows)
            for field in ("dataset", "classification", "act", "iteration"):
                wanted = query.get(field, ["all"])[0]
                if wanted != "all":
                    rows = [r for r in rows if str(r.get(field)) == wanted]
            phase = query.get("phase", ["all"])[0]
            if phase != "all":
                rows = [
                    r
                    for r in rows
                    if any(
                        k.startswith(phase + ":") and v
                        for k, v in (r.get("decision_counts") or {}).items()
                    )
                ]
            valid = [r for r in rows if r.get("classification") in ("win", "loss")]
            wins = sum(r.get("classification") == "win" for r in rows)
            summary = {
                "filtered": len(rows),
                "all_episodes": total,
                "valid": len(valid),
                "wins": wins,
                "faults": sum(r.get("classification") == "fault" for r in rows),
                "win_rate_valid": wins / len(valid) if valid else None,
            }
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(1000, max(1, int(query.get("limit", ["100"])[0])))
            except ValueError:
                raise ValueError("Invalid pagination")
            status = read_json(self.out / "status.json", {})
            budget = read_json(self.out / "budget.json", {})
            baseline = read_json(self.out / "baseline.json", {})
            champion = read_json(self.out / "champion_eval.json", {})
            queue = read_json(ROOT / "data/rl_runs/runner_status.json", {})
            return {
                "status": status,
                "learning_health": read_json(self.out / "health.json", {}),
                "critic_fresh_validation": read_json(
                    self.out / "critic_fresh_validation.json", {}
                ),
                "budget": budget,
                "queue_power": {
                    "on_battery": queue.get("on_battery"),
                    "caffeinated": queue.get("caffeinated"),
                    "heartbeat": queue.get("heartbeat"),
                },
                "baseline": baseline.get("summary"),
                "champion": champion.get("summary"),
                "series": series,
                "metric_keys": sorted(keys),
                "alerts": [
                    e
                    for e in events
                    if e.get("event") in ("alert", "fatal", "done")
                    or e.get("event") == "update"
                    and not e.get("accepted")
                ][-100:],
                "summary": summary,
                "episodes": rows[offset : offset + limit],
                "offset": offset,
                "limit": limit,
                "filters": {
                    "iterations": sorted(
                        set(str(r.get("iteration")) for r in self.episodes())
                    )
                },
                "updated": time.time(),
            }


def serve(out, port):
    telemetry = Telemetry(out)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/telemetry":
                    data = json.dumps(
                        telemetry.snapshot(parse_qs(parsed.query)), allow_nan=False
                    ).encode()
                    kind = "application/json"
                elif parsed.path in ("/", "/app.js", "/style.css"):
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    data = (ASSETS / name).read_bytes()
                    kind = {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[name]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind + "; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, OSError) as e:
                self.send_error(400, str(e))

        def log_message(self, *args):
            pass

    print(f"Dashboard http://127.0.0.1:{port} run={out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8793)
    a = p.parse_args()
    serve(a.out, a.port)
