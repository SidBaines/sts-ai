#!/usr/bin/env python
"""Bounded, single-attempt Ollama fallback for frozen September probe cohorts.

Default mode only prepares request files; --execute enables HTTP inference.
This is an independent backend/model/quantization/template bundle experiment,
not a matched-MLX model-size comparison. No grammar constraint is requested.
API contract: https://docs.ollama.com/api/chat
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse

from sts_ai.agents import parse_json_action
from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction

CAVEAT = (
    "Ollama HTTP fallback: model weights, architecture, quantization, backend, "
    "chat template, default model parameters and random-number stream can differ "
    "from MLX. Requested seeds are recorded, not guaranteed by backend to be "
    "honored or matched to MLX. Interpret as a complete model/backend bundle, "
    "never as a parameter-count-only effect. One free generation per decision; "
    "no retries, teacher scoring, or constrained decoding. The original MLX "
    "suite seeds batches; this runner seeds each probe independently."
)
HTTP_WORKER = '''import json, sys, urllib.request, urllib.error
p = json.load(sys.stdin)
data = None if p["body"] is None else json.dumps(p["body"]).encode()
req = urllib.request.Request(p["url"], data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=p["timeout"]) as response:
        raw = response.read().decode("utf-8", errors="replace")
        print(json.dumps({"status": response.status, "raw_body": raw}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"status": exc.code, "raw_body": exc.read().decode("utf-8", errors="replace")}))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    ids = [row.get("probe_id") for row in rows]
    if not rows or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("cohort requires nonempty, unique nonnegative integer probe IDs")
    for row in rows:
        actions = [LegalAction(**a) for a in row["legal_actions"]]
        if not actions or [a.index for a in actions] != list(range(len(actions))):
            raise ValueError("cohort requires nonempty, contiguous displayed action indices")
        if not isinstance(row.get("state_text"), str):
            raise ValueError("cohort state_text must be a string")
        for key in ("probe_screen", "world_seed", "decision_index", "source_trace"):
            if key not in row:
                raise ValueError(f"missing cohort identity {key}")


def make_request(row: dict[str, Any], *, model: str, temperature: float,
                 max_tokens: int, seed: int, num_ctx: int) -> dict[str, Any]:
    actions = [LegalAction(**a) for a in row["legal_actions"]]
    return {
        "model": model,
        "messages": [{"role": "user", "content": render_action_prompt(
            row["state_text"], actions, output_contract="action_text")}],
        "stream": False, "think": False, "keep_alive": "5m",
        "options": {"temperature": temperature, "num_predict": max_tokens,
                    "seed": (seed + row["probe_id"]) % 2147483647,
                    "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                    "repeat_penalty": 1.0, "num_ctx": num_ctx},
    }


def http_request(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    """Kill the HTTP client at a wall deadline, even if a server drips bytes."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", HTTP_WORKER],
            input=json.dumps({"url": url, "body": body, "timeout": timeout}),
            text=True, capture_output=True, timeout=timeout, check=False,
        )
        if completed.returncode:
            return {"error": "transport_error", "stderr": completed.stderr,
                    "stdout": completed.stdout, "latency_s": time.monotonic() - started}
        result = json.loads(completed.stdout)
        result["latency_s"] = time.monotonic() - started
        try:
            result["response"] = json.loads(result["raw_body"])
        except json.JSONDecodeError:
            result["error"] = "non_json_http_response"
        if result["status"] != 200:
            result["error"] = "http_error"
        return result
    except subprocess.TimeoutExpired:
        return {"error": "client_wall_timeout", "latency_s": time.monotonic() - started,
                "note": "Client killed; server-side cancellation is not guaranteed."}


def parse_response(row: dict[str, Any], wire: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    identity = {k: row[k] for k in ("probe_id", "probe_screen", "world_seed", "decision_index")}
    response = wire.get("response", {})
    if wire.get("error") or not isinstance(response, dict):
        return {**identity, "decision": None, "error": wire.get("error", "invalid_response_object")}
    message = response.get("message")
    if (response.get("done") is not True or not isinstance(message, dict)
            or not isinstance(message.get("content"), str) or response.get("error")):
        return {**identity, "decision": None, "error": "incomplete_or_invalid_chat_response"}
    # Parse only actual assistant content. Preserve any separate thinking in raw wire.
    decision = parse_json_action(message["content"], [LegalAction(**a) for a in row["legal_actions"]],
                                 completion_tokens=response.get("eval_count", 0),
                                 max_tokens=max_tokens, output_contract="action_text")
    decision.latency_s = wire.get("latency_s", 0.0)
    decision.prompt_tokens = response.get("prompt_eval_count", 0)
    decision.completion_tokens = response.get("eval_count", 0)
    decision.metadata.update({"backend": "ollama_http", "single_attempt": True,
                              "done_reason": response.get("done_reason"),
                              "separate_thinking_returned": bool(message.get("thinking"))})
    return {**identity, "decision": asdict(decision)}


def summarize(predictions: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    valid = sum(bool(p.get("decision") and p["decision"]["valid"]) for p in predictions)
    errors = sum(p.get("decision") is None for p in predictions)
    screens = sorted({p["probe_screen"] for p in predictions})
    return {"caveat": CAVEAT, "expected": expected, "attempted": len(predictions),
            "all_requests_attempted": len(predictions) == expected,
            "complete": len(predictions) == expected and errors == 0,
            "transport_or_protocol_errors": errors, "valid": valid,
            "valid_rate_per_expected": valid / expected,
            "by_screen": {s: {"attempted": sum(p["probe_screen"] == s for p in predictions),
                              "valid": sum(p["probe_screen"] == s and bool(p.get("decision") and p["decision"]["valid"])
                                           for p in predictions)} for s in screens}}


def run(args: argparse.Namespace) -> None:
    if urlparse(args.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("fallback runner only supports the local Ollama endpoint")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if min(args.timeout, args.total_seconds, args.max_tokens, args.num_ctx) <= 0 or args.temperature < 0:
        raise ValueError("positive bounds and nonnegative temperature required")
    original = args.source.read_bytes()
    rows = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
    validate_rows(rows)
    rows = rows[:args.limit] if args.limit is not None else rows
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "source_cohort.jsonl").write_bytes(original)
    planned = [{"probe_id": row["probe_id"], "request": make_request(
        row, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
        seed=args.seed, num_ctx=args.num_ctx)} for row in rows]
    (args.out / "requests.jsonl").write_text("".join(json.dumps(p) + "\n" for p in planned))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"source_sha256": hashlib.sha256(original).hexdigest(),
                   "requests_sha256": sha256(args.out / "requests.jsonl"),
                   "script_sha256": sha256(Path(__file__)), "caveat": CAVEAT,
                   "seed_algorithm": "(seed + probe_id) % 2147483647",
                   "selected_probe_ids": [row["probe_id"] for row in rows],
                   "screen_counts": dict(Counter(row["probe_screen"] for row in rows)),
                   "status": "prepared"})
    write_json(args.out / "config.json", config)
    if not args.execute:
        print(json.dumps({"status": "prepared_no_http_calls", "requests": len(planned), "out": str(args.out)}))
        return
    deadline = time.monotonic() + args.total_seconds
    metadata = {}
    for endpoint, body in (("/api/version", None), ("/api/tags", None),
                           ("/api/show", {"model": args.model, "verbose": True})):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("total deadline expired during metadata collection")
        metadata[endpoint] = http_request(args.base_url.rstrip("/") + endpoint, body,
                                         min(args.timeout, remaining))
        write_json(args.out / "backend_metadata.json", metadata)
        if metadata[endpoint].get("error"):
            raise RuntimeError(f"metadata collection failed: {endpoint}; no inference sent")
    predictions = []
    with (args.out / "responses.jsonl").open("x") as raw_out, (args.out / "predictions.jsonl").open("x") as pred_out:
        for row, plan in zip(rows, planned):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wire = http_request(args.base_url.rstrip("/") + "/api/chat", plan["request"],
                                min(args.timeout, remaining))
            raw_out.write(json.dumps({**plan, "wire": wire}) + "\n")
            raw_out.flush()
            prediction = parse_response(row, wire, args.max_tokens)
            predictions.append(prediction)
            pred_out.write(json.dumps(prediction) + "\n")
            pred_out.flush()
            write_json(args.out / "report.json", summarize(predictions, len(rows)))
            print(json.dumps({"completed": len(predictions), "expected": len(rows),
                              "probe_id": row["probe_id"], "valid": bool(prediction.get("decision") and prediction["decision"]["valid"])}), flush=True)
            # Fail stop prevents a client timeout from stacking another GPU request.
            if prediction.get("error"):
                break
    report = summarize(predictions, len(rows))
    write_json(args.out / "report.json", report)
    if not report["complete"]:
        raise RuntimeError("probe incomplete; inspect report and raw response; no automatic retry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:26b-mlx")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=905)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--total-seconds", type=float, default=600)
    parser.add_argument("--execute", action="store_true", help="Send real HTTP inference; default prepares only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
