#!/usr/bin/env python3
"""Bounded, unconstrained IFM decision-throughput benchmark.

Dependencies: aiohttp; transformers for the default raw /completions route.
Input JSONL: {"id": "...", "phase": "combat", "legal_indices": [0, 1],
              "messages": [{"role": "user", "content": "..."}]}.
One attempt per scheduled prompt, with no output repair or retries.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any


CLOSE_THOUGHT = re.compile(r"</ifm\|(?:think|think_fast|think_faster)>")
OPEN_THOUGHT = re.compile(r"<ifm\|(?:think|think_fast|think_faster)>")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_action(text: str, legal_indices: list[int]) -> dict[str, Any]:
    """Use the same strict final-channel contract as training and evaluation."""
    from sts_llm.generation import parse_action as parse_final
    parsed = parse_final(text, legal_indices)
    closes = list(CLOSE_THOUGHT.finditer(text))
    boundary = closes[0] if closes else None
    unclosed = boundary is None or bool(OPEN_THOUGHT.search(text[boundary.end():]))
    schema = parsed.error in (None, "illegal_action_index")
    return {
        "thinking_closed": not unclosed,
        "unclosed_thinking": unclosed,
        "closing_thinking_tag": boundary.group() if boundary else None,
        "json_valid": schema or parsed.error == "invalid_action_schema",
        "schema_valid": schema,
        "legal_action": parsed.valid,
        "action_index": parsed.action_index,
        "answer_text": parsed.final_text,
        "parse_error": parsed.error,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def row_metrics(rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    attempted = [r for r in rows if r.get("attempted")]
    http_ok = [r for r in attempted if r.get("http_ok")]
    completed = [r for r in http_ok if r.get("response_complete")]
    valid = [r for r in completed if r.get("legal_action") and not r.get("truncated")]
    token_rows = [r for r in http_ok if isinstance(r.get("generated_tokens"), int)]
    lengths = [float(r["generated_tokens"]) for r in token_rows]
    latency = [r.get("http_response_latency_seconds", r.get("latency_seconds")) for r in attempted]
    latency = [value for value in latency if value is not None]
    generated = sum(int(r["generated_tokens"]) for r in token_rows)
    return {
        "scheduled": len(rows), "attempted": len(attempted),
        "http_ok": len(http_ok), "complete_responses": len(completed),
        "valid_legal_untruncated_decisions": len(valid),
        "legal_actions_including_truncated": sum(bool(r.get("legal_action")) for r in http_ok),
        "unclosed_thinking": sum(bool(r.get("unclosed_thinking")) for r in http_ok),
        "invalid_or_missing_final_action": sum(not bool(r.get("legal_action")) for r in http_ok),
        "truncated": sum(bool(r.get("truncated")) for r in http_ok),
        "truncation_rate_of_http_ok": sum(bool(r.get("truncated")) for r in http_ok) / len(http_ok) if http_ok else None,
        "valid_rate_of_attempted": len(valid) / len(attempted) if attempted else None,
        "generated_tokens": generated,
        "requests_with_generated_token_count": len(token_rows),
        "valid_decisions_per_second": len(valid) / wall_seconds if wall_seconds else None,
        "generated_tokens_per_second": generated / wall_seconds if wall_seconds else None,
        "median_latency_seconds": percentile(latency, .5),
        "p95_latency_seconds": percentile(latency, .95),
        "median_output_tokens": percentile(lengths, .5),
        "p95_output_tokens": percentile(lengths, .95),
        "max_output_tokens": max(lengths) if lengths else None,
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in http_ok)),
        "errors": dict(Counter(r["error_kind"] for r in rows if r.get("error_kind"))),
        "token_count_sources": dict(Counter(str(r.get("generated_token_count_source")) for r in token_rows)),
        "requests_with_actual_output_token_ids": sum(isinstance(r.get("output_token_ids"), list) for r in http_ok),
    }


def load_prompts(path: Path, tokenizer: Any | None, effort: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        legal = row.get("legal_indices")
        if not isinstance(legal, list) or not legal or any(isinstance(i, bool) or not isinstance(i, int) for i in legal):
            raise ValueError(f"line {line_number}: legal_indices must be a nonempty list of integers")
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ValueError(f"line {line_number}: messages must be a nonempty list")
        item = {"id": str(row.get("id", line_number)), "phase": str(row.get("phase", "unknown")),
                "legal_indices": legal, "messages": row["messages"], "input_row": row}
        if tokenizer is not None:
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                add_generation_prompt=True, return_dict=False, reasoning_effort=effort)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if not isinstance(ids, list) or not ids or any(not isinstance(i, int) for i in ids):
                raise ValueError(f"line {line_number}: tokenizer did not return a flat token-ID list")
            item["prompt_token_ids"] = ids
            item["rendered_prompt"] = tokenizer.decode(ids, skip_special_tokens=False)
            item["prompt_token_count"] = len(ids)
        item["prompt_sha256"] = hashlib.sha256(json.dumps(
            {k: item[k] for k in ("messages", "legal_indices")}, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("prompt file is empty")
    return rows


def extract_response(response: dict[str, Any], row: dict[str, Any], tokenizer: Any | None) -> dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        return {"response_complete": False, "error_kind": "missing_choices"}
    choice = choices[0]
    message = choice.get("message") or {}
    raw_text = choice.get("text")
    if raw_text is None:
        raw_text = message.get("content")
    if not isinstance(raw_text, str):
        raw_text = "" if raw_text is None else json.dumps(raw_text, ensure_ascii=False)
    output_ids = choice.get("token_ids", response.get("token_ids"))
    if output_ids is None:
        output_ids = message.get("token_ids")
    if not isinstance(output_ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in output_ids):
        output_ids = None
    usage = response.get("usage") or {}
    generated = usage.get("completion_tokens")
    token_source = "server_usage"
    if not isinstance(generated, int):
        if output_ids is not None:
            generated, token_source = len(output_ids), "server_token_ids"
        elif tokenizer is not None:
            generated = len(tokenizer.encode(raw_text, add_special_tokens=False))
            token_source = "retokenized_text_estimate"
        else:
            generated, token_source = None, None
    finish_reason = choice.get("finish_reason")
    result = {
        "response_complete": True, "raw_text": raw_text,
        "output_token_ids": output_ids, "output_token_ids_source": "server" if output_ids is not None else None,
        "usage": usage, "generated_tokens": generated,
        "generated_token_count_source": token_source,
        "finish_reason": finish_reason, "stop_reason": choice.get("stop_reason"),
        "truncated": finish_reason in ("length", "max_tokens", "max_length"),
        # A reasoning-parser server is unsupported for validity scoring: retain
        # separated reasoning as evidence, but do not fabricate closing tags.
        "server_reasoning_content": message.get("reasoning_content", message.get("reasoning")),
    }
    if output_ids is not None and tokenizer is not None:
        result["decoded_output_token_ids"] = tokenizer.decode(output_ids, skip_special_tokens=False)
    # Decode actual server IDs when available so channel-boundary validation
    # does not depend on a server's optional text-normalization behavior.
    parse_text = result.get("decoded_output_token_ids", raw_text)
    result["parse_text_source"] = "decoded_server_token_ids" if "decoded_output_token_ids" in result else "raw_text"
    result.update(parse_action(parse_text, row["legal_indices"]))
    return result


async def run(args: argparse.Namespace, prompts: list[dict[str, Any]], tokenizer: Any | None) -> dict[str, Any]:
    import aiohttp

    output = Path(args.output)
    schedule = [(prompt, repeat) for repeat in range(args.repeats) for prompt in prompts]
    random.Random(args.seed).shuffle(schedule)
    records: list[dict[str, Any] | None] = [None] * len(schedule)
    queue: asyncio.Queue[tuple[int, dict[str, Any], int]] = asyncio.Queue()
    for index, (prompt, repeat) in enumerate(schedule):
        queue.put_nowait((index, prompt, repeat))
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    endpoint = "chat/completions" if args.endpoint == "chat" else "completions"
    url = f"{base}/{endpoint}"
    partial = output / "requests.partial.jsonl"
    handle = partial.open("x", encoding="utf-8")
    started_utc = utc_now()
    start = time.perf_counter()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, stop.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass
    end_reason = "completed"

    def save(record: dict[str, Any]) -> None:
        records[record["request_index"]] = record
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()

    def identity(index: int, row: dict[str, Any], repeat: int) -> dict[str, Any]:
        return {"request_index": index, "prompt_id": row["id"], "phase": row["phase"],
                "repeat": repeat, "request_seed": args.seed + index,
                "prompt_sha256": row["prompt_sha256"], "legal_indices": row["legal_indices"],
                "effort": args.effort, "case_name": args.case_name}

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
            async def worker() -> None:
                while not stop.is_set():
                    try:
                        index, row, repeat = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    record = identity(index, row, repeat)
                    record.update(attempted=True, http_ok=False, response_complete=False,
                                  started_at=utc_now(), start_offset_seconds=time.perf_counter() - start)
                    began = time.perf_counter()
                    payload: dict[str, Any] = {
                        "model": args.model, "max_tokens": args.max_tokens,
                        "temperature": args.temperature, "top_p": args.top_p,
                        "seed": record["request_seed"], "stream": False,
                        "skip_special_tokens": False, "spaces_between_special_tokens": False,
                        "stop_token_ids": args.stop_token_ids,
                    }
                    if not args.no_return_token_ids:
                        payload["return_token_ids"] = True
                    if args.endpoint == "chat":
                        payload["messages"] = row["messages"]
                        payload["chat_template_kwargs"] = {"reasoning_effort": args.effort}
                    else:
                        payload["prompt"] = row["prompt_token_ids"]
                    try:
                        async with session.post(url, json=payload) as reply:
                            record["http_status"] = reply.status
                            body = await reply.text()
                            record["http_response_latency_seconds"] = time.perf_counter() - began
                            try:
                                response = json.loads(body)
                            except json.JSONDecodeError:
                                record.update(error_kind="non_json_http_response", raw_http_body=body)
                            else:
                                record["raw_response"] = response
                                if reply.status < 200 or reply.status >= 300:
                                    record["error_kind"] = "http_error"
                                else:
                                    record["http_ok"] = True
                                    record.update(extract_response(response, row, tokenizer))
                    except asyncio.CancelledError:
                        record.update(error_kind="cancelled_at_run_end", error=end_reason)
                        raise
                    except asyncio.TimeoutError as exc:
                        record.update(error_kind="request_timeout", error=str(exc))
                    except Exception as exc:
                        record.update(error_kind=type(exc).__name__, error=str(exc))
                    finally:
                        record["latency_seconds"] = time.perf_counter() - began
                        record["ended_at"] = utc_now()
                        save(record)
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(min(args.concurrency, len(schedule)))]
            joined = asyncio.gather(*workers, return_exceptions=True)
            stopped = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait((joined, stopped), timeout=args.run_timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if joined not in done:
                end_reason = "signal" if stopped in done else "run_deadline"
                stop.set()
                for task in workers:
                    task.cancel()
                await joined
            else:
                failures = [repr(value) for value in joined.result() if isinstance(value, BaseException)]
                if failures:
                    end_reason = "worker_error"
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
    except Exception as exc:
        end_reason = f"client_error:{type(exc).__name__}:{exc}"
    finally:
        wall_seconds = time.perf_counter() - start
        for index, (row, repeat) in enumerate(schedule):
            if records[index] is None:
                record = identity(index, row, repeat)
                record.update(attempted=False, http_ok=False, response_complete=False,
                              error_kind="not_started", error=end_reason)
                save(record)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(partial, output / "requests.jsonl")
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        actual = [record for record in records if record is not None]
        summary = {
            "case_name": args.case_name, "model": args.model, "effort": args.effort,
            **args.server_settings,
            "endpoint": endpoint, "url": url,
            "concurrency": args.concurrency, "max_tokens": args.max_tokens,
            "repeats": args.repeats, "seed": args.seed,
            "temperature": args.temperature, "top_p": args.top_p,
            "tokenizer_revision": args.revision,
            "stop_token_ids": args.stop_token_ids,
            "stop_token": "<|ifm|im_end|>",
            "request_timeout_seconds": args.timeout, "run_timeout_seconds": args.run_timeout,
            "started_at": started_utc, "ended_at": utc_now(),
            "request_wall_seconds": wall_seconds, "end_reason": end_reason,
            "metrics": row_metrics(actual, wall_seconds),
            "by_phase": {phase: row_metrics([r for r in actual if r["phase"] == phase], wall_seconds)
                         for phase in sorted({r["phase"] for r in actual})},
            "notes": [
                "Rates use wall time for all scheduled requests, including errors; no warmup is excluded.",
                "Phase rates share the full mixed-workload denominator and sum to the overall rate.",
                "Valid decision requires a verified closing thinking tag, exact action_index schema, legal index, and no length truncation.",
                "Actual output token IDs are retained when provided by the server; missing IDs are explicit, never fabricated.",
                "No malformed-output retries, tools, constrained decoding, or action substitution.",
                "Server generation may continue briefly after a cancelled request; this client does not mutate the server.",
            ],
        }
        atomic_json(output / "summary.json", summary)
    return summary


def self_test() -> None:
    for tag in ("think", "think_fast", "think_faster"):
        result = parse_action(f'<ifm|{tag}>bad brace {{ </ifm|{tag}>{{"action_index": 2}}', [2])
        assert result["legal_action"] and result["thinking_closed"]
    assert not parse_action('{"action_index": 2}', [2])["legal_action"]
    assert not parse_action('<ifm|think>{"action_index": 2}', [2])["legal_action"]
    assert not parse_action('</ifm|think>{"action_index": true}', [1])["schema_valid"]
    assert not parse_action('</ifm|think>{"action_index": 2}', [1])["legal_action"]
    assert not parse_action('</ifm|think>{"action_index": 2}<ifm|think>again', [2])["legal_action"]
    assert not parse_action('</ifm|think>{"action_index": 2}<ifm|think>again</ifm|think>{"action_index": 1}', [1, 2])["legal_action"]
    assert parse_action('</ifm|think>{"action_index": 2}', [2])["legal_action"]
    assert not parse_action('</ifm|think>{"action_index": 2}{"action_index":', [2])["legal_action"]
    assert percentile([1, 2, 3], .5) == 2
    print("Synthetic parser/metric checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--timeout", type=float, default=180, help="Per-request seconds")
    parser.add_argument("--run-timeout", "--deadline", type=float, default=600, help="Whole request cohort seconds")
    parser.add_argument("--case-name", default="benchmark")
    parser.add_argument("--server-config", type=Path, help="Recorded server controls; no server mutation")
    parser.add_argument("--endpoint", choices=("completions", "chat"), default="completions")
    parser.add_argument("--tokenizer", help="HF tokenizer ID/path; defaults to --model")
    parser.add_argument("--revision", help="Pinned HF tokenizer/model revision")
    parser.add_argument("--limit", type=int, help="Use only the first N input prompt rows")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--no-return-token-ids", action="store_true", help="Compatibility fallback, explicitly reported")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ("base_url", "model", "prompts", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if min(args.concurrency, args.max_tokens, args.repeats, args.timeout, args.run_timeout) <= 0:
        parser.error("concurrency, max-tokens, repeats, timeout, and run-timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty or new")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "api_key"}
    allowed_controls = {"backend", "execution", "prefix", "max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization", "kv_cache_dtype", "dtype"}
    args.server_settings = json.loads(args.server_config.read_text()) if args.server_config else {}
    if set(args.server_settings) - allowed_controls:
        parser.error("unknown server control fields")
    config.update(args.server_settings)
    config["api_key_supplied"] = bool(args.api_key)
    config["client_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output / "config.json", config)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, revision=args.revision, trust_remote_code=True)
        stop_token = "<|ifm|im_end|>"
        stop_id = tokenizer.convert_tokens_to_ids(stop_token)
        if not isinstance(stop_id, int) or tokenizer.convert_ids_to_tokens(stop_id) != stop_token:
            raise ValueError(f"tokenizer does not expose expected stop token {stop_token}")
        if stop_id in {64030, 64051, 64053}:
            raise ValueError("refusing to use a reasoning closer as a generation stop")
        args.stop_token_ids = [stop_id]
        config.update(stop_token=stop_token, stop_token_ids=args.stop_token_ids)
        atomic_json(args.output / "config.json", config)
        prompts = load_prompts(args.prompts, tokenizer, args.effort, args.limit)
        prepared_path = args.output / "prompt_inputs.jsonl"
        temporary = prepared_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in prompts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, prepared_path)
        summary = asyncio.run(run(args, prompts, tokenizer))
    except Exception as exc:
        atomic_json(args.output / "summary.json", {
            "case_name": args.case_name, "end_reason": "setup_error",
            "error_kind": type(exc).__name__, "error": str(exc), "ended_at": utc_now(),
        })
        print(f"Setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"case_name": summary["case_name"], "end_reason": summary["end_reason"],
                      "request_wall_seconds": summary["request_wall_seconds"], **summary["metrics"]}, indent=2))
    return 0 if summary["end_reason"] == "completed" and not summary["metrics"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
