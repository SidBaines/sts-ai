#!/usr/bin/env python
"""September diagnostic: matched prompts and finite-menu constrained decoding.

Experimental only: does not change the production agent or frozen interfaces.
Constraints permit tokenizations of canonical legal JSON strings, plus EOS only
after a complete candidate. This defines a different decoding policy; validity
under constraints is by construction and is not itself evidence of competence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sts_ai.agents import MlxQwenJsonAgent, parse_json_action
from sts_ai.prompting import render_action_prompt, retry_instruction
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.schemas import LegalAction


class CandidateTrie:
    def __init__(self, candidates: list[list[int]], eos: list[int]):
        self.next: dict[tuple[int, ...], set[int]] = defaultdict(set)
        self.terminals = {tuple(candidate) for candidate in candidates}
        self.eos = set(eos)
        for candidate in candidates:
            if not candidate:
                raise ValueError("empty candidate")
            for i, token in enumerate(candidate):
                self.next[tuple(candidate[:i])].add(token)
            self.next[tuple(candidate)].update(eos)

    def allowed(self, prefix: list[int]) -> list[int]:
        # MLX may compute one lookahead step after emitting a stop token.
        trimmed = list(prefix)
        while trimmed and trimmed[-1] in self.eos:
            trimmed.pop()
        if len(trimmed) < len(prefix) and tuple(trimmed) in self.terminals:
            return sorted(self.eos)
        result = sorted(self.next.get(tuple(prefix), ()))
        if not result:
            raise ValueError(f"decoder left legal candidate trie: {prefix}")
        return result


class TrieProcessor:
    def __init__(self, trie: CandidateTrie):
        self.trie = trie
        self.offset: int | None = None

    def __call__(self, tokens: Any, logits: Any) -> Any:
        import mlx.core as mx

        if self.offset is None:
            self.offset = len(tokens)
        prefix = tokens[self.offset :].tolist()
        allowed = self.trie.allowed(prefix)
        mask = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        mask[..., mx.array(allowed)] = 0
        return logits + mask


def canonical_candidates(actions: list[LegalAction], contract: str) -> list[str]:
    if contract == "action_text":
        objects = [{"action": a.description} for a in actions]
    elif contract == "action_only":
        objects = [{"action_index": a.index} for a in actions]
    else:
        raise ValueError("constraints require action_text or action_only")
    return [json.dumps(obj, separators=(",", ":")) for obj in objects]


class DiagnosticAgent(MlxQwenJsonAgent):
    name = "astra_diagnostic"

    def __init__(self, *, constrained: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.constrained = constrained
        self.batch_number = 0

    @property
    def config(self) -> dict[str, Any]:
        return {**super().config, "experimental_decoder": "canonical_trie_v1" if self.constrained else "free_v1"}

    def choose_action(self, state_text, legal_actions, prompt_override=None, phase=None):
        if prompt_override is not None:
            raise ValueError("diagnostic agent has no prompt overrides")
        return self.choose_actions_batch([(state_text, legal_actions)])[0]

    def choose_actions_batch(self, items, retry_flags=None, phases=None):
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator

        retry_flags = retry_flags or [False] * len(items)
        prompts = []
        processors = []
        caps = []
        for (state_text, actions), retry in zip(items, retry_flags):
            prompt = render_action_prompt(state_text, actions, self.framing, output_contract=self.output_contract)
            if retry:
                prompt += retry_instruction(self.output_contract)
            prompts.append(self.tokenizer.encode(self._apply_chat_template(prompt)))
            if self.constrained:
                candidates = [self.tokenizer.encode(s, add_special_tokens=False)
                              for s in canonical_candidates(actions, self.output_contract)]
                processors.append([TrieProcessor(CandidateTrie(candidates, list(self.tokenizer.eos_token_ids)))])
                caps.append(max(map(len, candidates)) + 1)
            else:
                processors.append([])
                caps.append(self.max_tokens)

        started = time.monotonic()
        generator = BatchGenerator(self.model, stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
                                   sampler=self._sampler, completion_batch_size=len(items),
                                   prefill_batch_size=min(4, len(items)), prefill_step_size=512)
        uids = generator.insert(prompts, max_tokens=caps, logits_processors=processors)
        outputs: dict[int, list[int]] = {uid: [] for uid in uids}
        finishes = {}
        try:
            while responses := generator.next_generated():
                for response in responses:
                    if response.finish_reason != "stop":
                        outputs[response.uid].append(int(response.token))
                    if response.finish_reason is not None:
                        finishes[response.uid] = response.finish_reason
        finally:
            generator.close()
        decisions = []
        for uid, prompt, (_, actions), cap in zip(uids, prompts, items, caps):
            text = self.tokenizer.decode(outputs[uid])
            decision = parse_json_action(text, actions, completion_tokens=len(outputs[uid]),
                                         max_tokens=cap, output_contract=self.output_contract)
            decision.prompt_tokens = len(prompt)
            decision.completion_tokens = len(outputs[uid])
            decision.latency_s = (time.monotonic() - started) / len(items)
            decision.metadata.update({"experimental_decoder": self.config["experimental_decoder"],
                                      "finish_reason": finishes.get(uid)})
            if self.constrained and (not decision.valid or text not in canonical_candidates(actions, self.output_contract)):
                raise RuntimeError(f"constraint failure: {text!r}")
            decisions.append(decision)
        self.batch_number += 1
        print(json.dumps({"batch": self.batch_number, "n": len(items),
                          "seconds": round(time.monotonic() - started, 2),
                          "invalid": sum(not d.valid for d in decisions)}), flush=True)
        mx.clear_cache()
        return decisions


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def actions_for(row: dict[str, Any]) -> list[LegalAction]:
    return [LegalAction(**a) for a in row["legal_actions"]]


def prepare(args: argparse.Namespace) -> None:
    pools = defaultdict(list)
    paths = sorted(args.source.rglob("seed_*_r0.jsonl"))
    for path in paths:
        if not path.with_suffix(".meta.json").exists():
            continue
        for row in load_rows(path):
            if row.get("phase") != "out_of_combat" or len(row["legal_actions"]) < 2:
                continue
            screen = row["state"]["screen_state"].split(".")[-1]
            if screen == "EVENT_SCREEN" and row["state"]["floor"] == 0:
                screen = "NEOW"
            row["probe_screen"] = screen
            row["source_trace"] = str(path)
            pools[screen].append(row)
    rng = random.Random(905)
    selected = []
    for screen, rows in sorted(pools.items()):
        rng.shuffle(rows)
        selected.extend(rows[:args.per_screen])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for i, row in enumerate(selected):
            row["probe_id"] = i
            handle.write(json.dumps(row) + "\n")
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "n": len(selected), "source": str(args.source),
        "available_by_screen": {k: len(v) for k, v in pools.items()},
        "selected_by_screen": dict(Counter(r["probe_screen"] for r in selected)),
        "sha256": file_sha256(args.out), "selection_seed": 905,
    }, indent=2))
    print(args.out, len(selected))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["probe_screen"]].append(row)

    def summary(items):
        matches = Counter(r["decision"]["metadata"].get("semantic_match", "index_or_other") for r in items if r["decision"]["valid"])
        return {"n": len(items), "valid": sum(r["decision"]["valid"] for r in items),
                "valid_rate": sum(r["decision"]["valid"] for r in items) / len(items),
                "completion_tokens": sum(r["decision"]["completion_tokens"] for r in items),
                "matches": dict(matches)}
    return {"overall": summary(rows), "by_screen": {k: summary(v) for k, v in grouped.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "probe", "rollout"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-screen", type=int, default=16)
    parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    parser.add_argument("--adapter")
    parser.add_argument("--contract", choices=["action_text", "action_only", "reasoning_action"], default="action_text")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seeds", default=",".join(map(str, range(1000, 1016))))
    parser.add_argument("--max-decisions", type=int, default=250)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {**vars(args), "adapter_identity": adapter_provenance(args.adapter),
                  "source_sha256": file_sha256(args.source) if args.source else None,
                  "script_sha256": file_sha256(__file__), "started": time.time()}
    (args.out / "config.json").write_text(json.dumps(provenance, default=str, indent=2))
    agent = DiagnosticAgent(model_id=args.model, adapter_path=args.adapter, constrained=args.constrained,
                            output_contract=args.contract, enable_thinking=False, temperature=args.temperature,
                            max_tokens=args.max_tokens, max_retries=args.max_retries)
    if args.mode == "probe":
        rows = load_rows(args.source)
        results = []
        with (args.out / "predictions.jsonl").open("x") as handle:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                agent.reseed(905 + start)
                decisions = agent.choose_actions_batch([(r["state_text"], actions_for(r)) for r in batch])
                for row, decision in zip(batch, decisions):
                    result = {"probe_id": row["probe_id"], "probe_screen": row["probe_screen"],
                              "world_seed": row["world_seed"], "decision_index": row["decision_index"],
                              "decision": asdict(decision)}
                    results.append(result)
                    handle.write(json.dumps(result) + "\n")
                handle.flush()
        report = summarize(results)
    else:
        from sts_ai.lightspeed import LightspeedHybridEnv
        from sts_ai.parallel_rollout import run_parallel_rollouts

        seeds = [int(s) for s in args.seeds.split(",")]
        results = run_parallel_rollouts([(s, 0) for s in seeds],
            lambda seed: LightspeedHybridEnv(seed, battle_simulations=50, combat_control="search", combat_observation="combat_public_v2"),
            agent, output_for=lambda seed, ri: args.out / f"seed_{seed}_r{ri}.jsonl",
            batch_size=args.batch_size, max_decisions=args.max_decisions, max_retries=args.max_retries,
            run_meta={"experiment": "astra_5sep", "diagnostic": True})
        report = {"n": len(results), "mean_floor": sum(r.terminal_state["floor"] for r in results) / len(results),
                  "stopped_reasons": dict(Counter(r.stopped_reason for r in results))}
    report["wall_seconds"] = time.time() - provenance["started"]
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    agent.sleep()


if __name__ == "__main__":
    main()
