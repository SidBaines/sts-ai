"""Compare sampled MLX training groups with and without a shared within-group KV prefix."""

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from itertools import takewhile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.worktree / "src"))
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache
    from sts_llm.mlx_training import MLXPolicy, labelled_prompt, score_answer

    runs = args.worktree / "data/llm_runs"
    run = runs / "learning_trials_20261027/lr_small_step"
    settings = SimpleNamespace(
        **json.loads((run / "config.json").read_text())["settings"]
    )
    settings.mlx_model = args.worktree / "data/models/k2-horizon-0.9b-mlx"
    settings.adapter = (
        runs / "stage1_screen_ps_20261026/lr_3e-5/adapters/iter_0048.safetensors"
    )
    settings.reference_adapter = settings.adapter
    policy = MLXPolicy(settings)
    states = {
        r["id"]: r
        for r in map(
            json.loads, (runs / "broader_states_selected_20261026/train.jsonl").open()
        )
    }
    groups = {}
    for r in map(json.loads, (run / "generations/iter_0001.jsonl").open()):
        groups.setdefault(r["id"], []).append(r)
    selected = []
    for phase in ("combat", "out_of_combat"):
        rows = next(x for x in groups.values() if x[0]["phase"] == phase)
        state = states[rows[0]["id"]]
        menu = [labelled_prompt(state["observation"], r["order"]) for r in rows]
        selected.append((state, menu, [policy.prompt_ids(q.messages) for q in menu]))
    raw_insert = BatchGenerator.insert
    prefix_sizes = []

    def shared_insert(generator, prompts, max_tokens=None):
        length = sum(
            1 for _ in takewhile(lambda col: len(set(col)) == 1, zip(*prompts))
        )
        length = min(length, min(map(len, prompts)) - 1)
        prefix_sizes.append(length)
        if not length:
            return raw_insert(generator, prompts, max_tokens)
        cache = make_prompt_cache(policy.model)
        policy.model.model(mx.array([prompts[0][:length]]), cache=cache)
        mx.eval([c.state for c in cache])
        # Each request gets its own cache object at the same immutable prefix.
        copies = [
            [type(c).from_state(c.state, c.meta_state) for c in cache] for _ in prompts
        ]
        return raw_insert(
            generator,
            [x[length:] for x in prompts],
            max_tokens,
            caches=copies,
            all_tokens=[x[:length] for x in prompts],
        )

    summary = []
    reference = []
    for arm in ("original", "shared", "shared", "original"):
        prefix_sizes.clear()
        outputs = []
        actions = []
        mx.synchronize()
        mx.reset_peak_memory()
        t = time.perf_counter()
        ctx = (
            patch.object(BatchGenerator, "insert", shared_insert)
            if arm == "shared"
            else nullcontext()
        )
        with ctx:
            for index, (row, menus, prompts) in enumerate(selected):
                result = policy.generate(
                    prompts,
                    temperature=settings.temperature,
                    seed=settings.seed + 1000 + index,
                )
                outputs.extend(result)
                for menu, (ids, reason) in zip(menus, result):
                    text = policy.tokenizer.decode(ids, skip_special_tokens=False)
                    answer = score_answer(
                        row, menu, ids, text, reason, settings.invalid_reward
                    )
                    actions.append(answer["action_index"])
        mx.synchronize()
        elapsed = time.perf_counter() - t
        if not reference:
            reference = list(zip(outputs, actions))
        rec = dict(
            arm=arm,
            seconds=elapsed,
            peak_gb=mx.get_peak_memory() / 1e9,
            cap=settings.max_new_tokens,
            tokens=sum(len(x[0]) for x in outputs),
            prefix_lengths=list(prefix_sizes),
            exact_sequences=sum(x == y for x, (y, _) in zip(outputs, reference)),
            same_action_or_invalid=sum(x == y for x, (_, y) in zip(actions, reference)),
            valid=sum(x is not None for x in actions),
            samples=len(actions),
            hash=hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
        )
        summary.append(rec)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
