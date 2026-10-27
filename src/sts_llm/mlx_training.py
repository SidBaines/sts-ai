"""Teacher-feedback policy-gradient training of K2-Horizon with MLX on Apple silicon.

Each iteration takes ``--states-per-iteration`` training states (cycling through
a shuffled order, so every state is seen equally often) and presents each with a
freshly shuffled, letter-labelled menu (prompt version 2). It samples
``--group-size`` answers per state from the current LoRA policy in one batch,
rewards each valid answer with the frozen teacher's probability of the chosen
action (invalid or capped output gets ``--invalid-reward``), subtracts the group
mean, and takes one AdamW step on the generated-token log-probabilities: REINFORCE
with a group-mean baseline. With one update per sampled batch, a PPO ratio would be
exactly one, so no clipping or old-policy scoring pass is needed.

Every completion, including its reasoning, is retained. A greedy evaluation of
fixed menu orders runs on the training states and on a held-out probe every
``--eval-every`` iterations. Run with ``PYTHONPATH=src``; it needs mlx-lm and an
MLX model directory built by ``experiments/2026-10-25-mac-local/mlx_model.py``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any, Sequence

from .dataset import verified_split
from .decision_metrics import (centered_advantages, normalized_teacher_score, reasoning_token_count,
                               teacher_table, teacher_top)
from .generation import K2_ASSISTANT_END_TOKEN_ID, parse_action
from .observations import LABELLED_PROMPT_VERSION, labelled_prompt, legal_indices
from .teacher import teacher_reward

MLX_TRAINING_FORMAT_VERSION = 4
LORA_KEYS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def load_states(path: Path, *, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if verified_split(row) != split:
            raise ValueError(f"row {row['id']} is from the {row['split']} split, not {split}")
        if legal_indices(row["observation"]) != row["legal_indices"]:
            raise ValueError(f"row {row['id']} menu differs from its observation")
        teacher_table(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("state IDs must be unique")
    return rows


def menu_order(rng: random.Random, row: dict) -> list[int]:
    order = list(row["legal_indices"])
    rng.shuffle(order)
    return order


def training_evaluation_rows(train_rows: list[dict], path: Path | None) -> list[dict]:
    rows = load_states(path, split="train") if path else train_rows
    train_by_id = {row["id"]: row for row in train_rows}
    if not rows or any(train_by_id.get(row["id"]) != row for row in rows):
        raise ValueError("train-eval must be a nonempty unchanged subset of the training states")
    return rows


def group_orders(rng: random.Random, row: dict, group_size: int, per_sample: bool) -> list[list[int]]:
    """Menu orders for one group. With one shared order, a label and its action always
    coincide inside the group, so the group baseline cannot tell position from content;
    an order per sample lets it."""
    if per_sample:
        return [menu_order(rng, row) for _ in range(group_size)]
    return [menu_order(rng, row)] * group_size


def evaluation_orders(sets: dict[str, list[dict]], seed: int, count: int) -> dict[str, list[list[int]]]:
    """Keep evaluation menus fixed independently of training randomness.

    The offset preserves the menus produced by format 2 for the same seed.
    """
    rng = random.Random(seed + 2)
    return {row["id"]: [menu_order(rng, row) for _ in range(count)]
            for rows in sets.values() for row in rows}


def score_answer(row: dict, prompt, output_ids: Sequence[int], text: str, finish_reason: str,
                 invalid_reward: float, *, length_penalty_per_256: float = 0.0,
                 free_reasoning_tokens: int = 256) -> dict:
    """Parse one completion against its labelled menu and score it."""
    parsed = parse_action(text, row["legal_indices"], finish_reason=finish_reason, labels=prompt.labels)
    action = parsed.action_index if parsed.valid else None
    reasoning_tokens, closed = reasoning_token_count(output_ids)
    length_cost = (length_penalty_per_256 * max(0, reasoning_tokens - free_reasoning_tokens) / 256
                   if parsed.valid else 0.0)
    label = next((k for k, v in prompt.labels.items() if v == action), None)
    return {"valid": parsed.valid, "error": parsed.error, "action_index": action, "label": label,
            "capped": finish_reason == "length",
            "reward": teacher_reward(row, action, invalid_reward=invalid_reward) - length_cost,
            "length_penalty": length_cost,
            "teacher_probability": teacher_reward(row, action, invalid_reward=0.),
            "teacher_top_agreement": action == teacher_top(row),
            "normalized_teacher_score": normalized_teacher_score(row, action),
            "first_displayed_option": label == "A" if action is not None else None,
            "native_first_option": action == row["legal_indices"][0] if action is not None else None,
            "output_tokens": len(output_ids), "reasoning_tokens": reasoning_tokens,
            "reasoning_closed": closed}


def summarize(records: list[dict], prefix: str = "") -> dict:
    """Means of the decision metrics, overall and per phase."""
    def block(rows):
        if not rows:
            return {}
        valid = [r for r in rows if r["valid"]]
        scores = [r["normalized_teacher_score"] for r in rows if r["normalized_teacher_score"] is not None]
        reasoning = sorted(r["reasoning_tokens"] for r in rows)
        return {"decisions": len(rows), "valid_fraction": len(valid) / len(rows),
                "capped_fraction": sum(r["capped"] for r in rows) / len(rows),
                "mean_teacher_probability": statistics.fmean(r["teacher_probability"] for r in rows),
                "teacher_top_agreement": statistics.fmean(r["teacher_top_agreement"] for r in rows),
                "mean_normalized_teacher_score": statistics.fmean(scores) if scores else None,
                "first_displayed_option_rate": (statistics.fmean(r["first_displayed_option"] for r in valid)
                                                if valid else None),
                "median_reasoning_tokens": statistics.median(reasoning),
                "p95_reasoning_tokens": reasoning[min(len(reasoning) - 1, math.ceil(0.95 * len(reasoning)) - 1)],
                "empty_reasoning_fraction": sum(r["reasoning_tokens"] == 0 for r in rows) / len(rows),
                "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in rows)}
    result = {f"{prefix}all": block(records)}
    for phase in ("out_of_combat", "combat"):
        result[f"{prefix}{phase}"] = block([r for r in records if r["phase"] == phase])
    return result


def group_statistics(groups: list[list[dict]]) -> dict:
    rewards = [[r["reward"] for r in group] for group in groups]
    return {"zero_variance_group_fraction": statistics.fmean(max(g) - min(g) <= 1e-12 for g in rewards),
            "mean_group_reward_range": statistics.fmean(max(g) - min(g) for g in rewards),
            "mean_distinct_actions_per_group": statistics.fmean(len({r["action_index"] for r in group})
                                                                for group in groups),
            "mean_reward": statistics.fmean(x for g in rewards for x in g)}


class MLXPolicy:
    """The MLX model with LoRA adapters, its tokenizer, generation and scoring."""

    def __init__(self, args: Any):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from transformers import AutoTokenizer
        self.mx, self.args = mx, args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_repo, revision=args.revision, trust_remote_code=True)
        self.model, _ = load(str(args.mlx_model))
        self.model.freeze()
        # Generation seeds are set later; they cannot seed the random LoRA
        # factors already created here. Seed immediately before initialization.
        mx.random.seed(args.seed)
        linear_to_lora_layers(self.model, len(self.model.layers), {
            "rank": args.lora_rank, "scale": args.lora_alpha / args.lora_rank, "dropout": 0.0, "keys": list(LORA_KEYS)})
        if args.adapter:
            self.model.load_weights(str(args.adapter), strict=False)
        if args.grad_checkpoint:
            grad_checkpoint(self.model.layers[0])
        self.stops = [[K2_ASSISTANT_END_TOKEN_ID], [self.tokenizer.eos_token_id]]
        self.reference_weights = None
        if getattr(args, "kl_reference", "foundation") == "initial":
            current = dict(self.trainable_items())
            if getattr(args, "reference_adapter", None):
                self.model.load_weights(str(args.reference_adapter), strict=False)
            self.reference_weights = {k: mx.array(v) for k, v in self.trainable_items()}
            mx.eval(list(self.reference_weights.values()))
            self.model.load_weights(list(current.items()), strict=False)

    def prompt_ids(self, messages) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=False, reasoning_effort=self.args.effort)
        return list(ids)

    def generate(self, prompts: list[list[int]], *, temperature: float, seed: int,
                 concurrency: int | None = None, progress_callback=None) -> list[tuple[list[int], str]]:
        """One completion per prompt. A batch of identical prompts is about twice as fast
        per decision as a batch mixing prompt lengths."""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler
        self.model.eval()
        self.mx.random.seed(seed)
        width = min(len(prompts), concurrency or len(prompts))
        generator = BatchGenerator(self.model, max_tokens=self.args.max_new_tokens, stop_tokens=self.stops,
                                   sampler=make_sampler(temp=temperature, top_p=1.0),
                                   completion_batch_size=width, prefill_batch_size=min(8, width))
        uids = generator.insert(prompts, [self.args.max_new_tokens] * len(prompts))
        tokens, finished = {uid: [] for uid in uids}, {}
        started = last_report = time.monotonic()
        generated_tokens = 0

        def report(force=False):
            nonlocal last_report
            if progress_callback is None:
                return
            now = time.monotonic()
            if force or now - last_report >= 30:
                progress_callback({"completed": len(finished), "total": len(uids),
                                   "generated_tokens": generated_tokens,
                                   "elapsed_seconds": now - started})
                last_report = now

        report(force=True)
        while responses := generator.next_generated():
            for response in responses:
                tokens[response.uid].append(int(response.token))
                generated_tokens += 1
                if response.finish_reason is not None:
                    finished[response.uid] = response.finish_reason
            report()
        generator.close()
        report(force=True)
        return [(tokens[uid], finished[uid]) for uid in uids]

    def token_logps(self, prompt: list[int], output: list[int]):
        """Temperature-scaled log-probabilities of the generated tokens only."""
        mx = self.mx
        ids = mx.array([prompt + output])
        start = len(prompt) - 1
        hidden = self.model.model(ids)[:, start:-1]
        logits = self.model.lm_head(hidden).astype(mx.float32) / self.args.temperature
        logps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.take_along_axis(logps, ids[:, start + 1:, None], axis=-1)[0, :, 0]

    @contextmanager
    def adapters_disabled(self):
        from mlx_lm.tuner.lora import LoRALinear
        layers = [m for _, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        scales = [layer.scale for layer in layers]
        try:
            for layer in layers:
                layer.scale = 0.0
            yield
        finally:
            for layer, scale in zip(layers, scales):
                layer.scale = scale

    @contextmanager
    def reference_policy(self):
        """Score against either the foundation or the frozen starting adapter."""
        if self.reference_weights is None:
            with self.adapters_disabled():
                yield
            return
        current = dict(self.trainable_items())
        try:
            self.model.load_weights(list(self.reference_weights.items()), strict=False)
            yield
        finally:
            self.model.load_weights(list(current.items()), strict=False)

    def trainable_items(self):
        from mlx.utils import tree_flatten
        return tree_flatten(self.model.trainable_parameters())

    def save_adapter(self, path: Path, metadata: dict) -> None:
        self.mx.save_safetensors(str(path), dict(self.trainable_items()), metadata={"format": "mlx"})
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def response_loss(policy: MLXPolicy, item: dict, *, advantage: float, ref_logps, denominator: float,
                  kl_beta: float, token_scope: str = "all", with_metrics: bool = False):
    mx = policy.mx
    new = policy.token_logps(item["prompt_ids"], item["output_token_ids"])
    start = 0
    if token_scope == "answer":
        # Include the reasoning closer, final answer and end token. A capped
        # response with no closer has no answer gradient; KL still anchors it.
        start, _ = reasoning_token_count(item["output_token_ids"])
    policy_loss = -advantage * new[start:].sum() / denominator
    kl_loss = mx.array(0.0)
    kl_mean = mx.array(0.0)
    if kl_beta > 0:
        delta = ref_logps - new
        divergence = mx.exp(delta) - delta - 1
        kl_mean = divergence.mean()
        kl_loss = kl_beta * divergence.sum() / denominator
    loss = policy_loss + kl_loss
    return (loss, {"policy_loss": policy_loss, "kl_loss": kl_loss, "sampled_kl_per_token": kl_mean}) if with_metrics else loss


def update(policy: MLXPolicy, optimizer, items: list[dict], args: Any) -> dict:
    """One AdamW step on the mean per-response loss, accumulated one response at a time."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    policy.model.train()
    active = [item for item in items if item["advantage"] != 0.0 or args.kl_beta > 0]
    if not active:
        return {"optimizer_step": False, "responses_with_gradient": 0}
    total, loss_sum = None, 0.0
    components = {"policy_loss": 0.0, "kl_loss": 0.0, "sampled_kl_per_token": 0.0}
    for item in active:
        ref = mx.array(item["ref_logps"]) if args.kl_beta > 0 else None
        denominator = (args.loss_token_budget if args.loss_normalization == "fixed_token_budget"
                       else len(item["output_token_ids"]))
        grad_fn = nn.value_and_grad(policy.model, lambda: response_loss(
            policy, item, advantage=item["advantage"], ref_logps=ref,
            denominator=denominator, kl_beta=args.kl_beta,
            token_scope=getattr(args, "policy_token_scope", "all"), with_metrics=True))
        (loss, metrics), grads = grad_fn()
        # Weight by 1/N over all responses; zero-advantage ones contribute nothing.
        grads = tree_map(lambda g: g / len(items), grads)
        total = grads if total is None else tree_map(mx.add, total, grads)
        mx.eval(total, loss, metrics)
        loss_sum += loss.item() / len(items)
        for key in components:
            components[key] += metrics[key].item() / len(items)
    total, norm = optim.clip_grad_norm(total, args.max_grad_norm)
    norm = norm.item()
    if not math.isfinite(norm) or not math.isfinite(loss_sum):
        raise FloatingPointError("nonfinite loss or gradient; optimizer step withheld")
    optimizer.update(policy.model, total)
    mx.eval(policy.model.trainable_parameters(), optimizer.state)
    return {"optimizer_step": True, "loss": loss_sum, "gradient_norm_before_clip": norm,
            "responses_with_gradient": len(active), **components}


def evaluate(policy: MLXPolicy, sets: dict[str, list[dict]], orders: dict[str, list[list[int]]],
             args: Any, iteration: int, output: Path) -> dict:
    """Greedy answers for fixed menu orders, all states in one bounded-width batch."""
    jobs = [(name, row, labelled_prompt(row["observation"], order))
            for name, rows in sets.items() for row in rows for order in orders[row["id"]]]
    def progress(record):
        # Atomic replacement lets the watcher/dashboard read while generation runs.
        path = output / "evaluation_progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"iteration": iteration, "updated_at": time.time(),
                                         **record}) + "\n")
        temporary.replace(path)

    results = policy.generate([policy.prompt_ids(prompt.messages) for _, _, prompt in jobs],
                              temperature=0.0, seed=args.eval_seed, concurrency=args.eval_concurrency,
                              progress_callback=progress)
    records = []
    for (name, row, prompt), (output_ids, reason) in zip(jobs, results):
        text = policy.tokenizer.decode(output_ids, skip_special_tokens=False)
        records.append({"set": name, "id": row["id"], "phase": row["phase"], "order": prompt.order,
                        "raw_text": text, "output_token_ids": output_ids, "finish_reason": reason,
                        **score_answer(row, prompt, output_ids, text, reason, args.invalid_reward,
                                       length_penalty_per_256=args.length_penalty_per_256,
                                       free_reasoning_tokens=args.free_reasoning_tokens)})
    with (output / "evaluations" / f"iter_{iteration:04d}.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {"iteration": iteration}
    for name in sets:
        summary.update(summarize([r for r in records if r["set"] == name], prefix=f"{name}/"))
    return summary


def run_record(args: Any, policy: MLXPolicy, train_rows: list[dict], probe_rows: list[dict]) -> dict:
    import mlx.core as mx
    import mlx_lm
    source = Path(__file__).parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=source,
                                capture_output=True, text=True).stdout.strip())
    config_path = Path(args.mlx_model) / "config.json"
    teachers = sorted({row["teacher"].get("checkpoint_sha256", "") for row in train_rows + probe_rows})
    return {"format_version": MLX_TRAINING_FORMAT_VERSION, "prompt_version": LABELLED_PROMPT_VERSION,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_commit": commit, "source_uncommitted_changes": dirty,
            "source_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(source.glob("*.py"))},
            "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
            "device": {k: v for k, v in getattr(mx, "device_info", mx.metal.device_info)().items()
                       if isinstance(v, (str, int, float))},
            "mlx_model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "states_sha256": {"train": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                              "train_eval": hashlib.sha256(Path(args.train_eval).read_bytes()).hexdigest()
                              if args.train_eval else None,
                              "probe": hashlib.sha256(Path(args.probe).read_bytes()).hexdigest()
                              if args.probe else None},
            "parent_adapter_sha256": hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest()
            if args.adapter else None,
            "reference_adapter_sha256": hashlib.sha256(Path(args.reference_adapter).read_bytes()).hexdigest()
            if args.reference_adapter else None,
            "teacher_checkpoint_sha256": teachers,
            "trainable_parameters": sum(v.size for _, v in policy.trainable_items()),
            "train_states": len(train_rows), "probe_states": len(probe_rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--model-repo", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--revision", default="7c75f2f49e74ac97c1852e62b4c860091d6a00ea")
    parser.add_argument("--train", type=Path, required=True, help="JSONL of training-split states with teacher labels")
    parser.add_argument("--probe", type=Path, help="JSONL of validation-split states for the held-out probe")
    parser.add_argument("--train-eval", type=Path,
                        help="optional smaller training-state panel for frequent evaluations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="warm-start LoRA weights (optimizer and RNG reset)")
    parser.add_argument("--reference-adapter", type=Path, help="optional fixed adapter for the initial-policy KL reference")
    parser.add_argument("--label", help="human-readable experiment label, recorded in config")
    parser.add_argument("--eval-only", action="store_true", help="evaluate the supplied adapter without an optimizer update")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-run-seconds", type=float, default=0,
                        help="0 disables the wall-time bound; stop between iterations and save final weights")
    parser.add_argument("--states-per-iteration", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--menu-order", choices=("per_sample", "per_group"), default="per_sample",
                        help="a fresh menu order for every sample, or one shared by the group (format 1 behaviour)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--kl-reference", choices=("foundation", "initial"), default="foundation")
    parser.add_argument("--policy-token-scope", choices=("all", "answer"), default="all")
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--length-penalty-per-256", type=float, default=0.0)
    parser.add_argument("--free-reasoning-tokens", type=int, default=256)
    parser.add_argument("--loss-normalization", choices=("fixed_token_budget", "response_mean"),
                        default="fixed_token_budget")
    parser.add_argument("--loss-token-budget", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=10, help="0 disables evaluation")
    parser.add_argument("--eval-orders", type=int, default=2, help="fixed menu orders per evaluated state")
    parser.add_argument("--eval-concurrency", type=int, default=16, help="evaluation answers decoded at once")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="evaluation menu seed; defaults to --seed; hold fixed across training repeats")
    args = parser.parse_args(argv)
    if args.eval_seed is None:
        args.eval_seed = args.seed
    if min(args.iterations, args.states_per_iteration, args.group_size, args.max_new_tokens,
           args.loss_token_budget, args.lora_rank, args.eval_orders, args.eval_concurrency) < 1 or args.group_size < 2:
        parser.error("sizes must be positive and group size at least 2")
    if not (args.temperature > 0 and args.learning_rate > 0 and args.max_grad_norm > 0 and args.kl_beta >= 0
            and math.isfinite(args.invalid_reward) and math.isfinite(args.max_run_seconds)
            and args.max_run_seconds >= 0 and math.isfinite(args.length_penalty_per_256)
            and args.length_penalty_per_256 >= 0 and args.free_reasoning_tokens >= 0):
        parser.error("temperature, learning rate and gradient norm must be positive; KL nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("generations", "evaluations", "adapters"):
        (args.output / name).mkdir()
    stop_file = args.output / "STOP"

    import mlx.core as mx
    import mlx.optimizers as optim
    train_rows = load_states(args.train, split="train")
    probe_rows = load_states(args.probe, split="validation") if args.probe else []
    train_eval_rows = training_evaluation_rows(train_rows, args.train_eval)
    policy = MLXPolicy(args)
    record = run_record(args, policy, train_rows, probe_rows)
    (args.output / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    rng = random.Random(args.seed)
    order_rng = random.Random(args.seed + 1)
    eval_sets = {"train": train_eval_rows, **({"probe": probe_rows} if probe_rows else {})}
    eval_orders = evaluation_orders(eval_sets, args.eval_seed, args.eval_orders)
    adapter_metadata = {"format_version": MLX_TRAINING_FORMAT_VERSION, "model_repo": args.model_repo,
                        "revision": args.revision, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                        "lora_keys": list(LORA_KEYS), "effort": args.effort, "prompt_version": LABELLED_PROMPT_VERSION}
    log = (args.output / "iterations.jsonl").open("a")
    queue: list[dict] = []
    run_started = time.monotonic()
    last_iteration, last_evaluation, stop_reason = 0, None, "iterations_complete"

    def evaluation(iteration):
        nonlocal last_evaluation
        started = time.perf_counter()
        summary = evaluate(policy, eval_sets, eval_orders, args, iteration, args.output)
        summary["seconds"] = time.perf_counter() - started
        with (args.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        print(json.dumps({"event": "evaluation", **{k: v for k, v in summary.items() if k.endswith("/all")},
                          "iteration": iteration, "seconds": summary["seconds"]}), flush=True)
        last_evaluation = iteration

    if args.eval_every or args.eval_only:
        evaluation(0)
    if args.eval_only:
        log.close()
        (args.output / "run_summary.json").write_text(json.dumps({
            "last_iteration": 0, "stop_reason": "evaluation_only",
            "elapsed_seconds": time.monotonic() - run_started}, indent=2) + "\n")
        return 0
    for iteration in range(1, args.iterations + 1):
        timed_out = args.max_run_seconds > 0 and time.monotonic() - run_started >= args.max_run_seconds
        if stop_file.exists() or timed_out:
            stop_reason = "time_limit" if timed_out else "stop_file"
            print(json.dumps({"event": "stopped", "before_iteration": iteration, "reason": stop_reason}), flush=True)
            break
        began = time.perf_counter()
        if len(queue) < args.states_per_iteration:
            fresh = list(train_rows)
            rng.shuffle(fresh)
            queue.extend(fresh)
        states, queue = queue[:args.states_per_iteration], queue[args.states_per_iteration:]
        groups, items = [], []
        for position, row in enumerate(states):
            prompts = [labelled_prompt(row["observation"], order) for order in
                       group_orders(order_rng, row, args.group_size, args.menu_order == "per_sample")]
            prompt_ids = [policy.prompt_ids(prompt.messages) for prompt in prompts]
            results = policy.generate(prompt_ids, temperature=args.temperature,
                                      seed=args.seed + iteration * 1000 + position)
            group = []
            for sample, (prompt, ids, (output_ids, reason)) in enumerate(zip(prompts, prompt_ids, results)):
                text = policy.tokenizer.decode(output_ids, skip_special_tokens=False)
                group.append({"iteration": iteration, "id": row["id"], "phase": row["phase"], "sample": sample,
                              "order": prompt.order, "prompt_ids": ids, "output_token_ids": output_ids,
                              "raw_text": text, "finish_reason": reason,
                              **score_answer(row, prompt, output_ids, text, reason, args.invalid_reward,
                                             length_penalty_per_256=args.length_penalty_per_256,
                                             free_reasoning_tokens=args.free_reasoning_tokens)})
            for item, advantage in zip(group, centered_advantages([r["reward"] for r in group])):
                item["advantage"] = advantage
            groups.append(group)
            items.extend(group)
        generation_seconds = time.perf_counter() - began
        scoring_started = time.perf_counter()
        if args.kl_beta > 0:
            with policy.reference_policy():
                for item in items:
                    item["ref_logps"] = policy.token_logps(item["prompt_ids"], item["output_token_ids"]).tolist()
        scoring_seconds = time.perf_counter() - scoring_started
        update_started = time.perf_counter()
        step = update(policy, optimizer, items, args)
        update_seconds = time.perf_counter() - update_started
        with (args.output / "generations" / f"iter_{iteration:04d}.jsonl").open("x") as stream:
            for item in items:
                stream.write(json.dumps({k: v for k, v in item.items() if k not in ("prompt_ids", "ref_logps")}) + "\n")
        summary = {"iteration": iteration, **step, **group_statistics(groups), **summarize(items),
                   "generated_tokens": sum(len(i["output_token_ids"]) for i in items),
                   "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
                   "update_seconds": update_seconds, "iteration_seconds": time.perf_counter() - began,
                   "peak_memory_gb": mx.get_peak_memory() / 1e9}
        log.write(json.dumps(summary) + "\n")
        log.flush()
        last_iteration = iteration
        print(json.dumps({k: summary[k] for k in ("iteration", "mean_reward", "zero_variance_group_fraction",
                                                  "iteration_seconds")} | {"all": summary["all"]}), flush=True)
        if iteration % args.save_every == 0 or iteration == args.iterations:
            policy.save_adapter(args.output / "adapters" / f"iter_{iteration:04d}.safetensors",
                                {**adapter_metadata, "iteration": iteration})
        if args.eval_every and (iteration % args.eval_every == 0 or iteration == args.iterations):
            evaluation(iteration)
    log.close()
    if last_iteration:
        final = args.output / "adapters" / f"iter_{last_iteration:04d}.safetensors"
        if not final.exists():
            policy.save_adapter(final, {**adapter_metadata, "iteration": last_iteration})
        if args.eval_every and last_evaluation != last_iteration:
            evaluation(last_iteration)
    (args.output / "run_summary.json").write_text(json.dumps({
        "last_iteration": last_iteration, "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - run_started}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
