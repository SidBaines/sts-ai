"""Minimal synchronous full-game outcome RL over the clean isolated environment.

The parent owns the HF/PEFT model; sts_env owns native workers and replayable
game traces. This module is an implementation, not evidence of learned strength.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
from collections import Counter

from .generation import parse_action
from .observations import PROMPT_VERSION, legal_indices, prompt
from .training import (TRAINING_FORMAT_VERSION, Completion, atomic_json, freeze_cohort_logps, load_model,
                       normalized_advantages, optimizer_step, trainable_hash)

FORMAT_VERSION = 1


class StopRequested(RuntimeError):
    pass


class InvalidDecision(RuntimeError):
    pass


def validate_world_seeds(seeds):
    """Match the clean environment's signed-63-bit, nonnegative seed contract."""
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("seeds file must contain distinct integer worlds in [0, 2**63-1]")
    return seeds


def validate_full_game_scope(max_act):
    if type(max_act) is not int or max_act != 3:
        raise ValueError("full-game training and evaluation require --max-act 3; bound diagnostics with max-decisions")


@dataclass
class Episode:
    world_seed: int
    policy_seed: int
    status: str
    outcome: str | None
    reward: float | None
    summary: dict
    decisions: list[Completion] = field(default_factory=list)
    forced_actions: int = 0
    advantage: float = 0.

    def record(self):
        return {k: v for k, v in asdict(self).items() if k != "decisions"} | {
            "model_decisions": len(self.decisions),
            "generated_tokens": sum(len(d.output_ids) for d in self.decisions)}


def terminal_result(summary, *, win_reward, loss_reward):
    """Recognize genuine game endings; faults and every truncation stay unscored."""
    if summary.get("error"):
        return "failed", None, None
    if summary.get("truncated") or summary.get("status") != "terminated" or not summary.get("terminated"):
        return "incomplete", None, None
    outcome = str(summary.get("info", {}).get("outcome", "")).rsplit(".", 1)[-1]
    if outcome == "PLAYER_VICTORY":
        return "complete", outcome, float(win_reward)
    if outcome == "PLAYER_LOSS":
        return "complete", outcome, float(loss_reward)
    return "unknown_terminal", outcome, None


class HFPolicy:
    def __init__(self, model, tokenizer, stop_id, args):
        self.model, self.tokenizer, self.stop_id, self.args = model, tokenizer, stop_id, args

    def __call__(self, messages, *, seed, prompt_id, phase):
        import torch
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, reasoning_effort=self.args.effort)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError("Expected flat prompt token IDs")
        if len(ids) + self.args.max_new_tokens > self.args.max_sequence_length:
            raise InvalidDecision("prompt_plus_completion_budget_exceeds_context")
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        self.model.eval()
        torch.manual_seed(seed)
        sampling = {"do_sample": self.args.temperature > 0}
        if self.args.temperature > 0:
            sampling.update(temperature=self.args.temperature, top_p=1., top_k=0)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                use_cache=True, **sampling,
                max_new_tokens=self.args.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.stop_id, self.tokenizer.eos_token_id],
                repetition_penalty=1., renormalize_logits=False)
        tokens = outputs[0, len(ids):].tolist()
        ends = [i for i, token in enumerate(tokens) if token in {self.stop_id, self.tokenizer.eos_token_id}]
        if ends:
            tokens = tokens[:ends[0] + 1]
        return Completion(prompt_id, phase, ids, tokens, reward=0.,
            raw_text=self.tokenizer.decode(tokens, skip_special_tokens=False), truncated=not bool(ends))


class CollectingAgent:
    def __init__(self, policy, ledger, stop_file=None, episode_id="episode"):
        self.policy, self.ledger, self.stop_file = policy, ledger, stop_file
        self.episode_id = episode_id
        self.decisions, self.forced_actions, self.step = [], 0, 0

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.decisions, self.forced_actions, self.step = [], 0, 0

    def choose_action(self, observation):
        if self.stop_file is not None and self.stop_file.exists():
            raise StopRequested("STOP requested between game decisions")
        self.step += 1
        indices = legal_indices(observation)
        if len(indices) == 1:
            self.forced_actions += 1
            self._record({"event": "forced_action", "decision": self.step, "action_index": indices[0]})
            return indices[0]
        messages = prompt(observation)
        generation_seed = self.rng.randrange(2**31)
        item = self.policy(messages, seed=generation_seed,
                           prompt_id=f"{self.episode_id}:decision_{self.step}", phase=observation["phase"])
        parsed = parse_action(item.raw_text, indices, finish_reason="length" if item.truncated else "stop")
        item.action_index, item.valid, item.error = parsed.action_index, parsed.valid, parsed.error
        self.decisions.append(item)
        self._record({"event": "model_action", "stage": "before_terminal_outcome_credit",
                      "decision": self.step, "generation_seed": generation_seed, "messages": messages,
                      "completion": asdict(item), "parse": parsed.to_dict()})
        if not parsed.valid:
            raise InvalidDecision(parsed.error)
        return parsed.action_index

    def _record(self, record):
        self.ledger.write(json.dumps(record, allow_nan=False) + "\n")
        self.ledger.flush()


def collect_episode(policy, *, world_seed, policy_seed, env_config, output,
                    win_reward, loss_reward, stop_file=None, policy_identity=None):
    from sts_env.cli import run_episode
    output.mkdir(parents=True, exist_ok=False)
    with (output / "responses.jsonl").open("x") as ledger:
        agent = CollectingAgent(policy, ledger, stop_file, f"world_{world_seed}:policy_{policy_seed}")
        summary = run_episode(seed=world_seed, policy_seed=policy_seed,
            agent_spec="sts_llm.game_training:parent_hf_policy", agent_config=policy_identity or {},
            config=env_config, output=output / "game.jsonl", agent_cache={"agent": agent})
    status, outcome, reward = terminal_result(summary, win_reward=win_reward, loss_reward=loss_reward)
    episode = Episode(world_seed, policy_seed, status, outcome, reward, summary,
                      agent.decisions, agent.forced_actions)
    atomic_json(output / "result.json", episode.record())
    return episode


def credit_groups(groups, *, expected_group_size):
    """Drop whole incomplete groups; assign one terminal advantage to all decisions."""
    eligible, records = [], []
    for index, group in enumerate(groups):
        complete = len(group) == expected_group_size and all(g.status == "complete" and
            g.reward is not None and math.isfinite(g.reward) for g in group)
        record = {"group": index, "eligible": complete,
                  "statuses": dict(Counter(g.status for g in group))}
        if complete:
            if len({g.world_seed for g in group}) != 1:
                raise ValueError("a comparison group must share one world seed")
            advantages, variance = normalized_advantages([g.reward for g in group])
            for game, advantage in zip(group, advantages):
                game.advantage = advantage
                for decision in game.decisions:
                    decision.reward, decision.advantage = game.reward, advantage
            eligible.extend(group)
            record.update(world_seed=group[0].world_seed, reward_variance=variance,
                          rewards=[g.reward for g in group])
        records.append(record)
    return eligible, records


def optimize_games(model, optimizer, games, args, pad_id, *, on_frozen=None, on_update=None, should_stop=lambda: False):
    """Sum decisions within games, then average games, through the shared optimizer.

    optimizer_step averages N completions. Scaling both advantages and KL beta by
    N/G instead gives sum(decision losses)/G, including games with zero choices.
    The explicitly chosen response token normalization is otherwise unchanged.
    """
    cohort = [copy.deepcopy(d) for game in games for d in game.decisions]
    if not cohort:
        return [], [], 0.
    factor = len(cohort) / len(games)
    for decision in cohort:
        decision.advantage *= factor
    update_args = copy.copy(args)
    update_args.kl_beta *= factor
    freeze_cohort_logps(model, cohort, microbatch=args.microbatch, pad_id=pad_id,
                        temperature=args.temperature)
    if on_frozen:
        on_frozen(cohort)
    updates = []
    for epoch in range(args.update_epochs):
        if should_stop():
            raise StopRequested("STOP before a full-game optimizer update")
        updates.append({"epoch": epoch, **optimizer_step(model, optimizer, cohort, update_args, pad_id)})
        if on_update:
            on_update(updates)
    return cohort, updates, factor


def settings(args, seeds):
    from sts_env.cli import environment_identity
    environment = environment_identity()
    excluded = {"output", "resume", "adapter", "iterations", "stop_file", "seeds_file", "evaluate_only"}
    return {"format_version": FORMAT_VERSION, "world_seeds": seeds,
            "prompt_version": PROMPT_VERSION, "shared_trainer_format_version": TRAINING_FORMAT_VERSION,
            "environment_contract": {key: environment[key] for key in ("environment_version", "observation_version")},
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in excluded},
            "episode_reduction": "sum_decisions_then_mean_games", "reference": "frozen_base_adapters_disabled",
            "advantage_normalization": "group_mean_and_population_std"}


def provenance():
    from sts_env.cli import environment_identity
    directory = Path(__file__).parent
    return {"environment": environment_identity(), "prompt_version": PROMPT_VERSION,
            "shared_trainer_format_version": TRAINING_FORMAT_VERSION,
            "source_files_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                for name in ("game_training.py", "training.py", "generation.py", "observations.py")}}


def save_checkpoint(model, tokenizer, optimizer, rng, *, output, name, next_iteration, history, invariants):
    import torch
    target, pending = output / name, output / (name + ".pending")
    if target.exists() or pending.exists():
        raise FileExistsError("refusing to replace an existing checkpoint")
    model.save_pretrained(pending)
    tokenizer.save_pretrained(pending)
    torch.save({"settings": invariants, "provenance": provenance(), "next_iteration": next_iteration, "history": history,
        "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else []},
        pending / "game_training_state.pt")
    os.replace(pending, target)
    atomic_json(output / "recovery.json", {"last_completed_checkpoint": str(target), "next_iteration": next_iteration})
    return str(target)


def execute(model, tokenizer, stop_id, args, seeds):
    import torch
    rng, history, start = random.Random(args.policy_seed), [], 0
    invariants = settings(args, seeds)
    optimizer = None if args.evaluate_only else torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.)
    if args.resume:
        state = torch.load(args.resume / "game_training_state.pt", map_location=args.device, weights_only=False)
        if state["settings"] != invariants:
            raise ValueError("resume settings/world seeds differ from the completed checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        start, history = state["next_iteration"], state["history"]
        atomic_json(args.output / "recovery.json", {"last_completed_checkpoint": str(args.resume), "next_iteration": start})
    elif not args.evaluate_only:
        save_checkpoint(model, tokenizer, optimizer, rng, output=args.output, name="adapter_initial",
                        next_iteration=0, history=[], invariants=invariants)
    policy = HFPolicy(model, tokenizer, stop_id, args)
    env_config = {"mode": "full", "max_act": args.max_act, "max_steps": args.max_decisions,
                  "step_timeout": args.step_timeout, "ascension": args.ascension, "isolated": True}
    if args.build_dir:
        env_config["build_dir"] = str(args.build_dir)
    for iteration in range(start, 1 if args.evaluate_only else args.iterations):
        if args.stop_file.exists():
            raise StopRequested("STOP before collecting games")
        worlds = seeds if args.evaluate_only else rng.sample(seeds, min(args.worlds_per_iteration, len(seeds)))
        identity = {"model": args.model, "revision": args.revision, "effort": args.effort,
                    "trainable_sha256": trainable_hash(model)}
        groups = []
        for world in worlds:
            group = []
            for replica in range(args.group_size):
                if args.stop_file.exists():
                    raise StopRequested("STOP between games; retained incomplete iteration is not promoted")
                group.append(collect_episode(policy, world_seed=world, policy_seed=rng.randrange(2**31),
                    env_config=env_config, output=args.output / f"iteration_{iteration:04d}" / f"world_{world}_replica_{replica}",
                    win_reward=args.win_reward, loss_reward=args.loss_reward,
                    stop_file=args.stop_file, policy_identity=identity))
                if args.stop_file.exists():
                    raise StopRequested("STOP during the last game; its trace and response ledger are retained")
            groups.append(group)
        games = [g for group in groups for g in group]
        eligible, group_records = credit_groups(groups, expected_group_size=args.group_size)
        record = {"iteration": iteration, "groups": group_records,
            "episode_statuses": dict(Counter(g.status for g in games)),
            "wins": sum(g.outcome == "PLAYER_VICTORY" and g.status == "complete" for g in games),
            "losses": sum(g.outcome == "PLAYER_LOSS" and g.status == "complete" for g in games),
            "eligible_games": len(eligible), "model_decisions": sum(len(g.decisions) for g in games),
            "native_provenance": [json.loads(value) for value in sorted({json.dumps(
                g.summary.get("info", {}).get("native_provenance", {}), sort_keys=True) for g in games})]}
        atomic_json(args.output / f"collection_{iteration:04d}.json", record)
        if not args.evaluate_only:
            def frozen(cohort):
                path = args.output / f"cohort_{iteration:04d}.jsonl"
                with path.open("x") as stream:
                    for decision in cohort:
                        stream.write(json.dumps(asdict(decision), allow_nan=False) + "\n")
            _, updates, factor = optimize_games(model, optimizer, eligible, args, tokenizer.pad_token_id,
                on_frozen=frozen, should_stop=args.stop_file.exists,
                on_update=lambda updates: atomic_json(args.output / f"updates_{iteration:04d}.json", updates))
            record.update(updates=updates, completion_mean_to_game_mean_factor=factor)
        history.append(record)
        if not args.evaluate_only:
            record["checkpoint"] = str(args.output / f"adapter_{iteration:04d}")
            save_checkpoint(model, tokenizer, optimizer, rng, output=args.output,
                name=f"adapter_{iteration:04d}", next_iteration=iteration+1, history=history, invariants=invariants)
        atomic_json(args.output / "progress.json", history)
    result = {"purpose": "full_game_evaluation" if args.evaluate_only else "full_game_terminal_outcome_rl",
        "status": "complete", "iterations": history, "reward_source": "terminal_game_outcome_only",
        "episode_reduction": invariants["episode_reduction"], "settings": invariants,
        "incomplete_groups": "Excluded whole; failures/truncations never become game losses."}
    atomic_json(args.output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--revision", default="7c75f2f49e74ac97c1852e62b4c860091d6a00ea")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True, help="JSON list of explicit training or evaluation worlds")
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start-adapter", dest="adapter", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--effort", choices=("low", "medium", "high"), required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument("--max-act", type=int, required=True)
    parser.add_argument("--ascension", type=int, choices=range(21), required=True)
    parser.add_argument("--step-timeout", type=float, required=True)
    parser.add_argument("--win-reward", type=float, required=True)
    parser.add_argument("--loss-reward", type=float, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--worlds-per-iteration", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=.2)
    parser.add_argument("--kl-beta", type=float, default=.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--loss-normalization", choices=("response_mean", "fixed_token_budget"))
    parser.add_argument("--loss-token-budget", type=int)
    args = parser.parse_args()
    if args.resume and (args.adapter or args.evaluate_only):
        parser.error("resume is training-only and cannot be combined with a warm start")
    try:
        validate_full_game_scope(args.max_act)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.evaluate_only and args.group_size < 2:
        parser.error("training needs at least two independently sampled full games per world")
    if not args.evaluate_only and args.loss_normalization is None:
        parser.error("training requires an explicit loss-normalization choice")
    if args.loss_normalization == "fixed_token_budget" and args.loss_token_budget is None:
        parser.error("fixed normalization requires an explicit study-wide loss-token-budget")
    if args.loss_token_budget is None:
        args.loss_token_budget = 2048
    if not math.isfinite(args.temperature) or args.temperature < 0 or not args.evaluate_only and args.temperature == 0:
        parser.error("temperature must be positive for training, or nonnegative for evaluation (zero is greedy)")
    positive = (args.max_new_tokens, args.max_sequence_length, args.max_decisions,
        args.step_timeout, args.group_size, args.worlds_per_iteration, args.iterations, args.update_epochs,
        args.microbatch, args.learning_rate, args.max_grad_norm, args.loss_token_budget, args.lora_rank, args.lora_alpha)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not 0 < args.clip_epsilon < 1 or not 0 <= args.kl_beta < math.inf:
        parser.error("sizes/temperature/timeouts/rate must be finite positive; valid clipping and KL required")
    if not all(math.isfinite(v) for v in (args.win_reward, args.loss_reward)) or args.win_reward <= args.loss_reward:
        parser.error("finite win reward must exceed loss reward (the intended setting is +1/-1)")
    try:
        seeds = validate_world_seeds(json.loads(args.seeds_file.read_text()))
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be fresh or empty; resume uses a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.stop_file = args.stop_file or args.output / "STOP"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds_sha256=hashlib.sha256(args.seeds_file.read_bytes()).hexdigest(),
        settings=settings(args, seeds), provenance=provenance())
    atomic_json(args.output / "config.json", config)
    try:
        import torch
        torch.manual_seed(args.policy_seed)
        model, tokenizer, stop_id = load_model(args)
        config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            model_trainable_sha256=trainable_hash(model))
        atomic_json(args.output / "config.json", config)
        execute(model, tokenizer, stop_id, args, seeds)
    except Exception as exc:
        recovery = args.output / "recovery.json"
        atomic_json(args.output / "failure.json", {"kind": type(exc).__name__, "message": str(exc),
            "status": "stopped" if isinstance(exc, StopRequested) else "failed",
            "recovery": json.loads(recovery.read_text()) if recovery.exists() else None,
            "note": "Resume the last completed checkpoint into a fresh output; incomplete iteration is regenerated."})
        raise


if __name__ == "__main__":
    main()
