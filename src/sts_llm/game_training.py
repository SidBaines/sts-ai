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

from .dataset import canonical_world_identity
from .generation import parse_action
from .observations import PROMPT_VERSION, legal_indices, prompt
from .training import (TRAINING_FORMAT_VERSION, Completion, atomic_json, check_adapter, freeze_cohort_logps,
                       load_model, normalized_advantages, optimizer_step, trainable_hash)

# 2: model-invalid games are scored outcomes; episode statuses are win/loss/
# model_invalid/simulator_fault/budget_truncated/unknown_terminal/stopped.
FORMAT_VERSION = 2

# Scored games keep their comparison group eligible for credit. A model-invalid
# game is the model's own failure, so it is scored with --invalid-game-reward.
SCORED_STATUSES = ("win", "loss", "model_invalid")
# Unscored games exclude their whole group and are never counted as losses.
UNSCORED_STATUSES = ("simulator_fault", "budget_truncated", "unknown_terminal", "stopped")
EPISODE_STATUSES = SCORED_STATUSES + UNSCORED_STATUSES


class StopRequested(RuntimeError):
    pass


class InvalidDecision(RuntimeError):
    """The model's output could not be executed: unparseable, illegal or cut off at the output cap."""


class ContextBudgetExceeded(RuntimeError):
    """The prompt plus the completion budget exceeds the context window: a configuration budget."""


class NoUpdateLimitReached(RuntimeError):
    """Too many consecutive training iterations produced no optimizer update."""


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
    termination: dict | None = None

    def record(self):
        return {k: v for k, v in asdict(self).items() if k != "decisions"} | {
            "model_decisions": len(self.decisions),
            "generated_tokens": sum(len(d.output_ids) for d in self.decisions)}


def terminal_result(summary, *, win_reward, loss_reward, invalid_game_reward=None, termination=None):
    """Classify one game as (status, outcome, reward).

    Wins, losses and model-invalid endings are scored. Simulator faults,
    decision/act/context budgets and stops are unscored, never losses.
    """
    category = (termination or {}).get("category")
    if category == "model_invalid":
        if invalid_game_reward is None or not math.isfinite(invalid_game_reward):
            raise ValueError("a model-invalid game needs a finite invalid_game_reward")
        return "model_invalid", None, float(invalid_game_reward)
    if category in ("budget_truncated", "stopped"):
        return category, None, None
    if category is not None:
        raise ValueError(f"unexpected agent termination {category!r}")
    if summary.get("status") == "interrupted":
        return "stopped", None, None
    if summary.get("error"):
        return "simulator_fault", None, None
    if summary.get("truncated") or summary.get("status") != "terminated" or not summary.get("terminated"):
        return "budget_truncated", None, None
    outcome = str(summary.get("info", {}).get("outcome", "")).rsplit(".", 1)[-1]
    if outcome == "PLAYER_VICTORY":
        return "win", outcome, float(win_reward)
    if outcome == "PLAYER_LOSS":
        return "loss", outcome, float(loss_reward)
    return "unknown_terminal", outcome, None


def outcome_counts(games):
    """Per-status counts plus win rates over all attempted and over scored games."""
    counts = Counter(g.status for g in games)
    result = {status: counts.get(status, 0) for status in EPISODE_STATUSES}
    unexpected = set(counts) - set(EPISODE_STATUSES)
    if unexpected:
        raise ValueError(f"unexpected episode statuses {sorted(unexpected)}")
    attempted, scored = len(games), sum(result[s] for s in SCORED_STATUSES)
    result.update(attempted=attempted, scored=scored,
                  win_rate_over_attempted=result["win"] / attempted if attempted else None,
                  win_rate_over_scored=result["win"] / scored if scored else None)
    return result


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
            raise ContextBudgetExceeded("prompt_plus_completion_budget_exceeds_context")
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
    """Model agent for run_episode that remembers why it ended a game.

    run_episode turns every agent exception into an unscored "error" summary,
    so the agent records the category itself. Exceptions other than invalid
    output, the context budget and stop requests are kept in ``exception`` for
    collect_episode to re-raise.
    """
    def __init__(self, policy, ledger, stop_file=None, episode_id="episode"):
        self.policy, self.ledger, self.stop_file = policy, ledger, stop_file
        self.episode_id = episode_id
        self.decisions, self.forced_actions, self.step = [], 0, 0
        self.termination, self.exception = None, None

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.decisions, self.forced_actions, self.step = [], 0, 0
        self.termination, self.exception = None, None

    def choose_action(self, observation):
        try:
            return self._choose_action(observation)
        except BaseException as exc:
            category = ("model_invalid" if isinstance(exc, InvalidDecision) else
                        "budget_truncated" if isinstance(exc, ContextBudgetExceeded) else
                        "stopped" if isinstance(exc, StopRequested) else "policy_error")
            self.termination = {"category": category, "decision": self.step,
                                "kind": type(exc).__name__, "message": str(exc)}
            if category == "policy_error":
                self.exception = exc
            try:
                self._record({"event": "termination", **self.termination})
            except Exception:
                pass  # The original exception matters more than a ledger line.
            raise

    def _choose_action(self, observation):
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
                    win_reward, loss_reward, invalid_game_reward=None, stop_file=None, policy_identity=None):
    """Play one game. Policy errors, interrupts and stops are re-raised after result.json is written."""
    from sts_env.cli import run_episode
    output.mkdir(parents=True, exist_ok=False)
    with (output / "responses.jsonl").open("x") as ledger:
        agent = CollectingAgent(policy, ledger, stop_file, f"world_{world_seed}:policy_{policy_seed}")
        summary = run_episode(seed=world_seed, policy_seed=policy_seed,
            agent_spec="sts_llm.game_training:parent_hf_policy", agent_config=policy_identity or {},
            config=env_config, output=output / "game.jsonl", agent_cache={"agent": agent})
    if agent.exception is not None:
        # A crash inside the policy (for example CUDA out of memory) is not a game result.
        atomic_json(output / "result.json", Episode(world_seed, policy_seed, "policy_error", None, None, summary,
            agent.decisions, agent.forced_actions, termination=agent.termination).record())
        raise agent.exception
    status, outcome, reward = terminal_result(summary, win_reward=win_reward, loss_reward=loss_reward,
        invalid_game_reward=invalid_game_reward, termination=agent.termination)
    episode = Episode(world_seed, policy_seed, status, outcome, reward, summary,
                      agent.decisions, agent.forced_actions, termination=agent.termination)
    atomic_json(output / "result.json", episode.record())
    if summary.get("status") == "interrupted":
        raise KeyboardInterrupt("interrupted during a full game; its trace and result.json are retained")
    if status == "stopped":
        raise StopRequested("STOP during a game; its trace and response ledger are retained")
    return episode


def credit_groups(groups, *, expected_group_size):
    """Credit groups whose games are all scored; exclude any group with an unscored game.

    Every decision of a scored game, including a model-invalid final completion,
    receives that game's terminal advantage.
    """
    eligible, records = [], []
    for index, group in enumerate(groups):
        complete = len(group) == expected_group_size and all(g.status in SCORED_STATUSES and
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
                          rewards=[g.reward for g in group], outcome_signal=variance > 0)
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
    # Run-control flags are not scientific settings and may change on resume.
    excluded = {"output", "resume", "adapter", "iterations", "stop_file", "seeds_file", "evaluate_only",
                "max_no_update_iterations", "allow_adapter_mismatch", "allow_training_worlds"}
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


def save_checkpoint(model, tokenizer, optimizer, rng, *, output, name, next_iteration, history, invariants,
                    inherited_training_worlds=None):
    import torch
    target, pending = output / name, output / (name + ".pending")
    if target.exists() or pending.exists():
        raise FileExistsError("refusing to replace an existing checkpoint")
    model.save_pretrained(pending)
    tokenizer.save_pretrained(pending)
    torch.save({"settings": invariants, "provenance": provenance(), "next_iteration": next_iteration, "history": history,
        "inherited_training_worlds": inherited_training_worlds,
        "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else []},
        pending / "game_training_state.pt")
    os.replace(pending, target)
    atomic_json(output / "recovery.json", {"last_completed_checkpoint": str(target), "next_iteration": next_iteration})
    return str(target)


def adapter_training_worlds(record):
    """Worlds an adapter (and any adapter it was warm-started from) was trained on.

    Full-game adapters record their world seeds in game_training_state.pt.
    Teacher-trained adapters record them in their run's config.json
    (``training_world_identities``); older runs did not, and are reported as
    unrecorded rather than assumed clean.
    """
    worlds, sources = set(), []
    kind, settings = record.get("kind"), record.get("settings") or {}
    if kind == "full_game":
        seeds = settings.get("world_seeds")
        recorded = isinstance(seeds, list)
        worlds.update(canonical_world_identity(seed) for seed in seeds or [])
        sources.append({"kind": kind, "path": record.get("path"), "recorded": recorded, "worlds": len(seeds or [])})
        inherited = record.get("inherited_training_worlds")
        if inherited is not None:
            worlds.update(inherited.get("worlds", []))
            sources.extend(inherited.get("sources", []))
    elif kind == "teacher_feedback":
        identities = (record.get("run_config") or {}).get("training_world_identities")
        recorded = isinstance(identities, list)
        worlds.update(canonical_world_identity(world) for world in identities or [])
        sources.append({"kind": kind, "path": record.get("path"), "recorded": recorded,
                        "worlds": len(identities or [])})
    else:
        sources.append({"kind": kind, "path": record.get("path"), "recorded": False, "worlds": 0})
    return {"recorded": all(source["recorded"] for source in sources), "worlds": sorted(worlds),
            "sources": sources}


def evaluation_world_check(seeds, lineage, *, allow_training_worlds):
    """Refuse evaluation worlds that the adapter was trained on, unless explicitly allowed."""
    if lineage is None:
        return {"adapter": None, "result": "base model; no training worlds"}
    overlap = sorted({canonical_world_identity(seed) for seed in seeds} & set(lineage["worlds"]))
    result = {"training_worlds_recorded": lineage["recorded"], "training_worlds": len(lineage["worlds"]),
              "sources": lineage["sources"], "overlapping_worlds": overlap,
              "overlap_allowed_by_flag": bool(overlap and allow_training_worlds),
              "result": ("overlap allowed by --allow-training-worlds" if overlap else
                         "no overlap with recorded training worlds" if lineage["recorded"] else
                         "no overlap with the recorded part; some training worlds were not recorded")}
    if overlap and not allow_training_worlds:
        raise ValueError(f"{len(overlap)} evaluation worlds were used to train this adapter "
                         f"(for example {overlap[:3]}); choose held-out seeds or pass --allow-training-worlds")
    return result


def consecutive_no_update_iterations(history):
    count = 0
    for record in reversed(history):
        if record.get("updated") is not False:
            break
        count += 1
    return count


def execute(model, tokenizer, stop_id, args, seeds, *, inherited_training_worlds=None, world_check=None):
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
        inherited_training_worlds = state.get("inherited_training_worlds")
        atomic_json(args.output / "recovery.json", {"last_completed_checkpoint": str(args.resume), "next_iteration": start})
    elif not args.evaluate_only:
        save_checkpoint(model, tokenizer, optimizer, rng, output=args.output, name="adapter_initial",
                        next_iteration=0, history=[], invariants=invariants,
                        inherited_training_worlds=inherited_training_worlds)
    policy = HFPolicy(model, tokenizer, stop_id, args)
    env_config = {"mode": "full", "max_act": args.max_act, "max_steps": args.max_decisions,
                  "step_timeout": args.step_timeout, "ascension": args.ascension, "isolated": True}
    if args.build_dir:
        env_config["build_dir"] = str(args.build_dir)
    all_games = []
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
                    invalid_game_reward=args.invalid_game_reward,
                    stop_file=args.stop_file, policy_identity=identity))
                if args.stop_file.exists():
                    raise StopRequested("STOP during the last game; its trace and response ledger are retained")
            groups.append(group)
        games = [g for group in groups for g in group]
        all_games.extend(games)
        eligible, group_records = credit_groups(groups, expected_group_size=args.group_size)
        counts = outcome_counts(games)
        record = {"iteration": iteration, "groups": group_records, "status_counts": counts,
            "wins": counts["win"], "losses": counts["loss"], "model_invalid_games": counts["model_invalid"],
            "eligible_groups": sum(r["eligible"] for r in group_records),
            "groups_with_outcome_signal": sum(bool(r.get("outcome_signal")) for r in group_records),
            "eligible_games": len(eligible), "model_decisions": sum(len(g.decisions) for g in games),
            "invalid_model_decisions": sum(not d.valid for g in games for d in g.decisions),
            "native_provenance": [json.loads(value) for value in sorted({json.dumps(
                g.summary.get("info", {}).get("native_provenance", {}), sort_keys=True) for g in games})]}
        if not args.evaluate_only:
            record["update_skipped"] = ("no_eligible_groups" if not eligible else
                "no_model_decisions_in_eligible_games" if not any(g.decisions for g in eligible) else None)
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
            record.update(updates=updates, completion_mean_to_game_mean_factor=factor, updated=bool(updates))
        history.append(record)
        if not args.evaluate_only:
            record["checkpoint"] = str(args.output / f"adapter_{iteration:04d}")
            save_checkpoint(model, tokenizer, optimizer, rng, output=args.output,
                name=f"adapter_{iteration:04d}", next_iteration=iteration+1, history=history, invariants=invariants,
                inherited_training_worlds=inherited_training_worlds)
        atomic_json(args.output / "progress.json", history)
        if not args.evaluate_only:
            streak = consecutive_no_update_iterations(history)
            if streak >= args.max_no_update_iterations:
                raise NoUpdateLimitReached(
                    f"{streak} consecutive iterations made no optimizer update (latest: {record['update_skipped']}); "
                    f"latest status counts {json.dumps({k: counts[k] for k in EPISODE_STATUSES})}. Unscored games "
                    "(simulator faults, budget truncations) exclude whole groups: check collection_NNNN.json and "
                    "the game traces, or raise --max-no-update-iterations")
    result = {"purpose": "full_game_evaluation" if args.evaluate_only else "full_game_terminal_outcome_rl",
        "status": "complete", "iterations": history, "reward_source": "terminal_game_outcome_only",
        "episode_reduction": invariants["episode_reduction"], "settings": invariants,
        "status_counts_this_run": outcome_counts(all_games),
        "status_semantics": ("win/loss/model_invalid are scored (model_invalid receives invalid_game_reward); "
            "simulator_fault/budget_truncated/unknown_terminal/stopped are unscored, exclude their whole group "
            "and are never counted as losses.")}
    if args.evaluate_only:
        result["evaluation"] = outcome_counts(all_games)
        result["evaluation_world_check"] = world_check
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
    parser.add_argument("--allow-adapter-mismatch", action="store_true",
                        help="Load an adapter whose recorded model/revision/effort/format/prompt version differs")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--allow-training-worlds", action="store_true",
                        help="Evaluate-only: allow seeds that the adapter was trained on")
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
    parser.add_argument("--invalid-game-reward", type=float,
                        help="Reward for a game ended by invalid model output; default and maximum: --loss-reward")
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--worlds-per-iteration", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-no-update-iterations", type=int, default=3,
                        help="Stop with an error after this many consecutive iterations without an update")
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
    if args.allow_training_worlds and not args.evaluate_only:
        parser.error("--allow-training-worlds applies only to --evaluate-only")
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
        args.microbatch, args.learning_rate, args.max_grad_norm, args.loss_token_budget, args.lora_rank, args.lora_alpha,
        args.max_no_update_iterations)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not 0 < args.clip_epsilon < 1 or not 0 <= args.kl_beta < math.inf:
        parser.error("sizes/temperature/timeouts/rate must be finite positive; valid clipping and KL required")
    if not all(math.isfinite(v) for v in (args.win_reward, args.loss_reward)) or args.win_reward <= args.loss_reward:
        parser.error("finite win reward must exceed loss reward (the intended setting is +1/-1)")
    if args.invalid_game_reward is None:
        args.invalid_game_reward = args.loss_reward
    if not math.isfinite(args.invalid_game_reward) or args.invalid_game_reward > args.loss_reward:
        parser.error("invalid-game reward must be finite and no larger than the loss reward")
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
        lineage, world_check = None, None
        if args.adapter:
            config["adapter_compatibility"], record = check_adapter(args.adapter, model=args.model,
                revision=args.revision, effort=args.effort, allow_mismatch=args.allow_adapter_mismatch)
            lineage = adapter_training_worlds(record)
            config["adapter_training_worlds"] = {k: v for k, v in lineage.items() if k != "worlds"} | {
                "world_count": len(lineage["worlds"])}
        if args.evaluate_only:
            world_check = evaluation_world_check(seeds, lineage, allow_training_worlds=args.allow_training_worlds)
            config["evaluation_world_check"] = world_check
        atomic_json(args.output / "config.json", config)
        model, tokenizer, stop_id = load_model(args)
        config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            model_trainable_sha256=trainable_hash(model))
        atomic_json(args.output / "config.json", config)
        execute(model, tokenizer, stop_id, args, seeds, inherited_training_worlds=lineage, world_check=world_check)
    except BaseException as exc:
        recovery = args.output / "recovery.json"
        status = ("stopped" if isinstance(exc, StopRequested) else
                  "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        atomic_json(args.output / "failure.json", {"kind": type(exc).__name__, "message": str(exc),
            "status": status,
            "recovery": json.loads(recovery.read_text()) if recovery.exists() else None,
            "note": "Resume the last completed checkpoint into a fresh output; incomplete iteration is regenerated."})
        raise


if __name__ == "__main__":
    main()
