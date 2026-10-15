"""Whole-world public collection/evaluation. Native integration is build-gated.

Only genuine terminals publish world shards. Cuts, faults and interrupted partial
worlds publish diagnostic quarantine receipts, never labels or loss observations.
Teacher queries label champion states and never choose trajectory actions.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
from pathlib import Path
import threading

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec, player_hp, public_hash
from sts_rl.recording import file_identity
from sts_rl.fresh_study import (Interrupted, PROTOCOL, Progress, RunControl, atomic_json,
                                finish_run, policy_observation, prepare_run, validate_world_range)

TEACHER_PROTOCOL = "canonical_rng_resampled_native_v2"
SPLIT_PROTOCOL = "fresh-start-split-v1"
REPEAT_SAMPLE_OFFSET = 1024
MAX_SEARCH_SAMPLES = 1008
PROVENANCE_KEYS = ("python_interface_sha256", "runtime_python_version", "environment_version",
                   "observation_version", "native_provenance")
DEFAULT_PROBABILITIES = {f"{room}{act}": probability for act in (1, 2, 3)
                         for room, probability in (("BOSS", 1.0 if act > 1 else .25), ("ELITE", .6 if act > 1 else .2),
                                                   ("MONSTER", .08 if act > 1 else .04),
                                                   ("EVENT", .08 if act > 1 else .04))}


def provenance(info):
    return {key: info[key] for key in PROVENANCE_KEYS}


def group(observation):
    state = observation["state"]
    return f"{str(state.get('room', '')).rsplit('.', 1)[-1]}{int(state.get('act', 0))}"


def task(seed, expected=None, prefix=(), expected_hash=None):
    validate_world_range(seed, 1)
    return TaskSpec(seed=seed, kind="run", prefix=tuple(prefix), seek_combat=False,
                    ascension=0, max_act=3, horizon=4096, battle_simulations=1000,
                    expected_provenance=expected, expected_public_hash=expected_hash)


def search_plan(observation_hash, samples=24, repeat=False):
    """Record the actual draw-order, battle-RNG and search seeds passed by PanelEpisode."""
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_SEARCH_SAMPLES:
        raise ValueError("Invalid search sample count; first/repeat seed sets must stay disjoint")
    offset = REPEAT_SAMPLE_OFFSET if repeat else 0
    base = int(observation_hash[:10], 16) * 4096 + offset
    sample_ids = list(range(offset + 1, offset + samples + 1))
    return {"search_seed": base, "samples": sample_ids, "draw_order_seeds": sample_ids,
            "effective_rng_seeds": [base * 1009 + sample for sample in sample_ids],
            "effective_search_seeds": [base + sample for sample in sample_ids]}


def is_test_world(world):
    """Same predeclared whole-world split as the corpus builder; no training dependency."""
    digest = hashlib.sha256(f"{SPLIT_PROTOCOL}:{world}".encode()).hexdigest()
    return int(digest, 16) % 10 in {0, 1}


def plurality(votes, observation):
    legal = policy_observation(observation)["legal_actions"]
    indices = [int(a["index"]) for a in legal]
    if any(int(k) not in indices or isinstance(v, bool) or int(v) != v or v <= 0 for k, v in votes.items()):
        raise ValueError("Teacher histogram contains illegal indices/counts")
    if not votes:
        return None
    best = max(votes.values())
    return next(index for index in indices if votes.get(index) == best)


class WorldMetrics:
    def __init__(self):
        self.bosses, self.decisions = {}, Counter()
        self.counters = {name: {"opportunities": 0, "chosen": 0, "amount": 0} for name in (
            "smith_hp>=.85", "smith_hp>=.5", "smith_hp<.5", "elite_choice_hp>=.9",
            "card_skip", "removal", "combat_potion_drink", "combat_potion_discard", "shop_spend")}

    def count(self, name, chosen, amount=0):
        row = self.counters.setdefault(name, {"opportunities": 0, "chosen": 0, "amount": 0})
        row["opportunities"] += 1
        row["chosen"] += int(chosen)
        row["amount"] += amount

    def transition(self, before, action, after, *, win=False):
        state, next_state = before["state"], after["state"]
        act, phase = int(state.get("act", 0)), before["phase"]
        self.decisions[f"{phase}:{act}"] += 1
        if phase == "combat" and str(state.get("room", "")).rsplit(".", 1)[-1] == "BOSS":
            row = self.bosses.setdefault(str(act), {"reach": True, "pass": False,
                "entry_hp": player_hp(before), "entry_max_hp": state.get("max_hp")})
            # Secret Portal can enter a boss at any floor. A live exit, advancement
            # or terminal victory establishes passage, never floor arithmetic.
            if win or (player_hp(after) > 0 and (after["phase"] != "combat" or
                                                int(next_state.get("act", act)) > act)):
                row["pass"] = True
        legal = policy_observation(before)["legal_actions"]
        if len(legal) <= 1:
            return  # Behavior denominators describe choices, not forced transitions.
        descriptions = [a["description"] for a in legal]
        chosen = next(a["description"] for a in legal if a["index"] == action)
        hp = player_hp(before) / max(1, int(state.get("max_hp", 1)))
        if any(d.startswith("smith") for d in descriptions):
            band = ">=.85" if hp >= .85 else ">=.5" if hp >= .5 else "<.5"
            self.count("smith_hp" + band, chosen.startswith("smith"))
        paths = [d for d in descriptions if "map node" in d]
        if hp >= .9 and any("room=ELITE" in d for d in paths) and any("room=ELITE" not in d for d in paths):
            self.count("elite_choice_hp>=.9", "room=ELITE" in chosen)
        for name, prefix in (("card_skip", "skip card"), ("removal", "buy card remove"),
                             ("combat_potion_drink", "drink potion "), ("combat_potion_discard", "discard potion ")):
            if any(d.startswith(prefix) for d in descriptions):
                self.count(name, chosen.startswith(prefix))
        if any(d.startswith("buy ") for d in descriptions):
            spent = max(0, int(state.get("gold", 0)) - int(next_state.get("gold", 0)))
            self.count("shop_spend", spent > 0, spent)


class FightBank:
    def __init__(self, seed, expected, source):
        self.seed, self.expected, self.source = seed, expected, source
        self.rows, self.pending = [], None

    def transition(self, before, action, after, prefix, *, win=False, terminal=False):
        # Complete an existing fight first, including the final victorious boss.
        if self.pending is not None and (terminal or after["phase"] != "combat" or
                after["state"].get("act") != before["state"].get("act")):
            won = win or player_hp(after) > 0
            hp_after = player_hp(after) if won else 0
            self.pending.update(baseline_win=won, baseline_hp_after=hp_after,
                                baseline_hp_lost=self.pending["entry_hp"] - hp_after)
            self.rows.append(self.pending)
            self.pending = None
        if before["phase"] != "combat" and after["phase"] == "combat" and not terminal:
            entry = next(a for a in before["legal_actions"] if a["index"] == action)
            self.pending = {"protocol": PROTOCOL, "world_seed": self.seed,
                "start_id": len(prefix), "group": group(after), "prefix": list(prefix),
                "task": task(self.seed, self.expected, prefix, public_hash(before)).to_dict(),
                "observation": before, "public_hash": public_hash(before),
                "entering_action": action, "entering_description": entry["description"],
                "first_combat_hash": public_hash(after), "expected_provenance": self.expected,
                "source": self.source, "entry_hp": player_hp(after),
                "act": after["state"].get("act"), "room": after["state"].get("room"),
                "floor": after["state"].get("floor"),
                "encounter": after["state"].get("encounter"),
                "public_enemy_names": [e.get("name") for e in after["state"].get("combat", {}).get("enemies", [])]}


def query_votes(worker, observation, plan, config, control):
    if len(policy_observation(observation)["legal_actions"]) <= 1:
        raise ValueError("Teacher queries require an informative combat menu")
    votes = Counter()
    for offset in range(0, len(plan["samples"]), 4):
        control.check()
        chunk = worker.search_votes_public(config["simulations"], plan["samples"][offset:offset+4],
                                           plan["search_seed"])
        if not chunk:
            return {}  # Entire teacher target abstains, never partial-vote fallback.
        if sum(chunk.values()) != len(plan["samples"][offset:offset+4]):
            raise ValueError("Teacher returned an incomplete vote histogram")
        plurality(chunk, observation)  # Validate secondary histograms as well as targets.
        votes.update(chunk)
    # The child checks the live public hash; the parent's cached observation
    # cannot establish that a native scratch query left the live state unchanged.
    return dict(votes)


def play_world(worker, policy, seed, config, control, progress, source, label_count):
    prefix, labels, sampling, attempted_action = [], [], Counter(), None
    metrics, expected, bank = WorldMetrics(), None, None
    spec = task(seed)
    try:
        control.check()
        observation, info = worker.reset(spec)
        expected = provenance(info)
        if expected["environment_version"] != "0.1.5":
            raise RuntimeError("fresh_start_v1 requires environment 0.1.5; current checkout is preparation-only")
        spec = task(seed, expected, expected_hash=public_hash(observation))
        bank = FightBank(seed, expected, source)
        while True:
            control.check()
            action = policy.choose_action(observation)
            attempted_action = action
            if config["mode"] == "collect":
                phase = observation["phase"]
                key, state_hash = group(observation), public_hash(observation)
                sample = {"observation": observation, "public_hash": state_hash, "group": key,
                          "world_seed": seed, "prefix": list(prefix), "source": source,
                          "expected_provenance": expected, "source_action": action}
                if not config["teacher_labels"]:
                    sample.update(label_kind="probe_unlabelled")
                    labels.append(sample)
                    sampling[key + ":probe"] += 1
                elif phase != "combat":
                    sample.update(label=action, label_kind="champion_hard",
                                  choice=len(policy_observation(observation)["legal_actions"]) > 1)
                    labels.append(sample)
                    sampling["ooc"] += 1
                    sampling["ooc:choice" if sample["choice"] else "ooc:forced"] += 1
                else:
                    sampling[key + ":seen"] += 1
                    informative = len(policy_observation(observation)["legal_actions"]) > 1
                    sampling[key + (":choice" if informative else ":forced")] += 1
                    probability = config["teacher_probabilities"].get(key, 0.0)
                    uniform = int(hashlib.sha256((state_hash + ":sample").encode()).hexdigest()[:13], 16) / 16**13
                    if informative and uniform < probability:
                        plan = search_plan(state_hash, config["samples"])
                        votes = query_votes(worker, observation, plan, config, control)
                        label = plurality(votes, observation)
                        sample.update(label=label, votes=votes, search=plan,
                                      label_kind=TEACHER_PROTOCOL,
                                      tie_break="first_current_legal_index_among_max_votes")
                        sampling[key + (":abstain" if label is None else ":labelled")] += 1
                        if (is_test_world(seed) and config["self_consistency_every"] and
                                int(state_hash, 16) % config["self_consistency_every"] == 0):
                            repeat = search_plan(state_hash, config["samples"], repeat=True)
                            sample["self_consistency"] = {"search": repeat,
                                "votes": query_votes(worker, observation, repeat, config, control),
                                "use_for_training": False, "split_protocol": SPLIT_PROTOCOL}
                        labels.append(sample)
                        if label is not None:
                            label_count(1)
            after, _, terminated, truncated, info = worker.step(action)
            win = info.get("task_outcome") in {"act_clear", "victory"}
            metrics.transition(observation, action, after, win=win)
            if config["bank"]:
                bank.transition(observation, action, after, prefix, win=win, terminal=terminated)
            prefix.append(action)
            attempted_action = None
            progress.add(decisions=1)
            if terminated or truncated:
                if truncated or info.get("task_outcome") not in {"act_clear", "victory", "defeat"}:
                    raise Interrupted("cut:" + info.get("termination_reason", "unknown"))
                if "action_prefix" in info and info["action_prefix"] != prefix:
                    raise RuntimeError("Completed worker/collector action prefixes differ")
                record = {"protocol": PROTOCOL, "world_seed": seed, "status": "completed",
                    "task": spec.to_dict(), "expected_provenance": expected, "source": source,
                    "win": win, "floor": after["state"].get("floor"),
                    "bosses": metrics.bosses, "decisions": dict(metrics.decisions),
                    "behavior": metrics.counters, "action_prefix": prefix,
                    "action_hash": public_hash({"actions": prefix}), "terminal_hash": public_hash(after),
                    "sampling": dict(sampling), "labels": labels, "bank": bank.rows if config["bank"] else []}
                return record
            observation = after
    except (SimulatorError, Interrupted) as exc:
        # Programming errors are intentionally not caught here.
        return {"protocol": PROTOCOL, "world_seed": seed,
                "status": "fault" if isinstance(exc, SimulatorError) else "censored",
                "reason": str(exc), "last_fault_path": getattr(worker, "last_fault_path", None),
                "action_prefix": prefix, "attempted_action": attempted_action, "expected_provenance": expected,
                "source": source, "discarded_partial_labels": len(labels),
                "discarded_partial_bank": len(bank.rows) + int(bank.pending is not None) if bank else 0}
    except Exception as exc:
        # Carry recovery information to the coordinator, then propagate the bug.
        # This is a programming-error quarantine, never a simulator-fault result.
        exc.fresh_partial = {"protocol": PROTOCOL, "world_seed": seed, "status": "programming_error",
            "reason": f"{type(exc).__name__}: {exc}", "action_prefix": prefix,
            "attempted_action": attempted_action, "expected_provenance": expected,
            "source": source, "discarded_partial_labels": len(labels)}
        raise


def summarize(records, requested, sampling, labels):
    completed = [r for r in records if r["status"] == "completed"]
    counts = Counter(r["status"] for r in records)
    complete = len(completed) == requested
    boss = {}
    behavior = {}
    for act in (1, 2, 3):
        rows = [r["bosses"][str(act)] for r in completed if str(act) in r["bosses"]]
        boss[str(act)] = {"reach": len(rows), "pass": sum(r["pass"] for r in rows),
                          "cohort_denominator": len(completed), "pass_denominator": len(rows),
                          "entry_hp": [r["entry_hp"] for r in rows]}
    for record in completed:
        for key, row in record["behavior"].items():
            target = behavior.setdefault(key, Counter())
            target.update(row)
    return {"protocol": PROTOCOL, "completed": len(completed), "faults": counts["fault"],
            "cuts": sum(r["status"] == "censored" and r.get("reason", "").startswith("cut:") for r in records),
            "censored": counts["censored"] + counts["programming_error"],
            "programming_errors": counts["programming_error"], "unstarted": requested-len(records),
            "cohort_complete": complete, "confidence_claims_allowed": complete,
            "wins": sum(r["win"] for r in completed),
            "mean_floor": sum(r["floor"] for r in completed) / len(completed) if completed else None,
            "bosses": boss, "behavior": behavior, "sampling": dict(sampling),
            "combat_labels": labels, "records": records}


def run_collection(config, policy, *, worker_factory=None):
    """Each thread owns one persistent worker; policy serializes singleton forwards."""
    from sts_rl.common import WORKER_CONFIG
    from sts_rl.fresh_panel import FreshPanelWorker
    validate_config(config)
    out = Path(config["out"])
    source = prepare_run(out, config, policy)
    control, records, sampling = RunControl(out, config.get("deadline_utc")), [], Counter()
    progress = Progress(out, control)
    lock = threading.Lock()
    stop_reasons = set()
    next_seed, queried_labels, saved_labels = config["world_start"], 0, 0

    def label_count(amount):
        nonlocal queried_labels
        with lock:
            queried_labels += amount

    def loop():
        nonlocal next_seed, saved_labels
        worker = (worker_factory or FreshPanelWorker)(config={**WORKER_CONFIG,
                  "fast_observations": False}, timeout=config["timeout"])
        try:
            while True:
                try:
                    control.check()
                except Interrupted as exc:
                    with lock:
                        stop_reasons.add(str(exc))
                    return
                with lock:
                    if next_seed >= config["world_start"] + config["n"]:
                        return
                    if config["max_labels"] and saved_labels >= config["max_labels"]:
                        stop_reasons.add("max_labels")
                        return
                    seed, next_seed = next_seed, next_seed + 1
                programming_error = None
                try:
                    record = play_world(worker, policy, seed, config, control, progress, source, label_count)
                except Exception as exc:
                    programming_error = exc
                    record = exc.fresh_partial
                completed = record["status"] == "completed"
                path = out / ("worlds" if completed else "quarantine") / f"{seed}.json.gz"
                if path.exists():
                    raise FileExistsError(path)
                atomic_json(path, record, compressed=True)
                # Full raw observations stay only in immutable per-world shards.
                brief = {k: v for k, v in record.items() if k not in {"labels", "bank", "source"}}
                brief["shard"] = str(path.relative_to(out))
                brief["shard_identity"] = file_identity(path)
                with lock:
                    records.append(brief)
                    if record["status"] == "censored" and not record["reason"].startswith("cut:"):
                        stop_reasons.add(record["reason"])
                    if completed:
                        sampling.update(record["sampling"])
                        saved_labels += sum(r.get("label_kind") == TEACHER_PROTOCOL and r.get("label") is not None
                                            for r in record["labels"])
                progress.add(**{"completed" if completed else "faults" if record["status"] == "fault" else "censored": 1})
                progress.emit("world", world_seed=seed, status=record["status"])
                if programming_error is not None:
                    raise programming_error
                if record["status"] == "censored" and not record["reason"].startswith("cut:"):
                    return
        finally:
            worker.close()

    error = None
    try:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            try:
                pending = {pool.submit(loop) for _ in range(config["workers"])}
                while pending:
                    done, pending = wait(pending, timeout=5)
                    progress.emit()
                    for future in done:
                        try:
                            future.result()
                        except Exception as exc:
                            error = error or exc
                            (out / "STOP").touch()
            except BaseException:
                # Wake only our workers at their next finite control boundary.
                # Set this before executor shutdown waits for outstanding RPCs.
                (out / "STOP").touch()
                raise
    except (Interrupted, KeyboardInterrupt) as exc:
        stop_reasons.add(str(exc) or "KeyboardInterrupt")
    except Exception as exc:
        error = error or exc
    finally:
        try:
            control.close()
        except Exception as exc:
            error = error or exc
        records.sort(key=lambda r: r["world_seed"])
        summary = summarize(records, config["n"], sampling, saved_labels)
        summary.update(queried_combat_labels=queried_labels, stop_reasons=sorted(stop_reasons),
                       max_labels_overshoot=max(0, saved_labels-config["max_labels"]) if config["max_labels"] else 0)
        if error is not None:
            summary.update(programming_error=f"{type(error).__name__}: {error}", confidence_claims_allowed=False)
        finish_run(out, summary, source, progress)
    if error is not None:
        raise error
    return summary


def validate_config(config):
    import math
    validate_world_range(config["world_start"], config["n"])
    if config["mode"] not in {"eval", "collect"}:
        raise ValueError("mode must be eval or collect")
    if config["workers"] < 1 or not math.isfinite(config["timeout"]) or config["timeout"] <= 0:
        raise ValueError("Require workers >=1 and finite positive timeout")
    if (isinstance(config["samples"], bool) or not isinstance(config["samples"], int) or
            not 1 <= config["samples"] <= MAX_SEARCH_SAMPLES or config["simulations"] < 1 or
            config["max_labels"] < 0 or config["self_consistency_every"] < 0):
        raise ValueError("Invalid search/collection bounds")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in config["teacher_probabilities"].values()):
        raise ValueError("Invalid frozen teacher sampling probabilities")
    if config["mode"] == "collect" and not config["teacher_labels"]:
        if not 93_000_000_000 <= config["world_start"] < config["world_start"] + config["n"] <= 94_000_000_000:
            raise ValueError("Unlabelled probes require the separate 93e9 namespace")
    if config["mode"] == "collect" and (config.get("combat") or config.get("ooc")):
        raise ValueError("Collection trajectories must use the champion")
