"""Fresh fight bank replay, pre-initialization salting, and world-paired panels.

Native contract intentionally differs from the old arbitrary-encounter panel:
execute the saved OOC action first, then copy the resulting BATTLE GameContext,
then initialize its stored encounter. Pure fakes test ordering; native parity
must still be checked against the study's new build before interpreting results.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
import math
import multiprocessing
from pathlib import Path
import statistics
import traceback

from sts_env.env import SimulatorError, SimulatorTimeout
from sts_policy.curriculum import player_hp, public_hash
from sts_rl.panel import PanelEpisode, PanelWorker
from sts_rl.fresh_native import install_native_guards
from sts_rl.controlled_policy import policy_factory
from sts_rl.fresh_panel_grid import (DEFAULT_TARGETS, HP_DEFINITION, bank_identity,
                                     freeze_grid, load_grid, save_grid)
from sts_rl.recording import file_identity
from sts_rl.fresh_study import (Interrupted, PROTOCOL, Progress, RunControl, StudyPolicy,
                                atomic_json, prepare_run, validate_world_range)

_POLICIES = {}
_GRID_FIELDS = ("grid_hash", "selection_protocol", "targets", "resamples", "coverage", "coverage_sufficient",
                "coverage_shortfall", "eligible_worlds", "absent_groups")


def _policy(champion, combat):
    key = champion, combat
    if key not in _POLICIES:
        _POLICIES[key] = policy_factory(champion, combat=combat, original_policy=StudyPolicy)
    return _POLICIES[key]


class WorkerProgrammingError(RuntimeError):
    """Unexpected child Python exceptions abort a run instead of censoring a world."""


def _serve_fresh(connection, config):
    episode = FreshPanelEpisode(config)
    try:
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            try:
                connection.send(("ok", getattr(episode, method)(*args)))
            except Exception as exc:
                connection.send(("error" if isinstance(exc, SimulatorError) else "python_error",
                                 {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class FreshPanelWorker(PanelWorker):
    """Bounded RPCs with fault receipts and separate programming-error propagation."""
    def _launch(self):
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve_fresh, args=(child, self.config),
                                        name="fresh-native", daemon=True)
        self._process.start()
        child.close()

    def _call(self, method, *args):
        from sts_policy.faults import serializable
        with self._lock:
            if method in {"reset", "fresh_fight"}:
                self._rpc_history, self._task, self.last_fault_path = [], serializable(args[0]), None
                self.observation, self.info = None, None
            request = {"method": method, "args": serializable(args)}
            if self._process is None:
                if method not in {"reset", "fresh_fight"}:
                    raise RuntimeError("Reset required after fault")
                self._launch()
            child_error = None
            try:
                self._connection.send((method, args))
                if not self._connection.poll(self.timeout):
                    raise SimulatorTimeout(f"Fresh {method} exceeded {self.timeout:g}s")
                status, value = self._connection.recv()
                if status == "python_error":
                    self._kill()
                    raise WorkerProgrammingError(f"{value['type']}: {value['message']}\n{value['traceback']}")
                if status != "ok":
                    child_error = value
                    raise SimulatorError(f"{value['type']}: {value['message']}")
                if method == "fresh_fight" and (value.get("cut") or value.get("ub")):
                    self._record_fault(request, SimulatorError("Quarantined panel cut/undefined behavior"),
                                       {"returned_diagnostics": value})
                    value["last_fault_path"] = self.last_fault_path
                    if value.get("ub"):
                        self._kill()
                self._rpc_history.append(request)
                return value
            except (EOFError, BrokenPipeError, OSError, SimulatorError) as exc:
                error = exc if isinstance(exc, SimulatorError) else SimulatorError(f"Native exited during {method}: {exc}")
                try:
                    self._record_fault(request, error, child_error)
                finally:
                    self._kill()
                if error is exc:
                    raise
                raise error from exc

    def fresh_fight(self, bank, salt, champion, combat=None, expected_policy=None):
        return self._call("fresh_fight", bank, salt, champion, combat, expected_policy)


class FreshPanelEpisode(PanelEpisode):
    def reset(self, spec):
        from sts_env._backend.lightspeed_import import import_lightspeed
        install_native_guards(import_lightspeed(self.config.get("build_dir")))
        self._prepared_start = None
        return super().reset(spec)

    def search_votes_public(self, simulations, samples, search_seed=0):
        before = public_hash(self.episode._observe())
        votes = super().search_votes_public(simulations, samples, search_seed)
        if public_hash(self.episode._observe()) != before:
            raise RuntimeError("Public-search query mutated the live decision")
        return votes

    def search_details_public(self, simulations, futures, search_seeds, rng_base):
        before = public_hash(self.episode._observe())
        details = super().search_details_public(simulations, futures, search_seeds, rng_base)
        if public_hash(self.episode._observe()) != before:
            raise RuntimeError("Public-search query mutated the live decision")
        return details

    def _preinitialization(self, bank):
        from sts_rl.fresh_collection import task
        validate_world_range(bank["world_seed"], 1)
        observation, _ = self.reset(task(bank["world_seed"], bank["expected_provenance"],
                                         bank["prefix"], bank["public_hash"]))
        if observation["phase"] == "combat" or public_hash(observation) != bank["public_hash"]:
            raise SimulatorError("Bank OOC replay mismatch")
        raw = self.episode.env._env
        if raw.bc is not None:
            raise RuntimeError("Bank entry is already initialized")
        actions, display, mapping = raw._action_views()
        position = next((i for i, action in enumerate(display)
                         if action.index == bank["entering_action"]), None)
        if position is None or display[position].description != bank["entering_description"]:
            raise SimulatorError("Original entering action changed")
        # Do not call env.step here: it initializes bc before we can salt gc.seed.
        actions[mapping[position]].execute(raw.gc)
        if raw.bc is not None or raw.gc.screen_state != raw.sts.ScreenState.BATTLE:
            raise RuntimeError("Entering action must set BATTLE without initializing bc")
        return raw, raw.gc.copy()

    def _scratch(self, raw, base, salt, bank, policy=None, max_decisions=600):
        sts = raw.sts
        wrapper = self.episode.env
        # Scratch code replaces these histories before using them. Keep the
        # original objects and counters, including on native/inference failure.
        saved = [(obj, dict(vars(obj))) for obj in (raw, wrapper, self.episode)]
        original_exit = sts.BattleContext.exit_battle
        ended = {}

        def record_exit(bc, context):
            ended.update(win=bc.outcome == sts.BattleOutcome.PLAYER_VICTORY,
                         hp_after=int(bc.player_cur_hp))
            # Boss/Portal entry has map row 15 (outside rows 0..14). A decided
            # outcome prevents any post-fight observation requesting map moves.
            context.outcome = (sts.GameOutcome.PLAYER_VICTORY if ended["win"]
                               else sts.GameOutcome.PLAYER_LOSS)
            context.screen_state = sts.ScreenState.MAP_SCREEN

        try:
            raw.gc, raw.bc = base.copy(), None
            raw.gc.seed = (int(base.seed) + 1_000_003 * (salt + 1)) % (1 << 63)
            raw._combat_recent_actions, raw._combat_history_turn = [], None
            self.episode.steps, self.episode.finished = 0, False
            # Reset wrapper-owned labels/history so previous salts cannot leak in.
            for key, empty in (("_raw_history", []), ("_public_history", []), ("_selection_labels", {})):
                if key in vars(wrapper):
                    setattr(wrapper, key, empty)
            sts.BattleContext.exit_battle = record_exit
            raw.advance_to_decision()
            observation = self.episode._observe()
            actual_hash = public_hash(observation)
            if salt == -1 and actual_hash != bank["first_combat_hash"]:
                raise SimulatorError("Unsalted first-combat public hash mismatch")
            if policy is None:
                return {"first_combat_hash": actual_hash}
            entry_hp, decisions = player_hp(observation), 0
            while not ended and observation["phase"] == "combat" and decisions < max_decisions:
                action = policy.choose_action(observation)
                observation, _, terminated, truncated, _ = self.episode.step(action)
                decisions += 1
                if terminated or truncated:
                    break
            ub = bool(raw._undefined_behavior_evoked or
                      (raw.bc is not None and raw.bc.undefined_behavior_evoked))
            valid = bool(ended) and not ub
            hp_after = (ended["hp_after"] if ended.get("win") else 0) if valid else None
            return {"salt": salt, "win": ended.get("win") if valid else None,
                    "entry_hp": entry_hp, "source_entry_hp": bank["entry_hp"], "hp_after": hp_after,
                    "hp_lost": entry_hp - hp_after if valid else None,
                    "decisions": decisions, "cut": not bool(ended), "ub": ub,
                    "first_combat_hash": actual_hash}
        finally:
            sts.BattleContext.exit_battle = original_exit
            for obj, state in saved:
                vars(obj).clear()
                vars(obj).update(state)

    def fresh_fight(self, bank, salt, champion, combat=None, expected_policy=None):
        if salt < -1:
            raise ValueError("salt must be -1 (unsalted) or nonnegative")
        # Includes immutable bank source, provenance, hashes and full start/task
        # identity. One entry bounds memory; scheduling keeps this start's salts
        # and policies on the same child. Never cache a post-init mutable battle.
        key = public_hash({"prepared_start": bank})
        prepared = getattr(self, "_prepared_start", None)
        if prepared is None or prepared[0] != key:
            self._prepared_start = None
            raw, base = self._preinitialization(bank)
            self._scratch(raw, base, -1, bank)
            self._prepared_start = key, raw, base
        else:
            _, raw, base = prepared
        policy = _policy(champion, combat)
        if expected_policy is not None:
            for key in ("champion", "combat"):
                if policy.identity[key] != expected_policy[key]:
                    raise RuntimeError("Panel checkpoint identity changed since launch")
        return self._scratch(raw, base, salt, bank, policy)


def bank_files(paths):
    from sts_rl.fresh_cohort import validate_cohorts
    cohort = validate_cohorts(paths)
    return [*cohort["inputs"], *cohort["shards"]]


def load_bank(paths, *, world_start=None, n=None):
    from sts_rl.fresh_cohort import validate_cohorts
    cohort = validate_cohorts(paths, world_start=world_start, n=n)
    return _bank_rows(cohort)


def _bank_rows(cohort):
    from sts_rl.fresh_cohort import read_shard
    from sts_rl.fresh_collection import task
    rows = []
    if cohort["config"].get("teacher_plays") is True:
        raise ValueError("Benchmark cohort with teacher_plays true cannot be used for a fight bank")
    if not cohort["config"].get("bank"):
        raise ValueError("Cohort did not collect a fight bank")
    for file in cohort["shards"]:
        shard = read_shard(file)
        for row in shard["bank"]:
            size = len(row["prefix"])
            if (row["world_seed"] != shard["world_seed"] or row["source"] != shard["source"] or
                row["expected_provenance"] != shard["expected_provenance"] or
                row["prefix"] + [row["entering_action"]] != shard["action_prefix"][:size+1] or
                public_hash(row["task"]) != public_hash(task(row["world_seed"], row["expected_provenance"],
                                    row["prefix"], row["public_hash"]).to_dict())):
                raise ValueError("Bank/shard world, source or action-prefix mismatch")
        rows.extend(shard["bank"])
    seen = set()
    for row in rows:
        if row.get("protocol") != PROTOCOL or not 92_000_000_000 <= row["world_seed"] < 93_000_000_000:
            raise ValueError("Fresh bank must use separate 92e9 source worlds")
        if row["expected_provenance"]["environment_version"] != "0.1.5":
            raise ValueError("Fresh bank requires environment 0.1.5")
        key = row["world_seed"], row["start_id"]
        if key in seen:
            raise ValueError("Duplicate bank start")
        seen.add(key)
        if public_hash(row["observation"]) != row["public_hash"] or row["source"]["protocol"] != PROTOCOL:
            raise ValueError("Bank observation/source identity mismatch")
        if row["start_id"] != len(row["prefix"]):
            raise ValueError("Bank prefix/start identity mismatch")
        entry = next((a for a in row["observation"]["legal_actions"] if a["index"] == row["entering_action"]), None)
        if entry is None or entry["description"] != row["entering_description"]:
            raise ValueError("Bank entering action/description mismatch")
    return rows


def paired_summary(records, frozen):
    """Summarize paired cells; legacy grids without C5 metadata are diagnostic."""
    expected = {(r["world_seed"], r["start_id"], r["salt"], p)
                for r in frozen["grid"] for p in ("champion", "replacement")}
    actual = [(r["world_seed"], r["start_id"], r["salt"], r["policy"]) for r in records]
    if len(set(actual)) != len(actual) or not set(actual) <= expected:
        raise ValueError("Duplicate or unexpected panel grid result")
    grid_groups = {(r["world_seed"], r["start_id"], r["salt"]): r["group"] for r in frozen["grid"]}
    if any(r["group"] != grid_groups[(r["world_seed"], r["start_id"], r["salt"])] or
           ("grid_hash" in r and r["grid_hash"] != frozen.get("grid_hash")) for r in records):
        raise ValueError("Panel group/grid hash mismatch")
    valid = [r for r in records if not (r.get("fault") or r.get("cut") or r.get("ub") or
                                       r.get("censored") or r.get("programming_error"))]
    results_complete = len(valid) == len(expected)
    complete = results_complete and bool(frozen.get("coverage_sufficient", False))
    # Cell completeness alone cannot establish the provenance of legacy input.
    confidence_allowed = complete and all(frozen.get(key) is not None for key in _GRID_FIELDS)
    diagnostic_only = not confidence_allowed
    groups = {}
    for group in sorted(frozen.get("targets") or {r["group"] for r in frozen["grid"]}):
        # The unsalted control is reported separately; no mixture with resamples.
        for control in (False, True):
            rows = [r for r in valid if r["group"] == group and (r["salt"] == -1) == control]
            paired = defaultdict(dict)
            for r in rows:
                paired[(r["world_seed"], r["start_id"], r["salt"])][r["policy"]] = r
            pairs = [(key, row) for key, row in paired.items() if len(row) == 2]
            if any(row["champion"].get("first_combat_hash") != row["replacement"].get("first_combat_hash")
                   for _, row in pairs):
                raise ValueError("Paired policies received different first-combat states")
            metrics = {}
            for metric in ("win", "hp_lost"):
                clusters, baseline, replacement = defaultdict(list), defaultdict(list), defaultdict(list)
                for (world, _, _), pair in pairs:
                    clusters[world].append(float(pair["replacement"][metric]) - float(pair["champion"][metric]))
                    baseline[world].append(float(pair["champion"][metric]))
                    replacement[world].append(float(pair["replacement"][metric]))
                # Equal source-world estimand; starts and salts within a world are
                # dependent. Sample SD of world means / sqrt(number of worlds).
                means = [statistics.mean(values) for values in clusters.values()]
                se = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else None
                metrics[metric] = {"diagnostic_only": diagnostic_only,
                    "paired_difference": statistics.mean(means) if means else None,
                    "worlds": len(means), "se": se, "approx_80pct_power_mde": 2.8 * se if se is not None else None,
                    "champion_mean": statistics.mean(statistics.mean(v) for v in baseline.values()) if pairs else None,
                    "replacement_mean": statistics.mean(statistics.mean(v) for v in replacement.values()) if pairs else None}
            metrics["diagnostic_only"] = diagnostic_only
            groups[group + (":unsalted" if control else ":resampled")] = metrics
    return {"protocol": PROTOCOL, **_grid_fields(frozen), "grid_complete": complete,
            "results_complete": results_complete, "diagnostic_only": diagnostic_only,
            "confidence_claims_allowed": confidence_allowed,
            "expected_results": len(expected), "received": len(records), "valid": len(valid),
            "faults": sum(bool(r.get("fault")) for r in records),
            "cuts": sum(bool(r.get("cut")) for r in records), "ub": sum(bool(r.get("ub")) for r in records),
            "censored": sum(bool(r.get("censored") or r.get("programming_error")) for r in records),
            "unstarted": len(expected)-len(records), "missing_results": len(expected)-len(records),
            "invalid_results": len(records)-len(valid), "groups": groups,
            "method": "Equal-source-world paired means; SE=SD(world means)/sqrt(n worlds); MDE=2.8*SE. "
                      "Approximate normal 80% power at two-sided 5%; no final claims for incomplete grid "
                      "or missing manifest metadata.",
            "records": records}


def _grid_fields(frozen):
    return {**{key: frozen.get(key) for key in _GRID_FIELDS}, "hp_definition": dict(HP_DEFINITION)}


def _panel_bank(config):
    from sts_rl.fresh_cohort import validate_cohorts
    cohort = validate_cohorts(config["bank"], world_start=config.get("bank_world_start"), n=config.get("bank_n"))
    return cohort, _bank_rows(cohort), bank_identity(cohort)


def freeze_panel_grid(config):
    """Model-free entry point; --out is a new immutable manifest file."""
    _, rows, identity = _panel_bank(config)
    _, frozen = freeze_grid(rows, config.get("starts_per_group", DEFAULT_TARGETS), config.get("resamples", 8),
                            bank_identity=identity)
    save_grid(config["out"], frozen)
    return frozen


def _finish_panel(out, summary, source, progress):
    """Panel-specific completion fields without changing the running collector."""
    summary["source"] = source
    summary["cohort_complete"] = summary["grid_complete"]
    failed = bool(summary.get("programming_error") or summary.get("grid_validation_error") or summary["faults"])
    summary.update(event="done", status=("failed" if failed else "completed" if summary["cohort_complete"] else
                         "interrupted" if summary.get("stop_reasons") else "incomplete"))
    if summary["status"] != "completed":
        summary["confidence_claims_allowed"] = False
        summary["diagnostic_only"] = True
        for metrics in summary.get("groups", {}).values():
            metrics["diagnostic_only"] = True
            for key in ("win", "hp_lost"):
                metrics[key]["diagnostic_only"] = True
    summary["run_control"] = progress.control.stats()
    atomic_json(out / "summary.json", summary)
    keys = ("status", "cohort_complete", "confidence_claims_allowed", "diagnostic_only", "grid_complete",
            "faults", "cuts", "censored", "unstarted", "stop_reasons", "valid", "results_complete",
            "missing_results", "invalid_results", "expected_results", "received", "grid_input")
    fields = {key: summary[key] for key in keys}
    fields.update({key: summary[key] for key in (*_GRID_FIELDS, "hp_definition")})
    atomic_json(out / "completion.json", {"protocol": PROTOCOL, "event": "done", **fields,
                "source": source, "summary": file_identity(out / "summary.json"),
                "run_control": summary["run_control"]})
    progress.emit("done", **fields)


def run_panel(config, *, worker_factory=None):
    from sts_rl.common import WORKER_CONFIG
    if config["workers"] < 1 or not math.isfinite(config["timeout"]) or config["timeout"] <= 0:
        raise ValueError("Require positive workers and finite positive timeout")
    config = {"starts_per_group": dict(DEFAULT_TARGETS), "resamples": 8, **config}
    cohort, rows, identity = _panel_bank(config)
    grid_input = None
    if config.get("grid"):
        selected, frozen, grid_input = load_grid(config["grid"], rows, config["starts_per_group"],
                                                config["resamples"], bank_identity=identity)
    else:
        selected, frozen = freeze_grid(rows, config["starts_per_group"], config["resamples"], bank_identity=identity)
    # These fields reach begin_run via prepare_run, before any model is scored.
    config = {**config, **_grid_fields(frozen), "bank_identity": identity, "grid_input": grid_input}
    policy = policy_factory(config["checkpoint"], combat=config["combat"], original_policy=StudyPolicy)
    if grid_input is not None and file_identity(config["grid"]) != grid_input:
        raise ValueError("Frozen grid input changed before launch")
    source = prepare_run(config["out"], config, policy,
                         inputs=[*cohort["inputs"], *cohort["shards"], *([config["grid"]] if grid_input else [])])
    out = Path(config["out"])
    source_ref = file_identity(out / "receipt.json")
    save_grid(out / "grid.json", frozen)
    control = RunControl(out, config.get("deadline_utc"))
    progress, records = Progress(out, control), []
    # Assign a complete start to one child; every individual policy/salt remains
    # a finite RPC with a control boundary. Its prepared context is reused.
    import threading
    lock, cursor = threading.Lock(), 0
    stop_reasons = set()
    starts = {(r["world_seed"], r["start_id"]): r for r in selected}
    start_items = list(starts)

    def loop():
        nonlocal cursor
        worker = (worker_factory or FreshPanelWorker)(config={**WORKER_CONFIG, "fast_observations": False},
                                                     timeout=config["timeout"])
        try:
            while True:
                try:
                    control.check()
                except Interrupted as exc:
                    with lock:
                        stop_reasons.add(str(exc))
                    return
                with lock:
                    if cursor == len(start_items):
                        return
                    start, cursor = start_items[cursor], cursor + 1
                bank = starts[start]
                for salt, name, combat in ((salt, name, combat) for salt in frozen["salts"]
                        for name, combat in (("champion", None), ("replacement", config["combat"]))):
                    item = {"world_seed": start[0], "start_id": start[1], "group": bank["group"], "salt": salt}
                    programming_error = None
                    try:
                        control.check()
                        expected_policy = {**policy.identity, "combat": policy.identity["combat"] if combat else None}
                        result = worker.fresh_fight(bank, item["salt"], config["checkpoint"], combat, expected_policy)
                    except SimulatorError as exc:
                        result = {"fault": str(exc), "last_fault_path": worker.last_fault_path}
                    except Interrupted as exc:
                        result = {"censored": str(exc)}
                        with lock:
                            stop_reasons.add(str(exc))
                    except Exception as exc:
                        programming_error = exc
                        result = {"programming_error": f"{type(exc).__name__}: {exc}"}
                    row = {**result, **item, "policy": name, "source_ref": source_ref, "grid_hash": frozen["grid_hash"]}
                    invalid = any(result.get(k) for k in ("fault", "cut", "ub", "censored", "programming_error"))
                    path = out / ("quarantine" if invalid else "results") / f"{item['world_seed']}-{item['start_id']}-{item['salt']}-{name}.json"
                    atomic_json(path, row)
                    with lock:
                        records.append(row)
                    progress.add(decisions=result.get("decisions", 0),
                                 faults=int(bool(result.get("fault"))),
                                 censored=int(bool(result.get("censored") or result.get("cut") or result.get("ub") or result.get("programming_error"))),
                                 completed=int(result.get("win") is not None))
                    progress.emit("fight", **item, policy=name)
                    if programming_error is not None:
                        raise programming_error
                    if result.get("censored"):
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
        try:
            summary = paired_summary(records, frozen)
        except Exception as exc:
            error = error or exc
            summary = {"protocol": PROTOCOL, **_grid_fields(frozen), "grid_complete": False,
                       "confidence_claims_allowed": False, "diagnostic_only": True, "results_complete": False,
                       "valid": 0, "received": len(records), "expected_results": len(frozen["grid"])*2,
                       "missing_results": len(frozen["grid"])*2-len(records), "invalid_results": len(records),
                       "records": records, "grid_validation_error": str(exc),
                       "faults": sum(bool(r.get("fault")) for r in records),
                       "cuts": sum(bool(r.get("cut")) for r in records),
                       "censored": sum(bool(r.get("censored") or r.get("programming_error")) for r in records),
                       "unstarted": len(frozen["grid"])*2-len(records)}
        summary["stop_reasons"] = sorted(stop_reasons)
        summary["grid_input"] = grid_input
        if error:
            summary.update(programming_error=f"{type(error).__name__}: {error}", confidence_claims_allowed=False)
        _finish_panel(out, summary, source, progress)
    if error:
        raise error
    return summary
