"""Advance the fixed study using the power-aware queue. STOP prevents new submissions."""
import fcntl
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from study import CHAMPION, CONFIG, PREFIX, ROOT, RUN, SHA256, submit, train

sys.path.insert(0, str(ROOT / "src"))
from sts_rl.paired_endpoints import compare, digest

GATE_WORLDS = {8: 149_000_000_000, 16: 150_000_000_000}
LABEL_WORLDS = {8: 90_850_008_000, 16: 90_850_016_000}
BANK_WORLDS = {8: 92_850_008_000, 16: 92_850_016_000}
FROZEN = RUN / "control"
QUEUE = ROOT / "data/rl_runs/queue.json"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_bytes())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)


def lines(path):
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_bytes().split(b"\n")[:-1] if row]


def stamp(t):
    return datetime.fromtimestamp(t, ZoneInfo("Europe/London")).strftime("%d %b %H:%M %Z")


def package(active, specialist, destination):
    """Change just the boss branch. Preserve the milestone optimizer, RNG and normalizer."""
    import torch
    from sts_policy.model_agent import Act3BossActorCritic, policy_from_payload
    from sts_rl.common import save_atomic
    from sts_rl.controlled_policy import replacement_from_payload
    base = torch.load(active, map_location="cpu", weights_only=False)
    head = torch.load(specialist, map_location="cpu", weights_only=False)
    old = policy_from_payload(base)
    replacement = replacement_from_payload(head, "combat")
    model = Act3BossActorCritic(old.combat, old.ooc, replacement)
    meta = {**base["metadata"], "model_kind": model.kind,
            "act3_boss_refit": {"active": digest(active), "specialist": digest(specialist)}}
    payload = dict(base, model_state=model.state_dict(), model_config=model.metadata(), metadata=meta)
    policy_from_payload(payload)
    save_atomic(payload, destination)


class Monitor:
    def __init__(self, run=RUN, prefix=PREFIX, *, label_worlds=LABEL_WORLDS, bank_worlds=BANK_WORLDS,
                 gate_worlds=GATE_WORLDS, probe=RUN / "probe"):
        self.run, self.prefix, self.probe = run, prefix, probe
        self.label_worlds, self.bank_worlds, self.gate_worlds = label_worlds, bank_worlds, gate_worlds
        self.events = {r["key"]: r for r in lines(run / "progress.jsonl")}
        self.result_cache = {}

    def note(self, key, message, t=None, **details):
        if key in self.events:
            return
        event = dict(key=key, t=t if t is not None else time.time(), message=message, **details)
        with (self.run / "progress.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        self.events[key] = event
        self.render_progress()
        print(message, flush=True)

    def render_progress(self):
        entries = sorted(self.events.values(), key=lambda e: e["t"], reverse=True)
        write(ROOT / "research/log.md", "# Act 3 boss refit progress\n\n" +
              "\n".join(f"- {stamp(e['t'])} — {e['message']}" for e in entries if not e["key"].startswith(PREFIX + "smoke")) + "\n")

    def training_progress(self, folder):
        from sts_rl.common import train_tasks
        for r in lines(folder / "log.jsonl"):
            i = r.get("iteration", 0)
            event = r["event"]
            key = f"{folder.name}-{event}-{i}"
            if key in self.events:
                continue
            if event == "train":
                start = CONFIG["train_world_start"] + CONFIG["seed"] * 500_000_000 + i * 10_000_000
                with gzip.open(folder / f"collection_{i:04d}.json.gz", "rt") as f:
                    seeds = [w["seed"] for w in json.load(f)["tasks"]]
                if seeds != [w.seed for w in train_tasks(start, CONFIG["games"])]:
                    raise ValueError(f"World roster mismatch at {folder.name} iteration {i}")
                self.note(key, f"{folder.name} finished update {i}/24 on {len(seeds):,} matched worlds. "
                          f"Sampled wins {r['collect_wins']:,}, faults {r['faults']}, "
                          f"elapsed {r['iteration_seconds']/60:.1f} min.", r["t"])
            elif event == "eval":
                self.note(key, f"{folder.name} development evaluation at update {i}: "
                          f"{r['wins']:,}/{r['n']:,} wins, {r['faults']} faults.", r["t"])
            elif event in ("start", "resume"):
                self.note(key, f"{folder.name} started at update {i} with learning rate 0.0001.", r["t"])

    def job(self, name, directory, tool, args, lane, *, summary="summary.json"):
        """Retry a bounded number of interrupted jobs, retaining each attempt's outputs."""
        for attempt in range(1, 4):
            suffix = "" if attempt == 1 else f"-retry{attempt-1}"
            key = name + suffix
            folder = directory.with_name(directory.name + suffix)
            job = self.jobs.get(self.prefix + key)
            if not job:
                if self.on_battery:
                    return None
                submit(key, lane, tool, [*args, "--out", folder], prefix=self.prefix)
                self.note(key + "-submit", f"Queued {key}.")
                return None
            state = job["status"]
            if state in ("pending", "running", "paused"):
                if state == "running" and folder.is_dir():
                    self.job_progress(key, folder, args)
                return None
            if state == "cancelled":
                raise RuntimeError(f"{key} was cancelled")
            path = folder if summary is None else folder / summary
            if state == "done" and path.is_file():
                if path not in self.result_cache:
                    result = read(path)
                    if result.get("status", "completed") == "completed" and not result.get("error"):
                        self.result_cache[path] = True
                if path in self.result_cache:
                    return folder
            self.note(key + "-incomplete", f"{key} ended without a complete result. Its outputs were retained.")
        raise RuntimeError(f"{name} failed three attempts; inspect its preserved records")

    def job_progress(self, key, folder, args):
        # Read only the tail: label collection writes hundreds of heartbeats.
        path = folder / "progress.jsonl"
        if path.is_file() and "--n" in args:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                tail = stream.read().split(b"\n")[1:-1]
            if tail:
                r = json.loads(tail[-1])
                n = args[args.index("--n")+1]
                completed = r.get("recorded", r.get("completed", 0))
                quarter = min(4, 4 * completed // n)
                if quarter:
                    self.note(f"{key}-progress-{quarter}", f"{key} returned {completed:,} of {n:,} worlds.")
        status = folder / "status.json"
        if key.startswith("fit-") and status.is_file():
            info = read(status)
            step = info.get("updates", info.get("step", 0))
            if step:
                self.note(f"{key}-steps-{step//1000}", f"{key} reached {step:,} optimizer updates.")

    def evaluate_one(self, name, label, checkpoint, start, count, lane):
        args = ["--checkpoint", checkpoint, "--world-start", start, "--n", count,
                "--processes", 7, "--games-per-process", 8, "--progress-every", 256]
        return self.job(name + "-" + label, self.run / name / label, "scripts/rl/fast_eval.py",
                        args, f"joint-a3-eval-{lane}")

    def paired(self, name, labels, checkpoints, start, count):
        result = self.run / (name + ".json")
        if result.exists():
            return read(result)
        folders = []
        for index, (label, ck) in enumerate(zip(labels, checkpoints)):
            folders.append(self.evaluate_one(name, label, ck, start, count, index))
        if not all(folders):
            return None
        stats = compare(*folders, start, *labels, *checkpoints, count)
        stats["checkpoint_sha256"] = {label: digest(ck) for label, ck in zip(labels, checkpoints)}
        stats["folders"] = list(map(str, folders))
        if digest(checkpoints[0]) == digest(checkpoints[1]) and (stats["a_only"] or stats["b_only"]):
            raise ValueError("Identical endpoint weights produced different completed wins on paired worlds")
        write(result, stats)
        self.note(name + "-result", f"{name} finished. {labels[0]} won {stats['first']['wins']:,}, "
                  f"{labels[1]} won {stats['second']['wins']:,}. {stats['paired_completed']:,}/{count:,} "
                  f"complete pairs. Full-cohort gain bounds {stats['full_cohort_gain_bounds_pp']}.")
        return stats

    def gate(self, i, active):
        folder = self.run / f"refit{i}"
        folder.mkdir(exist_ok=True)
        decision = folder / "decision.json"
        if decision.exists():
            saved = read(decision)
            if saved["active_sha256"] != digest(active):
                raise ValueError("Active policy changed after gate decision")
            return saved
        common = ["--checkpoint", active, "--n", 512, "--workers", 6]
        labels = self.job(f"labels-{i}", folder / "labels", "scripts/rl/fresh_collect.py",
                          [*common, "--mode", "collect", "--world-start", self.label_worlds[i],
                           "--teacher-probabilities", '{"BOSS3":1}', "--record-search", "--vote-rule",
                           "pooled_mean", "--samples", 24, "--simulations", 1000, "--self-consistency-every", 0],
                          "joint-a3-search", summary="completion.json")
        bank = self.job(f"bank-{i}", folder / "bank", "scripts/rl/fresh_collect.py",
                        [*common, "--mode", "eval", "--bank", "--world-start", self.bank_worlds[i]],
                        "joint-a3-bank", summary="completion.json")
        if bank is None:
            return None
        panel_args = ["--bank", bank, "--bank-world-start", self.bank_worlds[i], "--bank-n", 512,
                      "--starts-per-group", '{"BOSS3":192}', "--resamples", 1]
        grid = self.job(f"grid-{i}", folder / "grid.json", "scripts/rl/fresh_panel.py",
                        [*panel_args, "--freeze-grid-only"], "joint-a3-bank", summary=None)
        if grid is None or labels is None:
            return None
        self.note(f"grid-{i}-complete", f"Update {i} froze {read(grid)['coverage']['BOSS3']} independent Act 3 boss starts.")
        if not read(grid)["coverage_sufficient"]:
            raise RuntimeError(f"Insufficient saved Act 3 boss entries at iteration {i}")
        if f"labels-{i}-complete" not in self.events:
            summary = read(labels / "summary.json")
            self.note(f"labels-{i}-complete", f"Update {i} collected {summary['combat_labels']:,} boss labels "
                      f"from {summary['completed']}/512 completed games.")
        probe = self.probe
        if read(probe / "completion.json")["status"] != "completed":
            raise ValueError("Unlabelled movement probes are incomplete")
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        fit = self.job(f"fit-{i}", folder / "fit", "scripts/rl/fresh_train.py",
                       ["--data", labels, "--probe", probe, "--phase", "combat", "--architecture", "old",
                        "--old-size", "large", "--lr", .001, "--batch", 128, "--target-mode", "pooled_values",
                        "--value-scale", 100, "--steps", 12000, "--min-steps", 2000, "--eval-every", 100,
                        "--patience", 6, "--seed", 1, "--device", "mps", "--defer-test", "--deadline-utc", deadline],
                       "joint-a3-fit")
        if fit is None:
            return None
        info = read(fit / "summary.json")
        self.note(f"fit-{i}-complete", f"Update {i} specialist fit on {info['actual_fit_rows']:,} "
                  f"training choices. Selected step {info['best_step']:,} on {info['validation_rows']:,} validation choices.")
        fitted = fit / "best.pt"
        if digest(fitted) != read(fit / "summary.json")["selected_checkpoint"]["sha256"]:
            raise ValueError("Validation-selected head changed")
        panel = self.job(f"fights-{i}", folder / "fights", "scripts/rl/fresh_panel.py",
                         [*panel_args, "--grid", grid, "--checkpoint", active, "--combat", fitted,
                          "--workers", 6], "joint-a3-bank", summary="completion.json")
        if panel is None:
            return None
        fights = read(panel / "summary.json")
        if not (fights["grid_complete"] and fights["coverage_sufficient"] and
                fights["grid_hash"] == read(grid)["grid_hash"]):
            raise ValueError("Incomplete or changed fight panel")
        for group, summary in fights["groups"].items():
            wins = summary["win"]
            self.note(f"fight-{i}-{group}", f"Update {i} {group} fight gate: "
                      f"active {wins['champion_mean']:.1%}, specialist {wins['replacement_mean']:.1%} "
                      f"on {wins['worlds']} identical entries.")
        fight_gain = fights["groups"]["BOSS3:unsalted"]["win"]["paired_difference"]
        adopt = False
        candidate = folder / "candidate.pt"
        full = None
        if fight_gain > 0:
            if not candidate.exists():
                package(active, fitted, candidate)
            full = self.paired(f"gate{i}", ("active", "specialist"), (active, candidate), self.gate_worlds[i], 4096)
            if full is None:
                return None
            adopt = full["full_cohort_gain_bounds_pp"][0] > 0
        outcome = dict(iteration=i, adopted=adopt, active_sha256=digest(active), specialist_sha256=digest(fitted),
                       checkpoint=str(candidate if adopt else active), checkpoint_sha256=digest(candidate if adopt else active),
                       fight_gain=fight_gain, fight_panel=str(panel), full_game_bounds_pp=full["full_cohort_gain_bounds_pp"] if full else None)
        write(decision, outcome)
        self.note(f"decision-{i}", f"Update {i}: " + ("adopted the specialist for Act 3 boss combat."
                  if adopt else "kept the active combat policy.") +
                  (" The full-game gate was skipped because the saved-fight gate did not pass." if full is None else ""))
        return outcome

    def resume(self, name, folder, source, target, template=None):
        job = self.jobs.get(PREFIX + name)
        if job is None:
            folder.mkdir(exist_ok=True)
            if not (folder / "latest.pt").exists():
                tmp = folder / "latest.copy"
                shutil.copyfile(source, tmp)
                tmp.replace(folder / "latest.pt")
            train(name, folder, template or source, target, True)
            self.note(name + "-submit", f"Queued matched joint strategy training through update {target}.")
        else:
            self.check_trainer(name, folder, template or source, target)
        return folder

    def check_trainer(self, name, folder, template, target):
        for attempt in range(3):
            key = name if attempt == 0 else f"{name}-resume{attempt}"
            job = self.jobs.get(self.prefix + key)
            if job is None:
                if self.on_battery:
                    return
                import torch
                saved = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
                if (not 0 < saved["iteration"] < target or not saved["optimizer_state"] or
                        not saved["rng_state"] or "trainer_state" not in saved):
                    raise RuntimeError(f"Cannot safely resume {name} from its latest checkpoint")
                train(key, folder, template, target, True, prefix=self.prefix)
                self.note(key + "-submit", f"Resuming {name} from saved update {saved['iteration']} after interruption.")
                return
            state = job["status"]
            if state in ("pending", "running", "paused"):
                return
            if state == "done" and (folder / f"iter_{target:04d}.pt").is_file():
                return
            if state == "cancelled":
                raise RuntimeError(f"{key} was cancelled")
        raise RuntimeError(f"{name} needs inspection after two recovery attempts")

    def tick(self):
        if (RUN / "STOP").exists():
            return "stopped"
        self.jobs = {r["name"]: r for r in read(QUEUE)["runs"]
                     if r["name"].startswith(PREFIX) and not r["name"].startswith(PREFIX + "smoke")}
        probe = RUN / "probe/completion.json"
        if probe.exists() and "probe-finished" not in self.events:
            info = read(probe)
            self.note("probe-finished", f"Collected {info['completed']}/32 movement-probe worlds "
                      f"with {info['faults']} faults.", probe.stat().st_mtime)
        runner = read(QUEUE.with_name("runner_status.json"))
        self.on_battery = runner["on_battery"]
        if time.time() - runner["heartbeat"] > 150:
            return "waiting for queue heartbeat"
        for name, j in self.jobs.items():
            state = j["status"]
            transition = j.get("resumed_at", j.get("started", j["added"]))
            if state in ("paused", "running"):
                self.note(f"{name}-{state}-{int(transition)}", f"{name} is {state}" +
                          (" on battery." if state == "paused" else "."), transition if state == "running" else None)
        for folder in [FROZEN, *RUN.glob("joint_after_*")]:
            self.training_progress(folder)
        self.check_trainer("control", FROZEN, CHAMPION, 24)
        if self.on_battery:
            return "paused on battery"
        fixed_endpoint = FROZEN / "iter_0024.pt"
        if fixed_endpoint.exists() and not (RUN / "primary.json").exists():
            self.evaluate_one("primary", "control", fixed_endpoint, 152_000_000_000, 16_384, 0)
            # paired() may visit the same submission in this tick.
            self.jobs = {r["name"]: r for r in read(QUEUE)["runs"] if r["name"].startswith(PREFIX)}
        current = FROZEN
        for i, target in ((8, 16), (16, 24)):
            active = current / f"iter_{i:04d}.pt"
            if not active.exists():
                return f"training to update {i}"
            outcome = self.gate(i, active)
            if outcome is None:
                return f"update {i} specialist and gates"
            source = Path(outcome["checkpoint"])
            if digest(source) != outcome["checkpoint_sha256"]:
                raise ValueError("Frozen gate outcome changed")
            if outcome["adopted"]:
                name = f"joint_after_{i}" + ("_joint" if current != FROZEN else "_control")
                current = self.resume(name, RUN / name, source, target)
            elif i == 16 and current != FROZEN:
                initial = Path(read(RUN / "refit8/decision.json")["checkpoint"])
                self.resume("joint-continue", current, active, target, initial)
        endpoints = (FROZEN / "iter_0024.pt", current / "iter_0024.pt")
        if not all(p.exists() for p in endpoints):
            return "training to update 24"
        names = ("control", "joint")
        primary = self.paired("primary", names, endpoints, 152_000_000_000, 16_384)
        if primary is None:
            return "paired endpoint evaluation"
        j = int(primary["full_cohort_gain_bounds_pp"][0] > 0)
        selected = endpoints[j]
        choice = dict(arm=names[j], checkpoint=str(selected), sha256=digest(selected))
        if choice["sha256"] != primary["checkpoint_sha256"][names[j]]:
            raise ValueError("Selected endpoint changed")
        selection = RUN / "selection.json"
        if not selection.exists():
            write(selection, choice)
        if read(selection) != choice or digest(CHAMPION) != SHA256:
            raise ValueError("Frozen confirmation inputs changed")
        self.note("selected", f"Froze the {names[j]} endpoint for independent confirmation.")
        result = self.paired("confirmation", ("incumbent", names[j]), (CHAMPION, selected), 153_000_000_000, 16_384)
        if result is None:
            return "independent confirmation"
        qualifies = result["worst_case_ci95_pp"][0] > 0
        verdict = dict(**choice, confirmation_passed=qualifies, starting_champion=SHA256)
        write(RUN / "verdict.json", verdict)
        from finish import record_finish
        record_finish(verdict, primary, result)
        self.note("verdict", f"The {names[j]} endpoint " +
                  ("passed confirmation." if qualifies else "did not pass confirmation; the incumbent stays champion."))
        return "complete"


def main():
    with (RUN / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor()
        while True:
            try:
                stage = monitor.tick()
            except Exception as error:
                monitor.note("attention-" + str(error), f"Study needs attention: {error}")
                write(RUN / "monitor-status.json", dict(stage="attention", error=repr(error), time=time.time()))
                raise
            write(RUN / "monitor-status.json", dict(stage=stage, time=time.time(), pid=os.getpid()))
            if stage in ("stopped", "complete"):
                break
            time.sleep(25)


if __name__ == "__main__":
    main()
