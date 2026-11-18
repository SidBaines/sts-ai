"""After the eight-update study finishes, run eleven matched boss-refit opportunities."""
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import time

from heartbeat import EVERY_TWO
from monitor import BANK_WORLDS, GATE_WORLDS, LABEL_WORLDS, Monitor, compare, digest, lines, read, stamp, write
from study import CHAMPION, CONFIG, ROOT, RUN, SHA256, train

PREFIX = "a3-e2-"
GATES = tuple(range(2, 24, 2))
LABELS = {i: LABEL_WORLDS.get(i, 90_860_000_000 + i * 1000) for i in GATES}
BANKS = {i: BANK_WORLDS.get(i, 92_860_000_000 + i * 1000) for i in GATES}
GAMES = {i: GATE_WORLDS.get(i, 154_000_000_000 + i * 1_000_000) for i in GATES}
QUEUE = RUN.parent / "queue.json"
TRAIN = EVERY_TWO / "train"
EARLY, PRIMARY, CONFIRMATION = 158_000_000_000, 160_000_000_000, 161_000_000_000


def previous_endpoints():
    """Resolve the actual gated endpoint even if frozen combat won selection."""
    if read(RUN / "monitor-status.json")["stage"] != "complete":
        return None
    primary = read(RUN / "primary.json")
    confirmation = read(RUN / "confirmation.json")
    verdict = read(RUN / "verdict.json")
    if verdict["starting_champion"] != SHA256 or digest(CHAMPION) != SHA256:
        raise ValueError("Starting champion changed")
    if (primary["n"], confirmation["n"]) != (16384, 16384):
        raise ValueError("The earlier study has incomplete endpoint panels")
    endpoints = {}
    for name, folder in zip((primary["a"], primary["b"]), primary["folders"]):
        ck = Path(read(Path(folder) / "run.json")["config"]["checkpoint"])
        if digest(ck) != primary["checkpoint_sha256"][name]:
            raise ValueError("Earlier endpoint changed")
        endpoints[name] = str(ck)
    official = Path(verdict["checkpoint"])
    if digest(official) != verdict["sha256"] or confirmation["checkpoint_sha256"][verdict["arm"]] != verdict["sha256"]:
        raise ValueError("Earlier confirmation does not bind the selected endpoint")
    incumbent = CHAMPION
    if verdict["confirmation_passed"]:
        incumbent = CHAMPION.with_name("champion_" + verdict["sha256"][:12] + ".pt")
        if digest(incumbent) != verdict["sha256"]:
            raise ValueError("Promoted incumbent differs from its confirmed checkpoint")
    eight = read(RUN / "refit8/decision.json")
    if digest(Path(eight["checkpoint"])) != eight["checkpoint_sha256"]:
        raise ValueError("Earlier update-8 policy changed")
    return dict(every8=endpoints["joint"], control=endpoints["control"], early8=eight["checkpoint"],
                incumbent=str(incumbent), original_sha256=SHA256,
                prior_confirmation_sha256=digest(RUN / "confirmation.json"))


def traces(path):
    with gzip.open(path, "rt") as stream:
        receipt = json.load(stream)
    return sorted((v["world_seed"], v["action_trace_sha256"], v.get("fault")) for v in receipt["episodes"])


class Cadence(Monitor):
    def __init__(self):
        super().__init__(EVERY_TWO, PREFIX, label_worlds=LABELS, bank_worlds=BANKS, gate_worlds=GAMES,
                         probe=RUN / "probe")
        self.reference = None
        self.previous_events = []

    def render_progress(self):
        if not (self.run / "started.json").exists():
            return
        events = [(e["t"], "Every two", e["message"]) for e in self.events.values()]
        events += [(e["t"], "Every eight / control", e["message"]) for e in self.previous_events
                   if not e["key"].startswith("joint-a3-smoke")]
        content = [f"- {stamp(t)} — {arm}: {message}" for t,arm,message in sorted(events, reverse=True)]
        write(ROOT / "research/log.md", "# Act 3 boss refit progress\n\n" + "\n".join(content) + "\n")

    def open_study(self):
        refs = previous_endpoints()
        if refs is None:
            return False
        self.previous_events = lines(RUN / "progress.jsonl")
        path = self.run / "started.json"
        if path.exists():
            if read(path)["reference"] != refs:
                raise ValueError("Prior study endpoints changed since starting")
        else:
            write(path, dict(t=time.time(), reference=refs, gates=GATES, labels=LABELS, banks=BANKS,
                             gate_worlds=GAMES, early=EARLY, primary=PRIMARY, confirmation=CONFIRMATION))
        self.reference = {k: Path(refs[k]) for k in ("every8", "control", "early8", "incumbent")}
        from cadence_report import starting
        starting()
        self.note("dense-start", "Earlier selection and confirmation finished. Starting from the original "
                  "November champion, with 512 new teacher worlds at each of eleven refit opportunities.")
        return True

    def segment(self, target, template, previous_decision):
        name = f"train-{target:02d}"
        if PREFIX + name not in self.jobs:
            if self.on_battery:
                return False
            if target > 2:
                import torch
                latest = TRAIN / "latest.pt"
                state = torch.load(latest, map_location="cpu", weights_only=False)
                if (state["iteration"] != target - 2 or not state["optimizer_state"]
                        or not state["rng_state"] or "trainer_state" not in state):
                    raise ValueError("Continuation is missing the previous optimizer, iteration or RNG")
                if previous_decision["adopted"]:
                    source = Path(previous_decision["checkpoint"])
                    if digest(source) != previous_decision["checkpoint_sha256"]:
                        raise ValueError("Accepted specialist changed before adoption")
                    tmp = TRAIN / "latest.adopting"
                    shutil.copyfile(source, tmp)
                    if digest(tmp) != previous_decision["checkpoint_sha256"]:
                        raise ValueError("Incomplete adoption copy")
                    tmp.replace(latest)
            train(name, TRAIN, template, target, target > 2, prefix=PREFIX)
            self.note(name + "-submitted", f"Queued strategy updates {target-1} and {target} on the matched PPO worlds.")
            return False
        self.check_trainer(name, TRAIN, template, target)
        attempts = [j for key,j in self.jobs.items() if key == PREFIX + name or key.startswith(PREFIX + name + "-resume")]
        return (any(j["status"] == "done" for j in attempts) and
                (TRAIN / f"iter_{target:04d}.pt").is_file())

    def verify_prefix(self, end):
        for i in range(1, end+1):
            key = f"matching-history-{i}"
            if key in self.events:
                continue
            receipt = f"collection_{i:04d}.json.gz"
            left, right = (traces(folder / receipt) for folder in (RUN / "control", TRAIN))
            if len(left) != CONFIG["games"] or left != right:
                raise ValueError(f"Pre-intervention trajectories differ at update {i}; inspect before attributing effects")
            self.note(key, f"All {len(left):,} pre-intervention trajectories matched the frozen control at update {i}.")

    def panel(self, name, start, count, players, pairs):
        """Evaluate each distinct checkpoint once on the same fixed roster, then compare the specified pairs."""
        result = self.run / (name + "-comparisons.json")
        if result.is_file():
            return read(result)
        by_sha, folders, hashes = {}, {}, {}
        for label, path in players.items():
            sha = digest(path)
            hashes[label] = sha
            if sha not in by_sha:
                by_sha[sha] = self.evaluate_one(name, label, path, start, count, len(by_sha) % 2)
            folders[label] = by_sha[sha]
        if any(f is None for f in folders.values()):
            return None
        comparisons = {}
        for a,b in pairs:
            stats = compare(folders[a], folders[b], start, a, b, players[a], players[b], count)
            if hashes[a] == hashes[b] and (stats["a_only"] or stats["b_only"]):
                raise ValueError("Same evaluated artifact has conflicting paired outcomes")
            stats["checkpoint_sha256"] = {a:hashes[a], b:hashes[b]}
            stats["folders"] = [str(folders[a]), str(folders[b])]
            comparisons[f"{b}_vs_{a}"] = stats
        write(result, comparisons)
        from finish import describe
        for key, stats in comparisons.items():
            self.note(f"{name}-{key}-result", f"{name}, {stats['b']} vs {stats['a']}: {describe(stats)}.")
        return comparisons

    def tick(self):
        if (self.run / "STOP").exists():
            return "stopped"
        if self.reference is None and not self.open_study():
            return "waiting for earlier training, paired evaluation and confirmation"
        runner = read(QUEUE.with_name("runner_status.json"))
        self.on_battery = runner["on_battery"]
        if time.time() - runner["heartbeat"] > 150:
            return "waiting for queue heartbeat"
        self.jobs = {j["name"]: j for j in read(QUEUE)["runs"] if j["name"].startswith(PREFIX)}
        self.training_progress(TRAIN)
        if self.on_battery:
            return "paused on battery"

        prev, template = None, CHAMPION
        early = self.run / "refit8/decision.json"
        early_result = None
        if early.is_file():
            early_result = self.panel("update8", EARLY, 4096,
                                      {"every8":self.reference["early8"], "every2":Path(read(early)["checkpoint"])},
                                      [("every8", "every2")])
        for target in (*GATES, 24):
            if not self.segment(target, template, prev):
                return f"training to update {target}"
            if template == CHAMPION:
                self.verify_prefix(target)
            ck = TRAIN / f"iter_{target:04d}.pt"
            if target in GATES:
                prev = self.gate(target, ck)
                if prev is None:
                    return f"update {target} specialist and gates"
                if prev["adopted"]:
                    template = Path(prev["checkpoint"])
        if early_result is None:
            return "finishing paired update-8 evaluation"
        players = {"every8":self.reference["every8"], "every2":ck, "control":self.reference["control"]}
        primary = self.panel("endpoint", PRIMARY, 16384, players, [("every8","every2"),("control","every2")])
        if primary is None:
            return "paired endpoint evaluation"
        qualifies = all(item["full_cohort_gain_bounds_pp"][0] > 0 for item in primary.values())
        selection = dict(passes_primary=qualifies, checkpoint=str(ck), sha256=digest(ck),
                         primary_sha256=digest(self.run / "endpoint-comparisons.json"),
                         every8_sha256=digest(players["every8"]), incumbent_sha256=digest(self.reference["incumbent"]))
        choice = self.run / "selection.json"
        if choice.exists() and read(choice) != selection:
            raise ValueError("Frozen confirmation decision changed")
        if not choice.exists():
            write(choice, selection)
        confirmed = None
        if qualifies:
            self.note("selected", "The every-two endpoint beat both comparators on the primary panel and is frozen for confirmation.")
            options = {"every8":players["every8"], "every2":ck, "incumbent":self.reference["incumbent"]}
            confirmed = self.panel("confirmation", CONFIRMATION, 16384, options,
                                   [("every8","every2"),("incumbent","every2")])
            if confirmed is None:
                return "independent confirmation"
        else:
            self.note("final-panel", "The every-two endpoint did not pass the primary selection rule for confirmation.")
        promote = bool(confirmed and all(r["worst_case_ci95_pp"][0] > 0 for r in confirmed.values()))
        verdict = dict(**selection, confirmation_run=bool(confirmed), promotion_qualified=promote)
        write(self.run / "verdict.json", verdict)
        from cadence_report import finish
        finish(self, early_result, primary, confirmed, verdict)
        self.note("complete", "All eleven refit decisions and paired schedule comparisons are filed.")
        return "complete"


def main():
    EVERY_TWO.mkdir(exist_ok=True)
    with (EVERY_TWO / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Cadence()
        while True:
            try:
                stage = study.tick()
            except Exception as error:
                study.note("attention-" + str(error), f"Study needs attention: {error}")
                write(EVERY_TWO / "monitor-status.json", dict(stage="attention", pid=os.getpid(), error=repr(error), time=time.time()))
                raise
            write(EVERY_TWO / "monitor-status.json", dict(stage=stage, pid=os.getpid(), time=time.time()))
            if stage in ("complete","stopped"):
                break
            time.sleep(20)


if __name__ == "__main__":
    main()
