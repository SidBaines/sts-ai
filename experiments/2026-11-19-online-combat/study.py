"""Dispatch the dependent pilot and evaluations through the shared power-aware queue."""

from __future__ import annotations
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sts_rl.fresh_study import atomic_json
from sts_rl.online_combat import select_exploration
from sts_rl.paired_endpoints import compare, digest

RUNS = (ROOT / "data" / "rl_runs").resolve()
BASE = RUNS / "online_combat_from_9700"
PARENT = RUNS / "champions" / "champion_9700741dca15.pt"
PARENT_SHA = "9700741dca15cbf451e5ad574a43283f076cccbfdfd1a91f6a3f43375c122c21"
QUEUE = ROOT / "scripts" / "rl" / "queue_runner.py"
EVAL = ROOT / "scripts" / "rl" / "fast_eval.py"
NATIVE = RUNS.parents[1] / "external" / "sts_lightspeed_rl" / "build013_clean"
TARGETS = {"near-greedy": 0.0, "gentle": 0.05, "moderate": 0.12, "broad": 0.22}
RATES = [(1e-5, 1), (3e-5, 1), (1e-4, 1), (3e-5, 2)]
EXTRA = [(1e-6, 1), (3e-6, 1), (3e-4, 1)]


def read(p):
    return json.loads(p.read_text())


def note(message):
    stamp = datetime.now(timezone.utc).strftime("%d %b %H:%M GMT")
    print(stamp, message, flush=True)
    p = ROOT / "research" / "log.md"
    s = p.read_text()
    first, rest = s.split("\n", 1)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        first + "\n\n- " + stamp + " — " + message + "\n" + rest.lstrip("\n")
    )
    os.replace(tmp, p)


def ensure(jobs, name, lane, script, args, marker):
    key = "online-combat-" + name
    recovery = BASE / "recovered-jobs.json"
    if recovery.exists():
        key = read(recovery).get(key, key)
    if key in jobs:
        status = jobs[key]["status"]
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"{key} {status}. See {RUNS / key / 'stdout.log'}")
        if status == "done" and not marker.is_file():
            raise RuntimeError(f"{key} finished without {marker}")
        return marker.is_file()
    if marker.exists():
        raise ValueError(f"unattributed output {marker}")
    cmd = [
        sys.executable,
        str(QUEUE),
        "add",
        key,
        "--lane",
        lane,
        "--",
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        "PYTHONPATH=" + str(ROOT / "src"),
        "STS_LIGHTSPEED_BUILD=" + str(NATIVE),
        sys.executable,
        "-u",
        str(Path(__file__)),
        "exec",
        str(script),
        *map(str, args),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    note("Queued " + name + ".")
    return False


def panel(jobs, key, checkpoints, start, n):
    paths = {}
    for i, (label, player) in enumerate(checkpoints.items()):
        folder = BASE / key / label
        ok = ensure(
            jobs,
            key + "-" + label,
            "online-combat-eval-" + str(i % 2),
            EVAL,
            [
                "--checkpoint",
                player,
                "--world-start",
                start,
                "--n",
                n,
                "--processes",
                7,
                "--games-per-process",
                8,
                "--progress-every",
                256,
                "--out",
                folder,
            ],
            folder / "summary.json",
        )
        if ok:
            paths[label] = folder
    if paths.keys() != checkpoints.keys():
        return None
    return paths


def tick(jobs):
    if not ensure(
        jobs,
        "calibrate",
        "online-combat-pilot",
        HERE / "pilot.py",
        ["calibration", "--out", BASE / "calibration"],
        BASE / "calibration" / "summary.json",
    ):
        return "calibrating exploration"
    ensure(
        jobs,
        "wiring",
        "online-combat-smoke",
        HERE / "train.py",
        ["smoke", "--out", BASE / "validation"],
        BASE / "validation" / "summary.json",
    )
    conditions = dict(TARGETS)
    for i, (label, target) in enumerate(conditions.items()):
        folder = BASE / f"explore-{label}"
        ensure(
            jobs,
            "explore-" + label,
            "online-combat-probe-" + ("a" if i % 2 == 0 else "b"),
            HERE / "pilot.py",
            ["exploration", "--target", target, "--out", folder],
            folder / "summary.json",
        )
    setting = BASE / "exploration-selection.json"
    if not setting.exists():
        if not all(
            (BASE / f"explore-{k}" / "summary.json").exists() for k in conditions
        ):
            return "measuring exploration"
        extra = "very-gentle"
        if (BASE / f"explore-{extra}" / "summary.json").exists():
            conditions[extra] = 0.025
        worlds = {k: read(BASE / f"explore-{k}" / "worlds.json") for k in conditions}
        choice, stats = select_exploration(
            worlds.pop("near-greedy"), worlds, ["moderate", "gentle", "broad", extra]
        )
        if choice is None and extra not in conditions:
            folder = BASE / f"explore-{extra}"
            ensure(
                jobs,
                "explore-" + extra,
                "online-combat-probe-a",
                HERE / "pilot.py",
                ["exploration", "--target", 0.025, "--out", folder],
                folder / "summary.json",
            )
            return "testing gentler exploration"
        if choice is None:
            raise RuntimeError("No exploration setting passed the fixed cost screens.")
        atomic_json(
            setting, dict(choice=choice, target=conditions[choice], comparisons=stats)
        )
        note(
            f"Selected {choice} combat exploration after {len(conditions)} matched 768-world checks."
        )
    target = read(setting)["target"]
    folder = BASE / "fit-data"
    if not ensure(
        jobs,
        "fit-data",
        "online-combat-pilot",
        HERE / "pilot.py",
        ["fit", "--target", target, "--processes", 6, "--out", folder],
        folder / "summary.json",
    ):
        return "collecting the combat learning-rate pilot"
    if not (BASE / "validation" / "summary.json").exists():
        return "checking matched training and weight changes"
    selection = BASE / "fit-selection.json"
    if not selection.exists():
        groups = [("fits", RATES)]
        while True:
            candidates = {}
            waiting = False
            for group, rates in groups:
                root = BASE / group
                force = ["--fallback"] if group == "extra-fits" else []
                last = root / f"lr{rates[-1][0]:g}-e{rates[-1][1]}" / "summary.json"
                ensure(
                    jobs,
                    group,
                    "online-combat-pilot-mps",
                    HERE / "fit.py",
                    ["--out", root, *force],
                    last,
                )
                for rate, epochs in rates:
                    name = f"lr{rate:g}-e{epochs}"
                    path = root / name
                    if not (path / "summary.json").exists():
                        waiting = True
                        continue
                    candidates[name] = (path / "player.pt", read(path / "summary.json"))
            players = {"reference": PARENT, **{k: v[0] for k, v in candidates.items()}}
            paths = panel(jobs, "pilot-eval", players, 243_000_000_000, 2048)
            if waiting or paths is None:
                return "fitting combat settings and comparing greedy play"
            results = {}
            for k, (ck, fit) in candidates.items():
                diff = compare(
                    paths["reference"],
                    paths[k],
                    243_000_000_000,
                    "reference",
                    k,
                    PARENT,
                    ck,
                    2048,
                )
                results[k] = dict(comparison=diff, fit=fit)
            eligible = [k for k, x in results.items() if x["fit"]["eligible"]]
            if eligible:
                break
            if len(groups) > 1:
                raise RuntimeError(
                    "No pilot fit passed the movement and stability screens."
                )
            groups.append(("extra-fits", EXTRA))

        def order(k):
            v = results[k]
            return (
                -v["comparison"]["full_cohort_gain_bounds_pp"][0],
                v["fit"]["rate"],
                v["fit"]["epochs"],
            )

        k = min(eligible, key=order)
        fit = results[k]["fit"]
        atomic_json(
            selection,
            dict(
                selected=dict(
                    name=k, target=target, rate=fit["rate"], epochs=fit["epochs"]
                ),
                results=results,
            ),
        )
        note(
            f"Selected combat learning rate {fit['rate']:g}, {fit['epochs']} pass(es), on the 2,048 pilot worlds."
        )
    players = {r: BASE / r / "iter_0024.pt" for r in ("control", "joint")}
    planned = BASE / "production-plan.json"
    if not planned.exists():
        from dataclasses import asdict
        from train import config

        selection_record = read(selection)["selected"]
        atomic_json(
            planned,
            {
                "parent_sha256": PARENT_SHA,
                "pilot": selection_record,
                "arms": {
                    role: asdict(config(path.parent, role))
                    for role, path in players.items()
                },
                "primary": {"start": 252_000_000_000, "worlds": 16384},
                "confirmation": {"start": 253_000_000_000, "worlds": 16384},
            },
        )
    # Use the Mac training lane once the preceding study's training weights are frozen.
    # Its confirmation jobs can keep using the CPU alongside the new training.
    previous = RUNS / "refit_timing_from_9700"
    prior_status = previous / "status.json"
    prior_stage = read(prior_status).get("stage") if prior_status.exists() else None
    prior_is_evaluating = prior_stage in {
        "paired endpoint selection",
        "independent confirmation",
        "finished",
    }
    if not prior_is_evaluating and not any(
        (previous / p).exists() for p in ("frozen.json", "verdict.json")
    ):
        return "pilot complete; waiting for the current training jobs to release the Mac GPU"
    for r, ck in players.items():
        ensure(
            jobs,
            "train-" + r,
            "refit-timing-ppo",
            HERE / "train.py",
            [r, "--out", ck.parent],
            ck,
        )
    partial = {r: ck for r, ck in players.items() if ck.exists()}
    primary = (
        panel(jobs, "primary", partial, 252_000_000_000, 16384) if partial else None
    )
    if len(partial) < 2 or primary is None:
        return "training and paired update-24 evaluation"
    first = BASE / "primary.json"
    if not first.exists():
        stats = compare(
            primary["control"],
            primary["joint"],
            252_000_000_000,
            "control",
            "joint",
            players["control"],
            players["joint"],
            16384,
        )
        choice = "joint" if stats["full_cohort_gain_bounds_pp"][0] > 0 else "control"
        atomic_json(
            first,
            dict(
                nominee=choice,
                comparison=stats,
                hashes={r: digest(c) for r, c in players.items()},
            ),
        )
        note(
            f"Froze {choice} as the primary nominee after the 16,384-world comparison."
        )
    primary_record = read(first)
    if any(digest(c) != primary_record["hashes"][r] for r, c in players.items()):
        raise ValueError("a frozen endpoint changed")
    players.update(parent=PARENT)
    paths = panel(jobs, "confirmation", players, 253_000_000_000, 16384)
    if paths is None:
        return "independent confirmation of both endpoints and the original champion"
    comparisons = {}
    for a, b in (("control", "joint"), ("parent", "control"), ("parent", "joint")):
        comparisons[a + "_vs_" + b] = compare(
            paths[a], paths[b], 253_000_000_000, a, b, players[a], players[b], 16384
        )
    # Also orient the worst-case missing outcomes towards the control when it is the nominee.
    if primary_record["nominee"] == "control":
        schedule = compare(
            paths["joint"],
            paths["control"],
            253_000_000_000,
            "joint",
            "control",
            players["joint"],
            players["control"],
            16384,
        )
        primary_direction = (
            primary_record["comparison"]["full_cohort_gain_bounds_pp"][1] < 0
        )
    else:
        schedule = comparisons["control_vs_joint"]
        primary_direction = (
            primary_record["comparison"]["full_cohort_gain_bounds_pp"][0] > 0
        )
    nominee = primary_record["nominee"]
    final = dict(
        primary=primary_record,
        confirmation=comparisons,
        schedule_advantage=primary_direction and schedule["worst_case_ci95_pp"][0] > 0,
        beats_start=comparisons["parent_vs_" + nominee]["worst_case_ci95_pp"][0] > 0,
    )
    from report import finish

    finish(final)
    atomic_json(BASE / "verdict.json", final)
    return "finished"


def main():
    BASE.mkdir(exist_ok=True)
    with (BASE / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if digest(PARENT) != PARENT_SHA:
            raise ValueError("unexpected parent checkpoint")
        stage = None
        heartbeat = 0
        while not (BASE / "STOP").exists() and not (BASE / "verdict.json").exists():
            jobs = {j["name"]: j for j in read(RUNS / "queue.json")["runs"]}
            for name, j in jobs.items():
                if name.startswith("online-combat-") and j["status"] in (
                    "failed",
                    "cancelled",
                ):
                    # A successful replacement attempt supersedes a prior failed launch.
                    replacement = BASE / "recovered-jobs.json"
                    if replacement.exists() and name in read(replacement):
                        continue
                    raise RuntimeError(
                        f"Child {name} {j['status']}. Inspect {RUNS / name / 'stdout.log'}"
                    )
            current = tick(jobs)
            if current != stage:
                stage = current
                note("Stage — " + stage + ".")
            now = time.time()
            atomic_json(BASE / "status.json", dict(stage=stage, t=now, pid=os.getpid()))
            if now - heartbeat >= 600:
                with (BASE / "progress.jsonl").open("a") as f:
                    print(
                        json.dumps(
                            dict(
                                t=now,
                                stage=stage,
                                active=[
                                    n
                                    for n, j in jobs.items()
                                    if n.startswith("online-combat-")
                                    and j["status"] == "running"
                                ],
                            )
                        ),
                        file=f,
                    )
                heartbeat = now
            if current == "finished":
                break
            time.sleep(20)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "exec":
        os.chdir(ROOT)
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv[2:]])
    main()
