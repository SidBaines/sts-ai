"""Cross three exploration levels, rates and pass counts, then test a fresh joint trajectory."""

from dataclasses import replace
from itertools import product
from pathlib import Path
import argparse
import gzip
import json

import study
from pilot import atomic_json
from sts_rl.online_combat import confirmed_combat_challenger, nominate_combat_challenger
from sts_rl.paired_endpoints import compare, digest
from train import config, train

RUN = study.BASE / "expanded"
TARGETS = (0.05, 0.12, 0.22)
RATES = (1e-5, 3e-5, 6e-5)
PASSES = (1, 2, 4)
SCREEN, DIAGNOSTIC, TRAIN_BASE, CHECK = (
    260_000_000_000,
    264_000_000_000,
    262_000_000_000,
    269_000_000_000,
)
REFERENCE = "x12-lr3e-05-e2"
PREFIX = "combat-grid-"
SPARE = "combat-grid-spare"


def fit_name(rate, epochs):
    return f"lr{rate:g}-e{epochs}"


def key(target, rate, epochs):
    return f"x{round(100 * target):02d}-{fit_name(rate, epochs)}"


def recipe(target, rate, epochs):
    return dict(name=key(target, rate, epochs), target=target, rate=rate, epochs=epochs)


def fit_folder(target, rate, epochs):
    name = fit_name(rate, epochs)
    if target == 0.12 and (rate, epochs) in study.RATES:
        return study.BASE / "fits" / name
    return RUN / f"fits-{round(100 * target):02d}" / name


def job(jobs, name, script, args, marker):
    return study.ensure(jobs, name, SPARE, script, args, marker, prefix=PREFIX, nice=8)


def fit_jobs(jobs):
    for target in TARGETS:
        tag = f"{round(100 * target):02d}"
        directory = study.BASE / "fit-data" if target == 0.12 else RUN / f"data-{tag}"
        if target != 0.12 and not job(
            jobs,
            "collect-" + tag,
            study.HERE / "pilot.py",
            ["fit", "--target", target, "--processes", 6, "--out", directory],
            directory / "summary.json",
        ):
            continue
        settings = [
            (r, e)
            for r, e in product(RATES, PASSES)
            if not (target == 0.12 and (r, e) in study.RATES)
        ]
        folder = RUN / f"fits-{tag}"
        last = folder / fit_name(*settings[-1]) / "summary.json"
        options = [v for rate, epochs in settings for v in ("--setting", rate, epochs)]
        job(
            jobs,
            "fit-" + tag,
            study.HERE / "fit.py",
            ["--data", directory, "--out", folder, *options],
            last,
        )


def evaluate(jobs, name, checkpoint, start, n, folder, max_active=3):
    return job(
        jobs,
        name,
        study.EVAL,
        [
            "--checkpoint",
            checkpoint,
            "--world-start",
            start,
            "--n",
            n,
            "--processes",
            7,
            "--max-active-processes",
            max_active,
            "--games-per-process",
            8,
            "--progress-every",
            128 if n == 768 else 512,
            "--out",
            folder,
        ],
        folder / "summary.json",
    )


def screen(jobs):
    ref = fit_folder(0.12, 3e-5, 2) / "player.pt"
    ref_eval = RUN / "screen" / REFERENCE
    existing = [
        j for j in jobs.values() if j["status"] in ("running", "paused", "pending")
    ]
    spare_busy = any(j.get("lane") == SPARE for j in existing)
    results = {}
    for target, rate, epochs in product(TARGETS, RATES, PASSES):
        label = key(target, rate, epochs)
        folder = fit_folder(target, rate, epochs)
        out = RUN / "screen" / label
        saved = RUN / "comparisons" / (label + ".json")
        if saved.exists():
            results[label] = study.read(saved)
            continue
        if not (folder / "summary.json").exists():
            continue
        checkpoint = folder / "player.pt"
        name = "screen-" + label
        if PREFIX + name not in jobs:
            if spare_busy:
                continue
            spare_busy = True
        if (
            not evaluate(jobs, name, checkpoint, SCREEN, 768, out)
            or not (ref_eval / "summary.json").exists()
        ):
            continue
        fit = study.read(folder / "summary.json")
        stat = compare(ref_eval, out, SCREEN, REFERENCE, label, ref, checkpoint, 768)
        row = {
            "setting": recipe(target, rate, epochs),
            "eligible": fit["eligible"],
            "movement": fit["movement"],
            "steps": fit["steps"],
            "comparison": stat,
        }
        atomic_json(saved, row)
        results[label] = row
        print(
            f"{label}: {stat['second']['wins']}/{stat['n']} wins, eligible={fit['eligible']}",
            flush=True,
        )
    return results


def short_job(jobs, role):
    folder = RUN / ("short-" + role)
    return job(
        jobs,
        "short-" + role,
        Path(__file__),
        ["train", role, "--out", folder],
        folder / "iter_0003.pt",
    )


def short_train(role, out):
    setting = (
        study.read(study.BASE / "fit-selection.json")["selected"]
        if role == "reference"
        else study.read(RUN / "shortlist.json")["challenger"]
    )
    cfg = replace(
        config(out, "joint", setting=setting),
        iterations=3,
        games=1536,
        seed=12,
        train_world_start=TRAIN_BASE,
        collectors=4,
        workers=128,
        update_threads=4,
        eval_every=3,
        eval_n=128,
        eval_world_start=DIAGNOSTIC,
    )
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "training-plan.json",
        {"label": "Expanded pilot · " + role, "iterations": 3},
    )
    train(cfg, resume=(out / "latest.pt").exists())


def training_records():
    """Validate all assigned worlds, including games that did not finish."""
    seen = set()
    rows = []
    for i in range(1, 4):
        sets = []
        row = {"update": i}
        for role in ("reference", "challenger"):
            directory = RUN / ("short-" + role)
            cfg = study.read(directory / "config.json")
            if (
                Path(cfg["checkpoint"]).resolve() != study.PARENT
                or cfg["seed"] != 12
                or not cfg["train_combat"]
            ):
                raise ValueError("unexpected joint training parent or seed")
            with gzip.open(directory / f"collection_{i:04d}.json.gz", "rt") as f:
                games = json.load(f)
            seeds = tuple(t["seed"] for t in games["tasks"])
            if len(seeds) != 1536 or len(set(seeds)) != 1536:
                raise ValueError("incomplete training assignment")
            sets.append(seeds)
            ep = games["episodes"]
            row[role] = {
                "wins": sum(
                    e["task_outcome"] in ("victory", "act_clear")
                    and e["terminated"]
                    and not e["truncated"]
                    for e in ep
                ),
                "faults": len(games["faults"]),
                "cuts": sum(e["truncated"] for e in ep),
                "assigned": len(seeds),
            }
        if sets[0] != sets[1] or seen.intersection(sets[0]):
            raise ValueError("the three joint updates must use matched fresh worlds")
        seen.update(sets[0])
        rows.append(row)
    return rows


def same_evaluation():
    saved = RUN / "cpu-parity.json"
    if saved.exists():
        return True
    capped_dir = RUN / "cpu-parity"
    reference_dir = RUN / "screen" / REFERENCE
    if not (capped_dir / "summary.json").exists():
        return False
    first, second = (
        study.read(p / "summary.json") for p in (reference_dir, capped_dir)
    )
    a, b = (study.read(p / "run.json") for p in (reference_dir, capped_dir))
    if (
        a["native"] != b["native"]
        or a["checkpoints"]["checkpoint"]["sha256"]
        != b["checkpoints"]["checkpoint"]["sha256"]
    ):
        raise ValueError(
            "process-cap comparison used different models or native builds"
        )
    keys = (
        "action_hash",
        "action_prefix",
        "terminal_hash",
        "win",
        "status",
        "floor",
        "bosses",
        "decisions",
        "behavior",
        "requeues",
    )

    def signatures(summary):
        if (
            summary.get("error")
            or summary["completed"] != 768
            or summary["faults"]
            or summary["cuts"]
        ):
            raise ValueError(
                "process-cap verification requires all 768 worlds to finish"
            )
        return {r["world_seed"]: tuple(r[k] for k in keys) for r in summary["records"]}

    original = signatures(first)
    if set(original) != set(range(SCREEN, SCREEN + 768)) or original != signatures(
        second
    ):
        raise ValueError("capping active evaluator processes changed a trajectory")
    atomic_json(
        saved,
        {
            "identical_worlds": 768,
            "wins": first["wins"],
            "seven_active_seconds": first["seconds"],
            "three_active_seconds": second["seconds"],
            "logical_processes": 7,
            "games_per_process": 8,
        },
    )
    return True


def advance(jobs):
    # A single low-priority supplemental job shares the Mac with the older trainer.
    # First drain jobs launched with the initial higher-concurrency arrangement.
    if any(
        j["name"].startswith(PREFIX)
        and j["status"] in ("running", "paused", "pending")
        and j.get("lane") != SPARE
        for j in jobs.values()
    ):
        return "finishing already queued work at lower concurrency"
    reference_player = fit_folder(0.12, 3e-5, 2) / "player.pt"
    reference_screen = RUN / "screen" / REFERENCE
    if not (reference_screen / "summary.json").exists():
        evaluate(
            jobs,
            "screen-" + REFERENCE,
            reference_player,
            SCREEN,
            768,
            reference_screen,
            max_active=7,
        )
        return "evaluating the original setting on the fresh screening worlds"
    if not same_evaluation():
        evaluate(jobs, "cpu-parity", reference_player, SCREEN, 768, RUN / "cpu-parity")
        return "checking all 768 trajectories with at most three evaluator processes"
    fit_jobs(jobs)
    ref_ready = short_job(jobs, "reference")
    results = screen(jobs)
    if len(results) < 27:
        return f"screened {len(results)} of 27 settings"
    nomination = RUN / "shortlist.json"
    if not nomination.exists():
        name = nominate_combat_challenger(results, REFERENCE)
        if name is None:
            raise RuntimeError(
                "No eligible distinct challenger despite the original pilot fits."
            )
        atomic_json(
            nomination,
            {
                "reference": results[REFERENCE]["setting"],
                "challenger": results[name]["setting"],
                "screen": results[name]["comparison"],
            },
        )
        study.note(
            f"Expanded pilot selected {name} for three fresh joint updates against the original recipe."
        )
    pair = study.read(nomination)
    alt_ready = short_job(jobs, "challenger")
    if not (ref_ready and alt_ready):
        return "testing the two recipes over three real joint updates"
    players = {
        r: RUN / ("short-" + r) / "iter_0003.pt" for r in ("reference", "challenger")
    }
    frozen = RUN / "frozen.json"
    if not frozen.exists():
        atomic_json(frozen, {r: digest(p) for r, p in players.items()})
    paths = {r: RUN / "check" / r for r in players}
    ready = [
        evaluate(jobs, "check-" + r, p, CHECK, 8192, paths[r])
        for r, p in players.items()
    ]
    if not all(ready):
        return "comparing the frozen third updates on 8,192 independent worlds"
    if study.read(frozen) != {r: digest(p) for r, p in players.items()}:
        raise ValueError("the frozen endpoints changed")
    comparison = compare(
        paths["reference"],
        paths["challenger"],
        CHECK,
        "reference",
        "challenger",
        players["reference"],
        players["challenger"],
        8192,
    )
    result = {
        **pair,
        "comparison": comparison,
        "qualified": confirmed_combat_challenger(pair["screen"], comparison),
        "training": training_records(),
    }
    from expanded_report import write_results

    write_results(results, result)
    study.note(
        f"Expanded pilot independent comparison: challenger {comparison['second']['wins']}, original {comparison['first']['wins']} wins on 8,192 worlds. Qualified to change settings: {result['qualified']}."
    )
    return "finished"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "train"))
    parser.add_argument("role", nargs="?", choices=("reference", "challenger"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "train":
        if args.role is None:
            parser.error("short training needs a named role")
        short_train(args.role, args.out)
    else:
        if args.out.resolve() != RUN.resolve():
            parser.error("unexpected expanded pilot root")
        study.main(RUN, advance, PREFIX, "Expanded pilot")
