"""Compare matched frozen checkpoints, keeping faulted and censored worlds explicit."""

import hashlib
import json
from pathlib import Path

N = 16_384
PRIMARY = 125_000_000_000
CONFIRM = 126_000_000_000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(folder, start, checkpoint):
    run = json.loads((folder / "run.json").read_text())
    cfg = run["config"]
    if (cfg["world_start"], cfg["n"], cfg["processes"], cfg["games_per_process"]) != (
        start,
        N,
        7,
        8,
    ):
        raise ValueError(f"Unexpected evaluation configuration in {folder}")
    if run["checkpoints"]["checkpoint"]["sha256"] != digest(checkpoint):
        raise ValueError(f"Checkpoint changed for {folder}")
    s = json.loads((folder / "summary.json").read_bytes())
    if s.get("error") or s.get("programming_errors"):
        raise ValueError(f"Evaluation error in {folder}: {s.get('error')}")
    records = s["records"]
    worlds = {}
    for r in records:
        seed, status = r["world_seed"], r["status"]
        if seed in worlds or status not in ("completed", "censored", "fault"):
            raise ValueError(f"Unexpected or duplicate world {seed}: {status}")
        if status == "completed" and type(r["win"]) is not bool:
            raise ValueError(f"Missing completion outcome for {seed}")
        worlds[seed] = r["win"] if status == "completed" else None
    if set(worlds) != set(range(start, start + N)):
        raise ValueError(f"World roster mismatch in {folder}")
    counts = {
        k: s[k]
        for k in (
            "completed",
            "wins",
            "faults",
            "cuts",
            "censored",
            "mean_floor",
            "requeues",
        )
    }
    counts["unknown_worlds"] = {
        r["world_seed"]: dict(status=r["status"], reason=r.get("reason"))
        for r in records
        if r["status"] != "completed"
    }
    return worlds, counts


def compare(a_folder, b_folder, start, a_name, b_name, a_ck, b_ck):
    from sts_rl.fresh_statistics import exact_discordance, paired_win_normal_interval

    a, first = compact(a_folder, start, a_ck)
    b, second = compact(b_folder, start, b_ck)
    up = down = low_unknown = high_unknown = unknown = 0
    for seed in a:
        x, y = a[seed], b[seed]
        if x is None or y is None:
            unknown += 1
            low_unknown += int(y or False) - int(True if x is None else x)
            high_unknown += int(True if y is None else y) - int(x or False)
        else:
            up += int(y and not x)
            down += int(x and not y)
    delta = up - down
    result = dict(
        start=start,
        n=N,
        a=a_name,
        b=b_name,
        first=first,
        second=second,
        paired_completed=N - unknown,
        b_only=up,
        a_only=down,
        full_cohort_gain_bounds_pp=[
            100 * (delta + low_unknown) / N,
            100 * (delta + high_unknown) / N,
        ],
        complete=unknown == 0,
    )
    if not unknown:
        result.update(
            difference_pp=100 * delta / N,
            ci95_pp=[100 * x for x in paired_win_normal_interval(N, up, down)],
            p_two_sided=exact_discordance(up, down),
        )
    else:
        # Descriptive complete-pair result only; preserve full-cohort identification bounds above.
        n = N - unknown
        result.update(
            complete_pair_difference_pp=100 * delta / n if n else None,
            complete_pair_ci95_pp=[
                100 * x for x in paired_win_normal_interval(n, up, down)
            ]
            if n > 1
            else None,
        )
    return result
