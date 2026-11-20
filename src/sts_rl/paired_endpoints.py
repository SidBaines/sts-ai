"""Compare matched frozen checkpoints, keeping faulted and censored worlds explicit."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(folder, start, checkpoint, n, *, processes=7, games_per_process=8):
    run = json.loads((folder / "run.json").read_text())
    cfg = run["config"]
    if (cfg["world_start"], cfg["n"], cfg["processes"], cfg["games_per_process"]) != (
        start,
        n,
        processes,
        games_per_process,
    ):
        raise ValueError(f"Unexpected evaluation configuration in {folder}")
    if any(cfg.get(k) for k in ("combat", "ooc", "no_skip_rares", "take_card")):
        raise ValueError(f"Unexpected phase replacement or menu overlay in {folder}")
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
    if set(worlds) != set(range(start, start + n)):
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
    if counts["wins"] != sum(x is True for x in worlds.values()) or counts["completed"] != sum(
            x is not None for x in worlds.values()):
        raise ValueError(f"Record counts disagree with summary in {folder}")
    return worlds, counts


def compare(a_folder, b_folder, start, a_name, b_name, a_ck, b_ck, n):
    natives = [json.loads((p / "run.json").read_bytes()).get("native", {}) for p in (a_folder, b_folder)]
    if not natives[0].get("module_sha256") or natives[0] != natives[1]:
        raise ValueError("Paired evaluations require the same recorded native build")
    a, first = compact(a_folder, start, a_ck, n)
    b, second = compact(b_folder, start, b_ck, n)
    return dict(start=start, a=a_name, b=b_name, first=first, second=second, **paired_binary_outcomes(a, b))


def paired_binary_outcomes(a, b):
    """Compare second minus first on identical keys with explicit unknown outcomes."""
    from sts_rl.fresh_statistics import exact_discordance, paired_win_normal_interval

    if a.keys() != b.keys() or len(a) < 2:
        raise ValueError("Paired outcomes require the same roster of at least two positions")
    if any(x is not None and type(x) is not bool for arm in (a, b) for x in arm.values()):
        raise ValueError("A binary outcome must be a boolean or unknown")
    n = len(a)
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
        n=n,
        paired_completed=n - unknown,
        b_only=up,
        a_only=down,
        full_cohort_gain_bounds_pp=[
            100 * (delta + low_unknown) / n,
            100 * (delta + high_unknown) / n,
        ],
        complete=unknown == 0,
    )
    if not unknown:
        result.update(
            difference_pp=100 * delta / n,
            ci95_pp=[100 * x for x in paired_win_normal_interval(n, up, down)],
            p_two_sided=exact_discordance(up, down),
        )
    else:
        # Descriptive complete-pair result only; preserve full-cohort identification bounds above.
        valid = n - unknown
        result.update(
            complete_pair_difference_pp=100 * delta / valid if valid else None,
            complete_pair_ci95_pp=[
                100 * x for x in paired_win_normal_interval(valid, up, down)
            ]
            if valid > 1
            else None,
        )
    # Complete each missing pair against the candidate, preserving any known outcome.
    worst_down = sum(a[k] is not False and b[k] is not True for k in a)
    result["worst_case_ci95_pp"] = [100 * v for v in paired_win_normal_interval(n, up, worst_down)]
    return result


def compare_boss_replays(grid_path, panel_folder, champion_sha256, fighter_sha256, starts):
    """Compare exact starting positions and a single matched reseed, retaining invalid outcomes."""
    from .fresh_panel_grid import canonical_grid_hash

    grid = json.loads(grid_path.read_bytes())
    panel = json.loads((panel_folder/"summary.json").read_bytes())
    if not (grid["grid_hash"] == canonical_grid_hash(grid) == panel["grid_hash"]
            and panel["grid_input"]["sha256"] == digest(grid_path)
            and grid["coverage"] == {"BOSS3": starts}):
        raise ValueError("Final boss position roster mismatch")
    identity = panel["source"]["policy"]
    if identity["champion"]["sha256"] != champion_sha256 or identity["combat"]["sha256"] != fighter_sha256:
        raise ValueError("Unexpected panel fighter")
    result = {}
    for salt, label in ((-1, "original"), (0, "reseeded")):
        keys = {(r["world_seed"],r["start_id"]) for r in grid["grid"] if r["salt"] == salt}
        if len(keys) != starts or len({s for s,_ in keys}) != len(keys):
            raise ValueError("Boss evaluation requires one starting position per world")
        outcomes = {p:dict.fromkeys(keys) for p in ("champion","replacement")}
        hashes, seen = {}, set()
        for row in panel["records"]:
            if row["salt"] != salt:
                continue
            key, arm = (row["world_seed"],row["start_id"]), row["policy"]
            if key not in keys or (key,arm) in seen:
                raise ValueError("Unexpected or duplicate boss position")
            seen.add((key,arm))
            if any(row.get(v) for v in ("fault","cut","ub","censored","programming_error")):
                continue
            fingerprint = row["first_combat_hash"]
            if not fingerprint or hashes.setdefault(key,fingerprint) != fingerprint:
                raise ValueError("The two policies did not receive identical first combat states")
            if type(row["win"]) is not bool:
                raise ValueError("Missing boss-fight outcome")
            outcomes[arm][key] = row["win"]
        champion, student = outcomes["champion"], outcomes["replacement"]
        result[label] = dict(champion_wins=sum(x is True for x in champion.values()),
                             student_wins=sum(x is True for x in student.values()),
                             **paired_binary_outcomes(champion, student))
    return result, panel["confidence_claims_allowed"]
