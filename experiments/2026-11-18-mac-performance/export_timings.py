"""Export the raw trial timings and the full-workload time breakdown for the research page."""

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path):
    return list(map(json.loads, path.read_text().splitlines()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    all_timings = []
    for suite in (
        "env_abba",
        "eval_abba",
        "collect_baab",
        "train_cpu",
        "mps_diagnostics",
        "full_mps",
    ):
        for i, t in enumerate(read_jsonl(args.runs / suite / "timings.jsonl")):
            arm = t["arm"]
            f = args.runs / suite / f"{i}-{arm}"
            all_timings.append(
                dict(
                    suite=suite,
                    arm=arm,
                    trial=i,
                    measure="process wall seconds",
                    value=t["seconds_including_startup"],
                )
            )
            if f.is_dir():
                row = next(
                    r for r in read_jsonl(f / "log.jsonl") if r["event"] == "train"
                )
                for k in (
                    "iteration_seconds",
                    "collect_seconds",
                    "explore_seconds",
                    "update_seconds",
                ):
                    all_timings.append(
                        dict(suite=suite, arm=arm, trial=i, measure=k, value=row[k])
                    )
            elif f.with_suffix(".json").exists():
                row = json.loads(f.with_suffix(".json").read_text())[0]
                all_timings.append(
                    dict(
                        suite=suite,
                        arm=arm,
                        trial=i,
                        measure="workload seconds",
                        value=row["seconds"],
                    )
                )
            else:
                import re

                val = float(
                    re.search(
                        r"([0-9.]+) us/decision", f.with_suffix(".log").read_text()
                    )[1]
                )
                all_timings.append(
                    dict(
                        suite=suite,
                        arm=arm,
                        trial=i,
                        measure="microseconds per decision",
                        value=val,
                    )
                )

    def add(suite, rows):
        for i, r in enumerate(rows):
            name = r.get("arm", "")
            if "method" in r:
                name = f"{r['method']} {name}"
            all_timings.append(
                dict(
                    suite=suite,
                    arm=name,
                    trial=i,
                    measure="seconds",
                    value=r["seconds"],
                )
            )

    for k in ("encoder", "receipts", "native_search", "mlx_prefix"):
        add(k, json.loads((args.runs / f"{k}.json").read_text()))
    for k, rows in json.loads((args.runs / "mlx.json").read_text()).items():
        if k in ("generation", "scoring", "gradients"):
            add("mlx_" + k, rows)
    add("summary", read_jsonl(args.runs / "summary.jsonl"))
    for f in sorted((args.runs / "parallel_eval").glob("*.json")):
        add("parallel_eval", [dict(json.loads(f.read_text()), arm=f.stem)])
    for f in sorted((args.runs / "native_builds").glob("timed-*.json")):
        add(f.stem, json.loads(f.read_text()))

    def write_csv(name, rows):
        with (output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    write_csv("timings.csv", all_timings)
    durations = []
    for i, row in enumerate(read_jsonl(args.runs / "full_mps/timings.jsonl")):
        events = read_jsonl(args.runs / "full_mps" / f"{i}-{row['arm']}" / "log.jsonl")
        train = next(r for r in events if r["event"] == "train")
        total = row["seconds_including_startup"]
        d = {
            "implementation": "Baseline" if row["arm"] == "a" else "Candidate",
            "Collect games": train["collect_seconds"],
            "Explore statistics": train["explore_seconds"],
            "Update and gradient check": train["update_seconds"],
            "Two evaluations": sum(
                r["elapsed"] for r in events if r["event"] == "eval"
            ),
        }
        d["Other"] = total - sum(v for k, v in d.items() if k != "implementation")
        durations.append(d)
    durations.sort(key=lambda x: x["implementation"])
    write_csv("full-iteration.csv", durations)
    import matplotlib.pyplot as plt

    colors = ["#23677A", "#62A996", "#DEAE65", "#7279AF", "#C4CBD0"]
    fig, ax = plt.subplots(figsize=(8.0, 2.9), layout="constrained")
    left = [0.0, 0.0]
    for key, color in zip(list(durations[0])[1:], colors):
        vals = [d[key] for d in durations]
        ax.barh([1, 0], vals, left=left, color=color, label=key, height=0.48)
        left = [a + b for a, b in zip(left, vals)]
    for y, val in zip([1, 0], left):
        ax.text(val + 3, y, f"{val:.0f} s", va="center", fontsize=9)
    ax.set_xlim(0, max(left) * 1.12)
    ax.set_yticks([1, 0], ["Baseline", "Candidate"])
    ax.set_xlabel("Wall time (seconds)")
    ax.set_title(
        "One 3,072-game PPO iteration and two 1,024-game evaluations",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, -0.37), ncol=3, frameon=False, fontsize=8
    )
    fig.savefig(output / "full-iteration.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
