"""Plot the fixed fight gates and paired full-game effects in percentage points."""
import csv
import json
import subprocess
import sys
from pathlib import Path


def draw(output: Path, gates, early, endpoint, confirmation):
    try:
        import matplotlib
    except ModuleNotFoundError:
        # The research runtime is frozen. Render with the cached plotting extra on this Mac.
        subprocess.run(["uv", "run", "--offline", "--no-project", "--with", "matplotlib==3.11.2",
                        "--python", sys.executable, "python", __file__, str(output)],
                       input=json.dumps([gates, early, endpoint, confirmation]), text=True, check=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=True, parents=True)
    rows = []
    for iteration, groups, adopted in gates:
        for mode in ("unsalted", "resampled"):
            m = groups["BOSS3:" + mode]["win"]
            effect, margin = 100 * m["paired_difference"], 196 * m["se"]
            rows.append(dict(panel="boss", comparison=mode, iteration=iteration, pairs=m["worlds"],
                             estimate=effect, lower=effect-margin, upper=effect+margin, adopted=adopted))
    full_rows, plotted_rows, seen = [], [], set()
    for phase, comparisons in (("Update 8", early), ("Update 24", endpoint), ("Confirmation", confirmation or {})):
        for item in comparisons.values():
            est = item.get("difference_pp", item.get("complete_pair_difference_pp"))
            ci = item.get("ci95_pp", item.get("complete_pair_ci95_pp"))
            label = phase + " vs " + {"every8":"every eight", "control":"frozen combat", "incumbent":"incumbent"}[item["a"]]
            full_rows.append(dict(panel="games", comparison=label, iteration=8 if phase == "Update 8" else 24,
                                  pairs=item["paired_completed"], estimate=est, lower=ci[0] if ci else None,
                                  upper=ci[1] if ci else None, adopted=""))
            baseline = item.get("checkpoint_sha256", {}).get(item["a"], item["a"])
            case = (phase, baseline)
            if case not in seen:
                seen.add(case)
                plotted_rows.append(full_rows[-1])
    with (output / "cadence-effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows + full_rows)

    fig, (fights, games) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.4,1]},
                                        layout="constrained")
    fig.suptitle("Act 3 boss refits every two strategy updates", x=.02, ha="left", weight="bold", fontsize=16)
    for ax in (fights, games):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(length=3, color="#8a9298")
    for key, label, offset, color in (("unsalted","Original entries",-.18,"#1a5880"),
                                     ("resampled","Reseeded entries",.18,"#bf6828")):
        values = [v for v in rows if v["comparison"] == key]
        fights.errorbar([r["iteration"]+offset for r in values], [r["estimate"] for r in values],
                        yerr=[r["upper"]-r["estimate"] for r in values], fmt="o", ms=4, lw=1.1,
                        capsize=2, alpha=.9, label=label, color=color)
    fights.axhline(0, lw=.8, color="#4e555a")
    fights.set(title="Specialist vs active combat", xlabel="Strategy update", ylabel="Boss win-rate difference (points)",
               xticks=[g[0] for g in gates])
    fights.legend(frameon=False, fontsize=9, loc="best")
    games.axvline(0, lw=.8, color="#4e555a")
    for y, row in enumerate(plotted_rows):
        if row["lower"] is not None:
            x = row["estimate"]
            games.errorbar(x,y,xerr=[[max(0,x-row["lower"])],[max(0,row["upper"]-x)]],
                           fmt="o", ms=5, lw=1.4, capsize=3, color="#1a5880")
    games.set_title("Every-two full-game results (95% intervals)")
    games.set_yticks(range(len(plotted_rows)),labels=[v["comparison"] for v in plotted_rows],fontsize=8)
    games.invert_yaxis()
    games.set_xlabel("Full-game win-rate difference (points)")
    games.margins(x=.18, y=.3)
    fig.savefig(output / "cadence-effects.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(sys.argv[1]), *json.load(sys.stdin))
