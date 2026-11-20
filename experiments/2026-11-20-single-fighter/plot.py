"""Export the fit curves and the three paired performance outcomes from saved records."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run, figures):
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained", sharey=True)
    with (figures/"imitation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["boss_share", "update", "role", "held_out_choices", "top_agreement", "kl"])
        for name, share, colour in (("mix25", .25, "#17698a"), ("mix50", .5, "#aa541a")):
            rows = [json.loads(line) for line in (run/name/"progress.jsonl").read_text().splitlines()]
            for ax, role in zip(axes, ("ordinary", "boss")):
                x, y = [], []
                for row in rows:
                    if "validation" not in row:
                        continue
                    m = row["validation"][role]
                    step = row.get("step", 0)
                    writer.writerow([share, step, role, m["choices"], m["top_agreement"], m["kl"]])
                    x.append(step)
                    y.append(100*m["top_agreement"])
                ax.plot(x, y, color=colour, linewidth=2, label=f"{share:.0%} boss examples")
                ax.set(xlabel="Training update", title="Ordinary fights" if role=="ordinary" else "Act 3 boss fights")
                ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Teacher agreement (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(figures/"imitation.png", dpi=135)
    plt.close(fig)

    outcome = json.loads((run/"verdict.json").read_bytes())
    comparisons = [("Full games",outcome["full_games"],-1),
                   ("Original boss starts",outcome["bosses"]["original"],-3),
                   ("Reseeded boss starts",outcome["bosses"]["reseeded"],None)]
    fig, ax = plt.subplots(figsize=(7.5, 2.9), layout="constrained")
    ax.axvline(0,color="#6c7277",linestyle="--",linewidth=.8)
    with (figures/"performance.csv").open("w",newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["panel","assigned","complete_pairs","student_only","champion_only",
                         "difference_pp","lower95","upper95","retention_margin_pp"])
        for index,(name,r,margin) in enumerate(comparisons):
            delta = r.get("difference_pp", r.get("complete_pair_difference_pp"))
            lo,hi = r.get("ci95_pp", r.get("complete_pair_ci95_pp"))
            writer.writerow([name,r["n"],r["paired_completed"],r["b_only"],r["a_only"],delta,lo,hi,margin])
            ax.errorbar(delta,index,xerr=[[delta-lo],[hi-delta]],fmt="o",color="#17698a",capsize=4,
                        label="Paired 95% interval" if index==0 else None)
            if margin is not None:
                ax.scatter(margin,index,marker="|",s=155,color="#aa541a",
                           label="Retention threshold" if index==0 else None)
    ax.set_yticks(range(len(comparisons)),[x[0] for x in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("Single fighter minus champion win rate (percentage points)")
    ax.spines[["right","top"]].set_visible(False)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.25),ncol=2,frameon=False,fontsize=8)
    fig.savefig(figures/"performance.png",dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1]),Path(sys.argv[2]))
