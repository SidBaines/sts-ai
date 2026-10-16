#!/usr/bin/env python3
"""Render fresh_gate_plot_data_v1 using the separate matplotlib runtime.

Consumes estimates and CI endpoints only. Never imports torch, fresh_statistics,
training code, source results or the frozen runtime. Does not compute intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
sys.dont_write_bytecode = True

# Matplotlib's font/cache writes belong in private temporary storage, never a venv.
_CONFIG = tempfile.TemporaryDirectory(prefix="fresh-gate-mpl-")
os.environ["MPLCONFIGDIR"] = _CONFIG.name
os.environ["XDG_CACHE_HOME"] = _CONFIG.name
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
ARM_LABELS = {"combat_only": "Combat replacement", "ooc_only": "OOC replacement", "fresh_composite": "Both replacements"}
RUN_LABELS = {"old_combat_lr3e4": "Old LARGE/NONE · LR 0.0003",
              "old_combat_lr1e3": "Old LARGE/NONE · LR 0.001", "new_ooc_full": "New OOC · full FIT"}
BEHAVIOR_LABELS = {"smith_hp>=.85": "Smith · HP ≥85% menus offering smith",
    "elite_choice_hp>=.9": "Elite · HP ≥90% menus offering elite / non-elite paths",
    "card_skip": "Explicit card-reward skip · per rewards menu offering it", "removal": "Shop card-removal purchase",
    "combat_potion_drink": "Potion drink · per combat menu offering drink",
    "combat_potion_discard": "Potion discard · per combat menu offering discard",
    "shop_spend": "Positive spend · menus offering a purchase"}
ARMS = tuple(ARM_LABELS)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#DCDCDC", "grid.alpha": .55, "grid.linewidth": .6,
        "svg.fonttype": "none", "axes.unicode_minus": True})


def finish(fig, out, name, title, caption, fixture=False, *, footer_handles=None):
    if fixture:
        title = "SYNTHETIC FIXTURE — NOT STUDY RESULTS\n" + title
    fig.suptitle(title, fontsize=15, y=.99, fontweight="bold")
    caption_artist = fig.text(.02, .015, caption, va="bottom", fontsize=9, linespacing=1.5)
    fig.canvas.draw()
    caption_top = caption_artist.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height
    bottom = max(.105, caption_top + .02)
    if footer_handles:
        legend = fig.legend(handles=footer_handles, loc="lower center", bbox_to_anchor=(.5, bottom),
                            ncol=3, fontsize=8.5, frameon=False)
        fig.canvas.draw()
        bottom = legend.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height + .02
    fig.tight_layout(rect=(0, bottom, 1, .94 if fixture else .95), pad=1.5)
    for extension in ("png", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white", bbox_inches="tight",
                    metadata={"Title": title, "Description": caption} if extension == "svg" else {"Title": title, "Description": caption})
    plt.close(fig)


def interval(ax, stat, position, *, color, scale=1, horizontal=True, marker="o"):
    """Plot absolute CI endpoints; support undefined/degenerate conditional CIs."""
    point, (lo, hi) = stat["point"], stat["ci95"]
    if point is None:
        if horizontal:
            ax.text(.98, position, "N/A: no opportunities", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8)
        return
    degenerate = stat.get("bootstrap_degenerate", False)
    if lo is not None and hi is not None and not degenerate:
        if horizontal:
            ax.hlines(position, lo * scale, hi * scale, color=color, lw=1.7)
            ax.plot([lo * scale, hi * scale], [position, position], "|", color=color, ms=8)
        else:
            ax.vlines(position, lo * scale, hi * scale, color=color, lw=1.7)
            ax.plot([position, position], [lo * scale, hi * scale], "_", color=color, ms=8)
    options = {"color": color, "ms": 6}
    if degenerate:
        marker, options["markerfacecolor"] = "D", "none"
    if horizontal:
        ax.plot(point * scale, position, marker, **options)
    else:
        ax.plot(position, point * scale, marker, **options)


def learning(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.4))
    names = ("new_combat_5000", "new_combat_20000", "new_combat_full")
    xs = [data["scores"][n]["actual_fit_rows"] for n in names]
    for ax, metric, scale, ylabel in zip(axes, ("agreement", "soft_ce"), (100, 1),
                                       ("Teacher plurality agreement (%)", "Soft cross entropy (nats / choice row)")):
        points = [data["scores"][n]["groups"]["choice"][metric] for n in names]
        ax.plot(xs, [s["point"] * scale for s in points], color=COLORS[0], label="New architecture · frozen selections")
        for x, stat in zip(xs, points):
            interval(ax, stat, x, color=COLORS[0], scale=scale, horizontal=False)
        old = data["scores"]["old_combat_full"]["groups"]["choice"][metric]
        ax.axhline(old["point"] * scale, color=COLORS[1], linestyle="--", label="Selected old LARGE/NONE · full FIT")
        ax.axhspan(old["ci95"][0] * scale, old["ci95"][1] * scale, color=COLORS[1], alpha=.12)
        if metric == "agreement":
            ref = data["teacher_reference"]["overall"]["groups"]["all"]
            ax.axhline(ref["point"] * 100, color="#555555", ls=":", label="Independent teacher repeat · REFERENCE")
            ax.axhspan(ref["ci95"][0] * 100, ref["ci95"][1] * 100, color="#888888", alpha=.10)
            paired = data["g1_labels"]["groups"]["choice"]
            ax.text(.98, .04, "Full FIT · paired new − old\n"
                f"{100*paired['point']:+.2f} pp; 95% CI "
                f"[{100*paired['ci95'][0]:+.2f}, {100*paired['ci95'][1]:+.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": .95, "pad": 5})
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:,}" + ("\n(full actual FIT)" if i == len(xs)-1 else "")
                           for i, x in enumerate(xs)])
        ax.set_xlabel("FIT labels (rows; log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title("Combat TEST · choice rows")
    overall = data["teacher_reference"]["overall"]["groups"]["all"]
    ref = data["teacher_reference"]["high_confidence"]["groups"]["all"]
    high_new, high_old = (data["scores"][name]["groups"]["high_confidence_choice"]["agreement"]
                          for name in ("new_combat_full", "old_combat_full"))
    high_paired = data["g1_labels"]["groups"]["high_confidence_choice"]
    caption = (f"fresh_start_v1 · exact argmax, actor-only singleton CPU TEST · n={data['scores']['new_combat_full']['test_rows']:,} complete combat rows per model.\n"
        "95% whole-WORLD bootstrap CIs; label-weighted. Agreement uses the paired full-FIT contrast annotation; soft-CE arm intervals are descriptive.\n"
        "Teacher is native-assisted; not certified public-only. Independent repeat agreement is a REFERENCE, not a strict ceiling.\n"
        f"Overall repeat REFERENCE: {100*overall['point']:.2f}% [{100*overall['ci95'][0]:.2f}, {100*overall['ci95'][1]:.2f}]; "
        f"n={overall['rows']:,} repeat rows / {overall['worlds']:,} worlds (separate from the TEST cohort).\n"
        f"High-confidence repeat REFERENCE: {100*ref['point']:.2f}% "
        f"[{100*ref['ci95'][0]:.2f}, {100*ref['ci95'][1]:.2f}]; target maximum ≥0.75.\n"
        f"High-confidence full-FIT TEST agreement: new {100*high_new['point']:.2f}% vs old {100*high_old['point']:.2f}%; "
        f"paired new − old {100*high_paired['point']:+.2f} pp [{100*high_paired['ci95'][0]:+.2f}, {100*high_paired['ci95'][1]:+.2f}].\n"
        "Old control fitted at full FIT only; its line is not an old learning curve and supports no data-efficiency comparison.")
    finish(fig, out, "combat_test_learning", "Frozen supervised warm-up · combat TEST learning curve", caption,
           data.get("fixture", False), footer_handles=axes[0].get_legend_handles_labels()[0])


def validation(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")
    combat = ("new_combat_5000", "new_combat_20000", "new_combat_full", "old_combat_lr3e4", "old_combat_lr1e3")
    for column, names in enumerate((combat, ("new_ooc_full",))):
        for i, name in enumerate(names):
            run = data["training"][name]
            steps = [e["step"] for e in run["series"]]
            color = COLORS[i]
            for row, metric, scale in ((0, "soft_ce", 1), (1, "agreement", 100)):
                ax = axes[row, column]
                for scope, line in (("validation", "-"), ("fit_diagnostic", "--")):
                    values = [e[scope]["groups"]["choice"][metric] * scale for e in run["series"]]
                    ax.plot(steps, values, color=color, ls=line, lw=1.5,
                            label=(f"New · FIT {run['summary']['actual_fit_rows']:,}" if name.startswith("new_combat_")
                                   else RUN_LABELS[name]) if scope == "validation" else "_nolegend_")
                chosen = next(e for e in run["series"] if e["step"] == run["summary"]["best_step"])
                ax.plot(chosen["step"], chosen["validation"]["groups"]["choice"][metric] * scale,
                        "o", ms=4, color=color)
                ax.set_title(("Combat" if column == 0 else "OOC") + " · choice rows")
                ax.set_ylabel("Soft cross entropy (nats / choice row)" if row == 0 else "Teacher plurality agreement (%)")
                ax.set_xlabel("Supervised optimizer step")
        axes[0, column].legend(loc="best", fontsize=8)
    footer_handles = [Line2D([0], [0], color="#333333", label="VALIDATION", ls="-"),
                      Line2D([0], [0], color="#333333", label="FIT diagnostic · fixed 128-row subset", ls="--"),
                      Line2D([0], [0], color="#333333", marker="o", ls="none", label="Frozen validation-selected step")]
    finish(fig, out, "validation_learning", "Supervised warm-up · logged FIT diagnostics and VALIDATION",
        "fresh_start_v1 · logged dropout-disabled supervised evaluation, before held-out TEST analysis.\n"
        "OOC soft CE is normalized over CHOICE rows, excluding forced rows. FIT diagnostics use the fixed 128-row subset (choice rows within it).\n"
        "Lines are logged point diagnostics; confidence intervals are not estimated for these logs. Cadence: 100 supervised optimizer steps.",
        data.get("fixture", False), footer_handles=footer_handles)


def probes(data, out):
    fields = (("kl_start_to_current", "KL since initial · initial ‖ current", "KL (nats / probe)"),
              ("kl_previous_to_current", "KL per block · previous ‖ current", "KL (nats / probe)"),
              ("argmax_change_start", "Argmax changes since initial", "Changed probes (%)"),
              ("argmax_change_previous", "Argmax changes since previous block", "Changed probes (%)"),
              ("entropy", "Current action entropy", "Entropy (nats / probe)"),
              ("top_probability", "Current top action probability", "Probability (%)"))
    for phase in ("combat", "ooc"):
        run = data["training"][f"new_{phase}_full"]
        series = run["series"]
        steps = [e["step"] for e in series]
        fig, axes = plt.subplots(4, 2, figsize=(13, 12.3))
        for ax, (field, title, ylabel) in zip(axes.flat, fields):
            scale = 100 if "argmax" in field or field == "top_probability" else 1
            for i, group in enumerate(("pooled", "1", "2", "3")):
                vals = [(e["fixed93e9_probes"]["pooled"] if group == "pooled" else e["fixed93e9_probes"]["by_act"][group])[field]
                        for e in series]
                ax.plot(steps, [v * scale for v in vals], color="#333333" if i == 0 else COLORS[i-1],
                        ls="--" if i == 0 else "-", lw=1.7 if i == 0 else 1.3)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Supervised optimizer step")
        axes[3, 0].plot(steps, [e["parameter_relative_delta"] for e in series], color=COLORS[3])
        axes[3, 0].set_ylabel("Relative L2 change\n(dimensionless)")
        axes[3, 0].set_xlabel("Supervised optimizer step")
        axes[3, 0].set_title("Actor weight displacement since initial\n‖weights − initial‖₂ / ‖initial‖₂", fontsize=10)
        axes[3, 1].axis("off")
        selection = run["probe_selection"]
        text = [f"{phase.upper()} · new full-FIT model", "Fixed unlabelled 93e9 probe choices"]
        for act, meta in selection["by_act"].items():
            text.append(f"Act {act}: {meta['selected']} probes / {len(meta['source_worlds'])} source worlds")
        text.extend([f"Act 3 covers {len(selection['by_act']['3']['source_worlds'])} worlds; descriptive coverage.",
                     "Critic explained variance: N/A", "PPO ratio / clipping: N/A (supervised warm-up)"])
        axes[3, 1].text(.02, .95, "\n".join(text), va="top", fontsize=11, linespacing=1.7)
        fig.legend(handles=[Line2D([0], [0], color="#333333", ls="--", label="Pooled probes"),
            *[Line2D([0], [0], color=COLORS[i], label=f"Act {i+1}") for i in range(3)]],
            loc="upper center", bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
        finish(fig, out, f"fixed93e9_movement_{phase}", f"Fixed 93e9 movement probes · new full {phase.upper()}",
            "fresh_start_v1 · actor-only exact-argmax singleton CPU probes; unlabelled, excluded from training and selection.\n"
            "Cadence: 100 supervised optimizer steps per logged block, NOT PPO updates. Previous block at first log means initialization.\n"
            "KL direction is reference ‖ current; curves are logged point diagnostics. Weight displacement is global actor L2 change, not per-act or per-block.",
            data.get("fixture", False))


def forest(ax, stats, labels, xlabel, *, scale=1, margin=False, colors=None):
    for i, stat in enumerate(stats):
        interval(ax, stat, i, color=(colors or COLORS)[i % len(colors or COLORS)], scale=scale)
    ax.axvline(0, color="#555555", lw=1, ls=":", label="Zero difference")
    if margin:
        ax.axvline(-1, color="#777777", lw=1.2, ls="--", label="Win NI margin: −1 pp")
        ax.legend(loc="best", fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def games(data, out):
    cohort_sizes = " / ".join(f"{n:,}" for n in sorted({v["cohort_worlds"] for v in data["g2"].values()}))
    labels = [ARM_LABELS[a] for a in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, scale, xlabel in zip(axes, ("win", "floor"), (100, 1),
        ("Win difference (percentage points)", "Mean floor difference (floors)")):
        forest(ax, [data["g2"][a][metric] for a in ARMS], labels, xlabel, scale=scale, margin=metric == "win")
        ax.set_title("G2 · student minus champion")
    finish(fig, out, "g2_paired_differences", "G2 · paired full-game differences",
        f"fresh_start_v1 · completed greedy exact-argmax full-game DEV evaluation · all {cohort_sizes} paired seeds per contrast.\n"
        "Direct paired difference estimates and 95% whole-WORLD bootstrap CIs from strict comparison JSONs. Win NI margin: 1 percentage point only.\n"
        "Mean floor has no margin. Greedy evaluation is not stochastic training game win rate; no gate verdict is assigned here.\n◇ = degenerate empirical bootstrap (constant observed contrast); it does not imply zero population uncertainty.",
        data.get("fixture", False))
    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5))
    metrics = (("reach", "Boss reach difference (pp)", 100), ("pass_marginal", "Boss pass / all worlds difference (pp)", 100),
               ("pass_conditional", "Boss pass / reached difference (pp)", 100),
               ("entry_hp_conditional", "Mean boss entry HP difference (HP)", 1))
    for i, act in enumerate(("1", "2", "3")):
        for j, (metric, xlabel, scale) in enumerate(metrics):
            labels = [f"{ARM_LABELS[a]}\nreached S {data['g2'][a]['bosses'][act]['student_reached']:,} / "
                      f"C {data['g2'][a]['bosses'][act]['champion_reached']:,}" for a in ARMS]
            forest(axes[i, j], [data["g2"][a]["bosses"][act][metric] for a in ARMS], labels, xlabel, scale=scale)
            axes[i, j].set_title("Act " + act)
    finish(fig, out, "g2_boss_differences", "G2 · boss outcomes and entry HP (descriptive; no margins)",
        f"fresh_start_v1 · greedy full-game DEV evaluation, {cohort_sizes} paired worlds per contrast; 95% WORLD bootstrap CIs from strict comparisons.\n"
        "S = student; C = champion. Marginal reach/pass use all cohort worlds. Conditional pass and entry HP use each arm's reached worlds.\n"
        "Reach cohorts depend on policy: conditional contrasts are descriptive, not causal. No NI margin for boss rates or HP.\n◇ = degenerate empirical bootstrap (constant observed contrast); it does not imply zero population uncertainty.",
        data.get("fixture", False))


def panel(data, out):
    groups = data["g1"]["groups"]
    ordered = sorted(groups, key=lambda key: (key != "BOSS2:resampled", key))
    labels = [f"{key.replace(':', ' · ')}{' [PRIMARY]' if groups[key]['role'] == 'primary' else ' [descriptive]'}"
              f"\n{groups[key]['win']['worlds']} worlds / {groups[key]['win']['rows']} cells"
              + (" · few worlds" if groups[key].get("few_worlds_descriptive") else "") for key in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, .64 * len(ordered) + 2.4)))
    for ax, metric, scale, xlabel in zip(axes, ("win", "hp_lost"), (100, 1),
        ("New minus old win difference (pp)", "New minus old pre-healing HP loss (HP)")):
        forest(ax, [groups[key][metric] for key in ordered], labels, xlabel, scale=scale,
               colors=[COLORS[0] if groups[key]["role"] == "primary" else "#666666" for key in ordered])
        ax.set_title("G1 direct paired replacement contrast")
    finish(fig, out, "g1_direct_panel", "G1 · new minus old combat replacement on the same frozen panel",
        "fresh_start_v1 · BOSS2 resampled win and HP loss are primary for full fits; all other groups and unsalted cells descriptive.\n"
        "95% whole-source-WORLD CIs, equal-world weighting; all salts/starts within a WORLD stay together. Salt −1 is separate from salts 0–7.\n"
        "HP loss = initial salted entry HP − battle-exit HP BEFORE end-of-fight relic healing; deaths use zero exit HP. Negative HP loss difference favors new. No margins.\n◇ = degenerate empirical bootstrap (constant observed contrast); it does not imply zero population uncertainty.",
        data.get("fixture", False))


def behaviors(data, out):
    cohort_sizes = " / ".join(f"{n:,}" for n in sorted({v["cohort_worlds"] for v in data["g2"].values()}))
    keys = tuple(BEHAVIOR_LABELS)
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    for ax, key in zip(axes.flat, keys):
        stats, labels = [], []
        for arm in ARMS:
            value = data["g2"][arm]["behaviors"][key]
            stats.append(value["chosen_per_opportunity"]["student_minus_champion"])
            s, c = (value["denominators"][a] for a in ("student", "champion"))
            labels.append(f"{ARM_LABELS[arm]}\nS {s['chosen']:,}/{s['opportunities']:,}; C {c['chosen']:,}/{c['opportunities']:,}")
        forest(ax, stats, labels, "Student minus champion chosen / opportunity (pp)", scale=100)
        title = BEHAVIOR_LABELS[key]
        if key == "removal" and all(
            data["g2"][arm]["behaviors"][key]["denominators"][a]["chosen"] == 0
            and data["g2"][arm]["behaviors"][key]["denominators"][a]["opportunities"] > 0
            for arm in ARMS for a in ("student", "champion")):
            title += "\nZero observed purchases · degenerate empirical bootstrap"
        ax.set_title(title, fontsize=11)
    axes.flat[-1].axis("off")
    note = axes.flat[-1].text(.02, .98,
        "Observed menus: ≥2 legal actions + opportunity.\nOther options may be unavailable.\n"
        "Smith: high HP; rest may be unavailable.\nPotions: per combat menu, not per game/fight.\nRemoval: shop purchases only.\n"
        "Card skip: explicit 'skip card reward' only; proceeding\npast an untaken card is not counted.\n"
        "Each gold/potion/relic pick on the same rewards\nscreen adds an opportunity.\n"
        "S/C = student/champion chosen/opportunities.\nPolicy-dependent; descriptive, not causal.\nNo pure free-choice propensity claim.\n"
        "◇ = degenerate empirical bootstrap, including [0,0].\nZero purchases do not establish population absence\nor zero uncertainty. N/A = undefined, never zero.",
        va="top", fontsize=10, linespacing=1.25)
    note.set_in_layout(False)  # Keep the explanatory panel from shrinking all four rows.
    finish(fig, out, "g2_behavior_rates", "G2 · observed behavior opportunity ratios",
        f"fresh_start_v1 · completed greedy full-game DEV evaluation · {cohort_sizes} paired worlds per comparison, original champion baseline.\n"
        "Ratios pool chosen/opportunities per arm. Paired student-minus-champion 95% WORLD bootstrap resamples both arms' components together.\n"
        "Denominators shown explicitly. Hollow diamonds mark degenerate empirical bootstrap CIs; they cannot quantify unobserved-event uncertainty.\n"
        "Policy-dependent opportunity cohorts are descriptive, not causal. No margins; no training game win-rate claim.",
        data.get("fixture", False))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    stats, labels = [], []
    for arm in ARMS:
        value = data["g2"][arm]["behaviors"]["shop_spend"]
        stats.append(value["gold_per_opportunity"]["student_minus_champion"])
        s, c = (value["denominators"][a] for a in ("student", "champion"))
        labels.append(f"{ARM_LABELS[arm]}\nS {s['amount_gold']:,.0f} gold / {s['opportunities']:,} opps; "
                      f"C {c['amount_gold']:,.0f} gold / {c['opportunities']:,} opps")
    forest(ax, stats, labels, "Student minus champion mean gold spent per shop opportunity (gold / opportunity)")
    finish(fig, out, "g2_shop_gold", "G2 · shop gold amount per opportunity",
        f"fresh_start_v1 · greedy full-game DEV evaluation · {cohort_sizes} paired worlds; 95% paired WORLD bootstrap of summed amount/opportunities.\n"
        "Includes zero spend at eligible purchase-choice menus; denominator is opportunities, not purchases. S = student; C = original champion.\n"
        "Policy-dependent opportunity cohorts: descriptive, not causal. No margin.\n"
        "◇ = degenerate empirical bootstrap (constant observed contrast); it does not imply zero population uncertainty.",
        data.get("fixture", False))


def tables(data, out):
    with (out / "test_strata.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "stratum", "metric", "unit", "point", "ci95_low", "ci95_high", "rows", "worlds", "weighting", "cluster_unit"))
        for name, model in data["scores"].items():
            for stratum, metrics in model["groups"].items():
                for metric, stat in metrics.items():
                    writer.writerow((name, stratum, metric, "fraction" if metric == "agreement" else "nats/row",
                        stat["point"], *stat["ci95"], stat["rows"], stat["worlds"], stat["weighting"], stat["cluster_unit"]))
    with (out / "behavior_components.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("comparison", "behavior", "metric", "unit", "student_point", "champion_point", "paired_difference",
            "diff_ci95_low", "diff_ci95_high", "student_chosen", "student_opps", "champion_chosen", "champion_opps",
            "student_gold", "champion_gold", "student_worlds_with_opps", "champion_worlds_with_opps", "cohort_worlds", "valid_bootstraps",
            "student_bootstrap_degenerate", "champion_bootstrap_degenerate", "bootstrap_degenerate",
            "behavior_label", "opportunity_definition", "choice_note", "uncertainty_note"))
        for name, comparison in data["g2"].items():
            for key, value in comparison["behaviors"].items():
                s, c = (value["denominators"][a] for a in ("student", "champion"))
                for metric in ("chosen_per_opportunity", "gold_per_opportunity"):
                    if metric not in value:
                        continue
                    estimates = value[metric]
                    diff = estimates["student_minus_champion"]
                    writer.writerow((name, key, metric, "fraction" if metric == "chosen_per_opportunity" else "gold/opportunity",
                        estimates["student"]["point"], estimates["champion"]["point"], diff["point"], *diff["ci95"],
                        s["chosen"], s["opportunities"], c["chosen"], c["opportunities"], s["amount_gold"], c["amount_gold"],
                        s["worlds_with_opportunities"], c["worlds_with_opportunities"], s["cohort_worlds"], diff["bootstrap_valid"],
                        estimates["student"]["bootstrap_degenerate"], estimates["champion"]["bootstrap_degenerate"],
                        diff["bootstrap_degenerate"], BEHAVIOR_LABELS[key], value["opportunity_definition"], value["note"],
                        value.get("uncertainty_note", "") if diff["bootstrap_degenerate"] else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="plot_data.json exported in frozen runtime")
    parser.add_argument("--out", type=Path, required=True, help="Fresh artifact directory (must not exist)")
    args = parser.parse_args(argv)
    try:
        raw = args.data.read_bytes()
        data = json.loads(raw)
        if data.get("schema") != "fresh_gate_plot_data_v1":
            raise ValueError("Require fresh_gate_plot_data_v1 dataset")
        # Full family required: no selective partial export from missing inputs.
        for key in ("scores", "training", "teacher_reference", "g2", "g1", "g1_labels", "statistics", "protocol"):
            if not data.get(key):
                raise ValueError("Incomplete plotting dataset: " + key)
        if set(data["g2"]) != set(ARMS) or "BOSS2:resampled" not in data["g1"]["groups"]:
            raise ValueError("Incomplete G1/G2 plotting dataset")
        out = args.out.resolve()
        if out.exists():
            raise FileExistsError("--out must be a fresh directory")
        study = Path(data.get("study", "/nonexistent-readonly-study")).resolve()
        if out.is_relative_to(study):
            raise ValueError("Output must be outside read-only study")
        style()
        # Stage all artifacts privately: a plotting failure leaves no partial final directory.
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fresh-gate-render-", dir=out.parent) as temporary:
            stage = Path(temporary) / "artifacts"
            stage.mkdir()
            for render in (learning, validation, probes, games, panel, behaviors, tables):
                render(data, stage)
            provenance = {"dataset": str(args.data.resolve()), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "schema": data["schema"], "protocol": data["protocol"], "matplotlib": matplotlib.__version__,
                "analysis_runtime": data.get("runtime"), "fixture": data.get("fixture", False),
                "statistics": data["statistics"], "notes": data.get("notes", []),
                "artifacts": sorted(p.name for p in stage.iterdir())}
            (stage / "artifact_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            # Recheck to refuse ordinary reuse, including while rendering.
            if out.exists():
                raise FileExistsError("Output appeared while rendering")
            stage.rename(out)
        print(json.dumps({"artifacts": str(out), "figures": 9, "formats": ["PNG 300 dpi", "SVG"]}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"plot_fresh_gate: {exc}\n")


if __name__ == "__main__":
    main()
