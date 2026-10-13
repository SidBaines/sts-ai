"""Snapshot the three Gemma arms and produce a captioned comparison report."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
from itertools import combinations
import json
from pathlib import Path
import shutil
import textwrap

from sts_ai import fight_lab as lab
from sts_ai.fight_comparison import ARM_NAMES, ARM_LABELS, compare, mean, outcome
from sts_ai.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/experiments/single_fight_v1/search_10min_2026-09-06/runs"
DEFAULTS = dict(zip(ARM_NAMES, (RUNS / name for name in (
    "gemma4_t1_p095_k64_v1", "gemma4_t1_p095_k64_public_v4_v1",
    "gemma4_t1_p095_k64_public_v4_thinking_v1"))))
COLORS = dict(zip(ARM_NAMES, ("#65748e", "#228878", "#8e59bb")))
OUTCOMES = {"Win": "#27916f", "Death": "#c74c52", "Invalid action": "#d69e38",
            "Timeout": "#9868ba", "Other incomplete": "#dc7e53", "Not finished": "#dde2e9"}
SHORT = ("Old format\nNo thinking", "New format\nNo thinking", "New format\nThinking")


def csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def ordered_pairs(pairs: dict[str, dict]) -> dict[str, dict]:
    """Keep report/plot order stable across sorted-key JSON snapshots."""
    return {f"{left}__{right}": pairs[f"{left}__{right}"]
            for left, right in combinations(ARM_NAMES, 2)}


def build(runs: dict[str, Path], output: Path, snapshot: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = lab.read_json(snapshot) if snapshot is not None else compare(runs)
    data["pairs"] = ordered_pairs(data["pairs"])
    if snapshot is not None:
        data["source_snapshot_sha256"] = file_sha256(snapshot)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / "snapshots" / now
    plots = destination / "plots"
    plots.mkdir(parents=True)
    cohort, arms = data["cohort"], data["by_arm"]
    all_done = all(data["run_status"][a].get("progress.json", {}).get("status") == "finished"
                   and len(arms[a]) == len(cohort)
                   for a in ("new_no_thinking", "new_thinking"))
    all_done = all_done and data["run_status"]["new_thinking"].get("queue_status.json", {}).get("status", "finished") == "finished"
    stage = "Final" if all_done else "Interim"
    data.update(created_at_utc=now, stage=stage, plotting_sha256=file_sha256(__file__))
    lab.write_new(destination / "comparison.json", data)
    csv_rows(destination / "arm_summary.csv", [{"arm": a, **data["summaries"][a]} for a in ARM_NAMES])
    csv_rows(destination / "common_cohort_summary.csv", [{"arm": a, **data["common_summary"][a]} for a in ARM_NAMES])
    csv_rows(destination / "fights.csv", [{"arm": a, **r} for a in ARM_NAMES for r in arms[a].values()])
    csv_rows(destination / "decisions.csv", [{"arm": a, **r} for a in ARM_NAMES for r in data["decisions"][a]])
    for key, pair in data["pairs"].items():
        csv_rows(destination / f"paired_{key}.csv", pair["rows"])
    for source in (Path(__file__), ROOT / "src/sts_ai/fight_comparison.py", ROOT / "src/sts_ai/fight_analysis.py"):
        shutil.copy2(source, destination / source.name)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    gallery = []

    def save(fig, slug, title, caption):
        if data.get("infrastructure_excluded_ids"):
            if slug.startswith(("02_", "04_", "05_", "enemy_", "hp_", "transitions_")):
                caption += " Documented infrastructure interruptions are excluded."
            elif slug in ("01_coverage", "03_outcome_matrix", "20_running_win_rate"):
                caption += " Historical coverage retains the annotated system-sleep timeout."
        wrapped_title = textwrap.fill(title, max(40, int(fig.get_figwidth() * 7.5)))
        fig.suptitle(wrapped_title, fontsize=15, fontweight="bold", x=.06, ha="left", y=.98)
        fig.tight_layout(rect=(.02, .16, .98, .87 if "\n" in wrapped_title else .91))
        fig.text(.06, .03, textwrap.fill(caption, 130), fontsize=8, va="bottom", color="#45556b")
        for ext in ("png", "svg"):
            fig.savefig(plots / f"{slug}.{ext}", dpi=150)
        plt.close(fig)
        gallery.append({"slug": slug, "title": title, "caption": caption})

    def blank(ax, message="No matched observations yet"):
        ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#758094")

    def bars(ax, values, ylabel):
        for i, (arm, value) in enumerate(zip(ARM_NAMES, values)):
            if value is not None:
                ax.bar(i, value, color=COLORS[arm], width=.6)
                ax.annotate(fmt(value), (i, value), xytext=(0, 4), textcoords="offset points", ha="center")
            else:
                ax.text(i, .03, "N/A", transform=ax.get_xaxis_transform(), ha="center")
        ax.set_xticks(range(3), SHORT)
        ax.set_xlim(-.6, 2.6)
        ax.set_ylabel(ylabel)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(3)
    for label, color in OUTCOMES.items():
        counts = [sum(outcome(arms[a].get(sid)) == label for sid in cohort) for a in ARM_NAMES]
        ax.bar(range(3), counts, bottom=bottom, color=color, label=label)
        bottom += np.array(counts)
    ax.set_xticks(range(3), SHORT)
    ax.set_ylabel("Fight starts")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.15))
    save(fig, "01_coverage", f"{stage} cohort coverage and outcomes",
         f"All {len(cohort)} selected starts per arm. Grey means no finished attempt, not a loss. The old arm has one retained interrupted fight; its other 201 attempts include invalid and timeout outcomes.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    common = data["common_summary"]
    bars(axes[0], [None if common[a]["win_rate"] is None else 100 * common[a]["win_rate"] for a in ARM_NAMES], "Successful attempts (%)")
    axes[0].set_ylim(0, 110)
    bars(axes[1], [common[a]["mean_net_hp_loss"] for a in ARM_NAMES], "Mean net HP loss")
    save(fig, "02_matched_primary", "Primary comparison on identical observed starts",
         f"Left: all three arms have finished on {len(data['common_finished_ids'])} starts; invalid/error/timeout attempts count as unsuccessful, not combat deaths. Right: all three resolved on {len(data['common_resolved_ids'])} starts. Lower HP loss is better; it includes healing. N/A is not zero.")

    fig, ax = plt.subplots(figsize=(10, 7))
    matrix = np.array([[list(OUTCOMES).index(outcome(arms[a].get(sid))) for a in ARM_NAMES] for sid in cohort])
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(list(OUTCOMES.values())), vmin=-.5, vmax=5.5)
    ax.set_xticks(range(3), SHORT)
    ax.set_ylabel("Fight index in fixed schedule")
    save(fig, "03_outcome_matrix", "Every selected fight, aligned across arms",
         "Rows follow the identical fixed schedule. Colours match the coverage plot: green win, red death, gold invalid, purple timeout, orange other incomplete, grey unfinished. Each row represents the same underlying combat start.")

    for metric, label, slug in (("mean_win_difference", "Change in successful-attempt fraction", "04_paired_win_changes"),
                                ("mean_hp_loss_difference", "Change in net HP loss", "05_paired_hp_changes")):
        fig, ax = plt.subplots(figsize=(11, 6))
        labels = []
        for i, pair in enumerate(data["pairs"].values()):
            summary = pair["summary"]
            labels.append(f"{ARM_LABELS[pair['right']]}\nminus {ARM_LABELS[pair['left']]}")
            value = summary[metric]
            if value is None:
                ax.text(0, i, " No matched data", va="center")
                continue
            ci = summary["win_difference_world_bootstrap_95" if metric == "mean_win_difference" else "hp_difference_world_bootstrap_95"]
            ax.plot(value, i, "o", color=COLORS[pair["right"]])
            if ci is not None:
                ax.plot(ci, [i, i], color=COLORS[pair["right"]])
            n = summary["paired_finished" if metric == "mean_win_difference" else "paired_resolved"]
            ax.annotate(f" n={n}", (value, i), xytext=(4, 8), textcoords="offset points", fontsize=8)
        ax.set_yticks(range(3), labels, fontsize=9)
        ax.axvline(0, color="#888", lw=1, linestyle="--")
        ax.set_xlabel(label)
        save(fig, slug, label + " on matched fights",
             "Each comparison uses only starts finished in both arms; HP additionally requires both fights to resolve. Intervals are deterministic 95% bootstrap intervals resampling whole worlds, since fights from one world are related. Win change: higher is better. HP-loss change: lower is better.")

    for pair_key, pair in data["pairs"].items():
        label = ARM_LABELS[pair["right"]] + " vs " + ARM_LABELS[pair["left"]]
        rows = pair["rows"]
        resolved = [r for r in rows if r["both_resolved"]]
        fig, ax = plt.subplots(figsize=(8, 7))
        if resolved:
            x, y = [r["left_hp_loss"] for r in resolved], [r["right_hp_loss"] for r in resolved]
            ax.scatter(x, y, alpha=.65, s=27, color=COLORS[pair["right"]])
            lo, hi = min(x + y) - 2, max(x + y) + 2
            ax.plot([lo, hi], [lo, hi], "--", color="#888")
        else:
            blank(ax)
        ax.set_xlabel(ARM_LABELS[pair["left"]] + " · net HP loss")
        ax.set_ylabel(ARM_LABELS[pair["right"]] + " · net HP loss")
        save(fig, f"hp_scatter_{pair_key}", label,
             f"One point per jointly resolved fight (n={len(resolved)}). Below the diagonal means the newer/right-hand arm retained more HP. Healing can make net loss negative; unresolved attempts are excluded, not assigned zero.")

        fig, ax = plt.subplots(figsize=(10, 6))
        if resolved:
            ax.hist([r["hp_loss_difference"] for r in resolved], bins=25, color=COLORS[pair["right"]], alpha=.85)
        else:
            blank(ax)
        ax.axvline(0, color="#888", linestyle="--")
        ax.set(xlabel="Right-hand arm minus left-hand arm · net HP loss", ylabel="Matched fights")
        save(fig, f"hp_delta_{pair_key}", label + " · HP changes",
             f"Distribution across {len(resolved)} jointly resolved fights. Negative values favour the right-hand arm. This conditional comparison should be read alongside invalid/death/timeout rates.")

        fig, ax = plt.subplots(figsize=(9, 7))
        labels = list(OUTCOMES)[:-1]
        counts = np.zeros((5, 5), dtype=int)
        for row in rows:
            counts[labels.index(row["left_outcome"]), labels.index(row["right_outcome"])] += 1
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())))
        for y in range(5):
            for x in range(5):
                ax.text(x, y, str(counts[y, x]), ha="center", va="center",
                        color="white" if counts[y, x] > counts.max() * .6 else "#25364b")
        ax.set_xticks(range(5), labels, rotation=25, ha="right")
        ax.set_yticks(range(5), labels)
        ax.set_xlabel(ARM_LABELS[pair["right"]]); ax.set_ylabel(ARM_LABELS[pair["left"]])
        save(fig, f"transitions_{pair_key}", label + " · outcome transitions",
             f"Matched finished attempts (n={len(rows)}), including failures. Each cell counts an old/left outcome becoming the new/right outcome. Unfinished attempts do not enter this matrix.")

    for key, title, xlabel, slug, logarithmic in (
        ("completion_tokens", "Generated tokens per decision", "Tokens, including thinking", "15_completion_tokens", True),
        ("prompt_tokens", "Prompt lengths", "Input tokens", "16_prompt_tokens", False),
        ("latency_s", "Decision latency", "Seconds per model call", "17_latency", True),
        ("thinking_tokens", "Recorded thinking tokens", "Thinking tokens per decision", "18_thinking_tokens", True)):
        fig, ax = plt.subplots(figsize=(10, 6))
        for arm in ARM_NAMES:
            values = sorted(d[key] for d in data["decisions"][arm])
            if values:
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS[arm], label=f"{ARM_LABELS[arm]} (n={len(values)})")
        if logarithmic:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=0)
        ax.set(xlabel=xlabel, ylabel="Fraction of recorded decisions")
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            blank(ax)
        save(fig, slug, title,
             "Empirical cumulative distributions across decisions in finished attempts. These are operational diagnostics, not matched-fight effect estimates: different policies produce different numbers of decisions. Thinking tokens are those extracted by the existing parser.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bars(axes[0], [data["summaries"][a]["mean_seconds"] for a in ARM_NAMES], "Mean seconds per finished attempt")
    bars(axes[1], [None if data["summaries"][a]["responses"] == 0 else 100 * data["summaries"][a]["cap_hits"] / data["summaries"][a]["responses"] for a in ARM_NAMES], "Truncation/budget indicators (%)")
    save(fig, "19_cost_and_caps", "Runtime and generation-budget diagnostics",
         "Means omit absent timeout durations. Indicators include saved truncated-thinking flags or unclosed preserved thought markers regardless of length, saved tokens at/above the arm's limit, or invalid replies within eight tokens below it. Counts re-tokenize decoded text; stop reasons are not preserved. Archived budget-calibration attempts are excluded.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for arm in ARM_NAMES:
        x, y, n, wins = [], [], 0, 0
        for i, sid in enumerate(cohort):
            if sid in arms[arm]:
                n += 1; wins += int(arms[arm][sid]["win"])
                x.append(i + 1); y.append(wins / n)
        if x:
            ax.plot(x, y, label=ARM_LABELS[arm], color=COLORS[arm])
    ax.set(xlabel="Index in fixed fight schedule", ylabel="Cumulative successful-attempt fraction", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    save(fig, "20_running_win_rate", "How observed success changes across the schedule",
         "Descriptive cumulative win fractions among finished attempts, including invalid/timeouts as unsuccessful. This is not a training curve: weights never change. Early prefixes can differ from the final cohort; use matched estimates for comparison.")

    enemy_names = sorted({e for sid in cohort for e in data["starts"][sid]["enemies"]})
    for enemy in enemy_names:
        ids = [sid for sid in data["common_finished_ids"] if enemy in data["starts"][sid]["enemies"]]
        resolved_ids = [sid for sid in data["common_resolved_ids"] if enemy in data["starts"][sid]["enemies"]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        bars(axes[0], [None if not ids else 100 * sum(arms[a][sid]["win"] for sid in ids) / len(ids) for a in ARM_NAMES], "Successful attempts (%)")
        axes[0].set_ylim(0, 110)
        bars(axes[1], [mean([arms[a][sid] for sid in resolved_ids], "hp_loss") for a in ARM_NAMES], "Mean net HP loss")
        save(fig, f"enemy_{enemy.lower()}", enemy.replace("_", " ").title() + " · matched comparison",
             f"All-three finished starts containing this enemy: n={len(ids)}; all-three resolved: n={len(resolved_ids)}. Mixed encounters appear in multiple enemy panels, so panel counts must not be added. These small subgroups are descriptive, not independent confirmations of generalization.")

    lines = [f"# {stage} comparison of three Gemma combat arms", "",
             f"Snapshot UTC: {now}. The comparison contains **{len(cohort)} fixed combat starts**.", "",
             "| Arm | Finished | Wins | Deaths | Invalid | Timeout | Other incomplete | Not finished |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARM_NAMES:
        s = data["summaries"][arm]
        lines.append(f"| {ARM_LABELS[arm]} | {s['finished']} | {s['wins']} | {s['deaths']} | {s['invalid']} | {s['timeouts']} | {s['other_incomplete']} | {s['not_finished']} |")
    lines += ["", "Historical counts above retain infrastructure interruptions. Matched model-performance estimates exclude annotated system-sleep timeouts, with their original records preserved.", "",
              "## Matched performance", "",
              f"Eligible starts finished in all three arms: **{len(data['common_finished_ids'])}**; jointly resolved: **{len(data['common_resolved_ids'])}**. Unfinished data is never treated as a loss or zero HP.", "",
              "| Arm | Win fraction, common finished starts | Mean net HP loss, common resolved starts |",
              "| --- | ---: | ---: |"]
    for arm in ARM_NAMES:
        s = common[arm]
        lines.append(f"| {ARM_LABELS[arm]} | {fmt(None if s['win_rate'] is None else 100*s['win_rate'])}% | {fmt(s['mean_net_hp_loss'])} |")
    lines += ["", "Pairwise effects use the right/newer arm minus the left/earlier arm. Positive win changes and negative HP-loss changes favour the newer arm. Intervals resample entire worlds (2,000 draws; fixed seed), retaining correlations among fights from the same world.", ""]
    for pair in data["pairs"].values():
        s = pair["summary"]
        lines += [f"### {ARM_LABELS[pair['right']]} versus {ARM_LABELS[pair['left']]}", "",
                  f"- Matched finished attempts: **{s['paired_finished']}**; wins gained/lost: **{s['wins_gained']} / {s['wins_lost']}**.",
                  f"- Infrastructure-interrupted matched starts excluded: **{len(pair.get('excluded_infrastructure_ids', []))}** (`{pair.get('excluded_infrastructure_ids', [])}`).",
                  f"- Successful-attempt change: **{fmt(None if s['mean_win_difference'] is None else 100*s['mean_win_difference'])} percentage points**; fraction-scale 95% interval: `{s['win_difference_world_bootstrap_95']}`.",
                  f"- Jointly resolved fights: **{s['paired_resolved']}**; mean net-HP-loss change: **{fmt(s['mean_hp_loss_difference'])} HP**; 95% interval: `{s['hp_difference_world_bootstrap_95']}`.", ""]
    lines += ["## Generation-budget diagnostics", "",
              "Counts re-tokenize decoded text; they do not preserve the generator's exact stop reason. Truncation/budget indicators count a saved thinking_truncated flag or unclosed preserved thought-channel markers, replies at/above the configured limit, or invalid replies within eight tokens below it. Channel completeness does not depend on re-tokenized length. The same rule applies to all three arms. The near-limit column includes both valid and invalid replies, since a valid action can coexist with a long or cut-off explanation.", "",
              "| Arm | Token limit | Responses | Count at/above limit | Count within eight below limit | Unclosed/truncated thinking | Truncation/budget indicators |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARM_NAMES:
        decisions = data["decisions"][arm]
        at_limit = sum(d["completion_tokens"] >= d["max_tokens"] for d in decisions)
        near_limit = sum(d["max_tokens"] - 8 <= d["completion_tokens"] < d["max_tokens"] for d in decisions)
        lines.append(f"| {ARM_LABELS[arm]} | {data['settings'][arm]['max_tokens']} | {len(decisions)} | {at_limit} | {near_limit} | {sum(bool(d.get("thinking_truncated", False)) for d in decisions)} | {data['summaries'][arm]['cap_hits']} |")
    recoveries = [(arm, item) for arm in ARM_NAMES
                  for item in data["run_status"][arm].get("infrastructure_recoveries", [])]
    annotations = [(arm, item) for arm in ARM_NAMES
                   for item in data["run_status"][arm].get("infrastructure_annotations", [])]
    if recoveries or annotations:
        lines += ["", "## Infrastructure interruptions", "",
                  "Machine-sleep time is not model computation time. Recovered attempts use their original seeds and preserve the interrupted attempt in an archive. Historical attempts that were not rerun remain in coverage counts but are excluded from matched model-performance estimates.", ""]
        for arm, item in annotations:
            lines.append(f"- Excluded historical attempt: {ARM_LABELS[arm]}, `{item['state_id']}`: {item['reason']}. Frozen trace/result hashes and evidence are in the snapshotted infrastructure annotation.")
        for arm, item in recoveries:
            lines.append(f"- {ARM_LABELS[arm]}, `{item['state_id']}`: {item['reason']}. Recovery record: `{item['path']}`.")
    lines += ["", "## Protocol and limits", "",
              "The pinned base model is identical across arms; no adapter or training update is involved. World starts and sampling seeds are matched, with temperature 1.0, top-p 0.95, top-k 64, and no output-format retry. The observation repair is a combined intervention, not an isolated test of any individual glossary change.", "",
              "The old arm stopped after 201 finished attempts and retains one interrupted fight. Its full original schedule was larger; this report restricts it to the same 202 starts selected for the new arms. The old partial is excluded from finished-pair comparisons. Invalid/error/timeout outcomes remain explicit; successful-attempt fractions include non-infrastructure failures as nonwins, whereas conditional combat win rates/HP metrics require resolution. Documented infrastructure interruptions are excluded from matched estimates, not counted as model failures.", "",
              "Net HP loss is starting HP minus final HP, including healing; negative values are possible. HP comparisons exclude unresolved outcomes, which can cause selection bias. Read them beside success/failure counts. Fights from one world share progression, so bootstrap resampling is by world. There is one sampled trajectory per start/arm and no held-out confirmation: this cohort was used to identify template issues.", "",
              "Thinking also has a larger generation budget. The adaptive first-20-generation check can archive calibration trajectories and restart the cohort at a larger limit. Those archives are preserved but excluded from the main comparison. This arm therefore tests thinking with its recorded budget, not thinking alone at an identical token cap. Generation-cap rates and queue/budget history are included in comparison.json.", "",
              "| Arm | Thinking | Completion limit | Finished-attempt mean seconds | Cap indicators / responses |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm in ARM_NAMES:
        s, cfg = data["summaries"][arm], data["settings"][arm]
        lines.append(f"| {ARM_LABELS[arm]} | {cfg['thinking']} | {cfg['max_tokens']} | {fmt(s['mean_seconds'])} | {s['cap_hits']} / {s['responses']} |")
    lines += ["", "## Reproduction", "",
              "`comparison.json` freezes the finished-result set, hashes, source provenance, settings, and calculated data. CSVs hold arm, fight, decision, and paired records. PNG and SVG plots have captions both in the images and gallery. Plotting and analysis source files are copied alongside this report.", "",
              "Rerun `PYTHONPATH=src .venv/bin/python scripts/compare_fight_formats.py` to create a new dated snapshot as results arrive. Earlier snapshots remain unchanged. Add `--snapshot /full/path/to/comparison.json` to rerender the saved analyzed data without rereading the live runs."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    lab.write_new(destination / "captions.json", gallery)
    body = "".join(f'<figure><h2>{html.escape(item["title"])}</h2><a href="plots/{item["slug"]}.svg"><img loading="lazy" src="plots/{item["slug"]}.png"></a><figcaption>{html.escape(item["caption"])}</figcaption></figure>' for item in gallery)
    (destination / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>Gemma three-arm comparison</title><style>body{{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#26344b}}img{{width:100%}}figure{{margin:35px 0;border-bottom:1px solid #ddd;padding-bottom:25px}}figcaption{{line-height:1.5;color:#526176}}a{{color:#5364a7}}</style><h1>{stage} Gemma three-arm comparison</h1><p>{now} · {len(cohort)} fixed starts · {len(gallery)} captioned plots</p><p><a href="REPORT.md">Written report</a> · <a href="comparison.json">Data and provenance</a> · <a href="arm_summary.csv">Summary CSV</a></p>{body}')
    relative = destination.relative_to(output)
    (output / "index.html").write_text(f'<!doctype html><meta http-equiv="refresh" content="0;url={relative}/index.html"><a href="{relative}/index.html">Latest comparison</a>')
    (output / "REPORT.md").write_text(f'# Latest comparison\n\n[{stage} report]({relative}/REPORT.md) · [Captioned gallery]({relative}/index.html)\n')
    sample = {"snapshot": str(destination.resolve()), "stage": stage,
              "finished": {a: data["summaries"][a]["finished"] for a in ARM_NAMES}, "plots": len(gallery)}
    (output / "latest.json").write_text(json.dumps(sample, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_NAMES:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path, default=DEFAULTS[arm])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gemma_three_arm_comparison_2026-09-06")
    parser.add_argument("--snapshot", type=Path, help="rerender frozen comparison.json data")
    args = parser.parse_args()
    print(build({a: getattr(args, a).resolve() for a in ARM_NAMES}, args.output.resolve(), args.snapshot))


if __name__ == "__main__":
    main()
