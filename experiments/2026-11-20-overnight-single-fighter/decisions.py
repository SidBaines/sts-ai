"""Numerical decisions and immutable checkpoints for the overnight study."""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from sts_policy.model_agent import CompositeActorCritic
from sts_rl.boss_refits import primary_nominee
from sts_rl.combat_distillation import choose_imitation_continuation
from sts_rl.common import load_policy
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import compare, compare_boss_replays, digest
from study import CFG, CHAMP, OLD, OUT, RUNS, models, note, read


def estimate(result):
    return (
        result.get("difference_pp", result.get("complete_pair_difference_pp")),
        result.get("ci95_pp", result.get("complete_pair_ci95_pp")),
    )


def evidence(result):
    value, bounds = estimate(result)
    if bounds is None:
        return f"insufficient completed pairs out of {result['n']:,}"
    return (
        f"{result['second']['wins']:,} vs {result['first']['wins']:,} wins, "
        f"{result['paired_completed']:,}/{result['n']:,} complete pairs, "
        f"{value:+.2f} points [{bounds[0]:+.2f}, {bounds[1]:+.2f}]"
    )


def freeze_file(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f"Frozen source changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_name(target.name + f".tmp-{os.getpid()}")
        shutil.copyfile(source, temp)
        if digest(temp) != expected:
            raise IOError(f"Incomplete copy of {source}")
        temp.replace(target)
    if digest(target) != expected:
        raise ValueError(f"Conflicting frozen checkpoint: {target}")


def compare_models(stage, checkpoints, baseline, challenger, folder=None):
    start, n = CFG[stage]
    dest = OUT / (folder or stage)
    return compare(
        dest / baseline,
        dest / challenger,
        start,
        baseline,
        challenger,
        checkpoints[baseline],
        checkpoints[challenger],
        n,
    )


def loss(summary):
    return sum(v["kl"] for v in summary["validation"].values()) / 2


def first_selection():
    path = OUT / "first-selection.json"
    if path.exists():
        return read(path)
    choices = models()
    checkpoints = {name: folder / "player.pt" for name, folder in choices.items()}
    refs = read(OLD / "mix25/summary.json")
    measured = {}
    losses = {"original": loss(refs)}
    for step in CFG["snapshots"]:
        name = f"step-{step}"
        info = read(choices[name] / "summary.json")
        if digest(checkpoints[name]) != info["player_sha256"]:
            raise ValueError("Screen candidate changed")
        losses[name] = loss(info)
        measured[name] = compare_models("screen", checkpoints, "original", name)
    winner, actor, feedback = choose_imitation_continuation(measured, losses)
    result = dict(
        chosen=winner,
        feedback_actor=actor,
        feedback=feedback,
        comparisons=measured,
        imitation_loss=losses,
    )
    atomic_json(path, result)
    for name, r in measured.items():
        note(
            "imitation selection",
            f"{name} against the original student on new selection worlds: {evidence(r)}.",
        )
    note(
        "imitation selection",
        f"First nominee: {winner}. Student-played feedback {'triggered' if feedback else 'not triggered'}.",
    )
    return result


def seal_student():
    path = OUT / "frozen.json"
    if path.exists():
        return read(path)
    result = first_selection()
    winner = result["chosen"]
    check = None
    if result["feedback"]:
        ck = {
            "previous": models()[winner] / "player.pt",
            "feedback": models()["feedback"] / "player.pt",
        }
        check = compare_models("feedback_screen", ck, "previous", "feedback")
        if primary_nominee(check, "previous", "feedback") == "feedback":
            winner = "feedback"
        note(
            "feedback selection", f"Student-played correction vs the first nominee: {evidence(check)}."
        )
    source = models()[winner]
    info = read(OLD / "selection.json") if winner == "original" else read(source / "summary.json")
    single = OUT / "finalist"
    for kind in ("player", "head"):
        freeze_file(source / (kind + ".pt"), single / (kind + ".pt"), info[kind + "_sha256"])
    student, _ = load_policy(single / "player.pt")
    teacher, _ = load_policy(CHAMP)
    head, _ = load_policy(single / "head.pt")
    if type(student) is not CompositeActorCritic:
        raise ValueError("Finalist must contain exactly one combat network")
    for a, b in ((student.ooc, teacher.ooc), (student.combat, head)):
        one, two = a.state_dict(), b.state_dict()
        if one.keys() != two.keys() or any(not torch.equal(v, two[k]) for k, v in one.items()):
            raise ValueError("Frozen fighter or strategy export mismatch")
    selected = dict(
        chosen=winner,
        player_sha256=info["player_sha256"],
        head_sha256=info["head_sha256"],
        feedback_screen=check,
    )
    atomic_json(path, selected)
    note("student frozen", f"Froze {winner} before opening the 49,152-world and boss replay tests.")
    return selected


def archive(source, incumbent, parent, result, method):
    sha = digest(source)
    stored = RUNS / "champions" / f"champion_{sha[:12]}.pt"
    freeze_file(source, stored, sha)
    atomic_json(
        stored.with_suffix(".json"),
        dict(
            sha256=sha,
            parent_sha256=digest(parent),
            replaced_sha256=digest(incumbent),
            source=str(source.relative_to(RUNS)),
            promoted=datetime.now(timezone.utc).date().isoformat(),
            by=method,
            environment="env-v0.1.5.2",
            evidence=evidence(result),
        ),
    )
    return stored


def dashboard_panel(tag, pairs):
    for name, counts in pairs.items():
        atomic_json(
            RUNS / "fresh_eval" / f"{CFG['run']}-{tag}-{name}.json",
            dict(
                label=f"{name} · overnight {tag}",
                summary=dict(
                    n=counts["completed"],
                    wins=counts["wins"],
                    mode="greedy",
                    mean_floor=counts["mean_floor"],
                ),
            ),
        )


def retention():
    path = OUT / "retention-verdict.json"
    if path.exists():
        return read(path)
    selected = read(OUT / "frozen.json")
    ck = {"champion": CHAMP, "student": OUT / "finalist/player.pt"}
    if (
        digest(ck["student"]) != selected["player_sha256"]
        or digest(OUT / "finalist/head.pt") != selected["head_sha256"]
    ):
        raise ValueError("Finalist changed")
    full = compare_models("retention", ck, "champion", "student")
    bosses, valid = compare_boss_replays(
        OUT / "grid.json",
        OUT / "boss-fights",
        CFG["champion_sha256"],
        selected["head_sha256"],
        CFG["boss_starts"],
    )
    lower = full["worst_case_ci95_pp"][0]
    retained = valid and lower > -1 and bosses["original"]["worst_case_ci95_pp"][0] > -3
    promote = retained and lower > 0
    incumbent = (
        archive(ck["student"], CHAMP, CHAMP, full, "Independent one-fighter retention study")
        if promote
        else CHAMP
    )
    answer = dict(
        full=full,
        bosses=bosses,
        valid_panel=valid,
        retained=retained,
        promoted=promote,
        incumbent=str(incumbent),
        incumbent_sha256=digest(incumbent),
    )
    dashboard_panel("retention", {"champion": full["first"], "student": full["second"]})
    note(
        "retention result",
        f"Student vs champion: {evidence(full)}. Both retention margins met: {retained}.",
    )
    for label, r in bosses.items():
        delta, ci = estimate(r)
        text = f"{delta:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "interval unavailable"
        note(
            "boss result",
            f"{label.capitalize()} boss starts: {r['student_wins']} student and {r['champion_wins']} champion wins on {r['n']:,} entries, {text}.",
        )
    atomic_json(path, answer)
    return answer


def endpoints():
    return {
        "two": OUT / "rl-two/iter_0024.pt",
        "one": OUT / "rl-one/iter_0024.pt",
        "two-parent": CHAMP,
        "one-parent": OUT / "finalist/player.pt",
    }


def choose_endpoint():
    outfile = OUT / "rl-selection.json"
    if outfile.exists():
        return read(outfile)
    ck = endpoints()
    for name in ("one", "two"):
        saved = read(OUT / f"rl-{name}" / "finished.json")
        if saved["sha256"] != digest(ck[name]) or saved["parent_sha256"] != digest(
            ck[name + "-parent"]
        ):
            raise ValueError("Trained endpoint or parent changed")
    results = {}
    for a, b in (("two", "one"), ("one-parent", "one"), ("two-parent", "two")):
        results[f"{b}-vs-{a}"] = compare_models("rl_panel", ck, a, b)
    best = primary_nominee(results["one-vs-two"], "two", "one")
    for label, r in results.items():
        note("trained endpoints", f"{label} on the update-24 panel: {evidence(r)}.")
    result = dict(chosen=best, sha256=digest(ck[best]), comparisons=results)
    atomic_json(outfile, result)
    return result


def finish():
    selected = choose_endpoint()
    base = retention()
    parent = Path(base["incumbent"])
    winner = selected["chosen"]
    ck = {"incumbent": parent, "nominee": endpoints()[winner]}
    if digest(parent) != base["incumbent_sha256"] or digest(ck["nominee"]) != selected["sha256"]:
        raise ValueError("Confirmation checkpoint changed")
    result = compare_models("rl_confirmation", ck, "incumbent", "nominee")
    promoted = result["worst_case_ci95_pp"][0] > 0
    champion = (
        archive(
            ck["nominee"],
            parent,
            endpoints()[winner + "-parent"],
            result,
            "Overnight joint-learning confirmation",
        )
        if promoted
        else parent
    )
    dashboard_panel(
        "joint-learning confirmation",
        {"incumbent": result["first"], winner + " fighters": result["second"]},
    )
    msg = (
        f"Overnight finished. {winner}-fighter nominee against the incumbent: {evidence(result)}. "
        + ("The nominee is champion." if promoted else "The incumbent keeps the title.")
    )
    note("finished", msg)
    answer = dict(chosen=winner, paired=result, promoted=promoted, champion=str(champion), message=msg)
    atomic_json(OUT / "verdict.json", answer)
    return answer
