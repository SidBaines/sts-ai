"""Freeze the development nominee and score the two independent performance tests."""
import json
import shutil
from datetime import datetime, timezone

import torch
from sts_policy.model_agent import CompositeActorCritic
from sts_rl.boss_refits import primary_nominee
from sts_rl.common import load_policy
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import compare, compare_boss_replays, digest
from study import CFG, CHAMP, HERE, OUT, RUNS, check_parent, note


def read(path):
    return json.loads(path.read_bytes())


def freeze():
    selection = OUT / "selection.json"
    if selection.exists():
        return read(selection)
    a, b = "mix25", "mix50"
    start, count = CFG["development"]
    paired = compare(OUT/"development"/a, OUT/"development"/b, start, a, b,
                     OUT/a/"player.pt", OUT/b/"player.pt", count)
    chosen = primary_nominee(paired, a, b)
    fitted = read(OUT/chosen/"summary.json")
    dest = OUT/"finalist"
    dest.mkdir(exist_ok=True)
    for kind in ("player", "head"):
        source = OUT/chosen/(kind+".pt")
        target = dest/(kind+".pt")
        tmp = target.with_suffix(".copying")
        shutil.copyfile(source, tmp)
        if digest(tmp) != fitted[kind+"_sha256"]:
            raise ValueError("Candidate differs from the validation-selected checkpoint")
        tmp.replace(target)
    result = dict(chosen=chosen, boss_share=fitted["boss_share"], comparison=paired,
                  player_sha256=fitted["player_sha256"], head_sha256=fitted["head_sha256"])
    atomic_json(selection, result)
    note("finalist frozen", f"On {count:,} development worlds the 25% boss mix won {paired['first']['wins']:,} "
         f"and the 50% mix won {paired['second']['wins']:,}. Froze the {fitted['boss_share']:.0%} mix for both independent tests.")
    return result


def verify_player(selection):
    check_parent()
    for kind in ("player", "head"):
        if digest(OUT/"finalist"/(kind+".pt")) != selection[kind+"_sha256"]:
            raise ValueError("Frozen finalist changed")
    champion, _ = load_policy(CHAMP)
    player, _ = load_policy(OUT/"finalist/player.pt")
    head, _ = load_policy(OUT/"finalist/head.pt")
    if type(player) is not CompositeActorCritic or set(player._modules) != {"combat","ooc"}:
        raise ValueError("The finalist has an unexpected architecture")
    for a,b in ((champion.ooc, player.ooc),(player.combat, head)):
        first, second = a.state_dict(), b.state_dict()
        if first.keys() != second.keys() or any(not torch.equal(t, second[k]) for k,t in first.items()):
            raise ValueError("Strategy changed or the boss panel used a different fighter")


def boss_scores(selection):
    return compare_boss_replays(OUT/"grid.json", OUT/"boss-fights", CFG["champion_sha256"],
                                selection["head_sha256"], CFG["boss_starts"])


def report():
    selection = read(OUT/"selection.json")
    verify_player(selection)
    start, n = CFG["confirmation"]
    player = OUT/"finalist/player.pt"
    full = compare(OUT/"confirmation/champion", OUT/"confirmation/student", start,
                   "champion","student",CHAMP,player,n)
    bosses, valid_panel = boss_scores(selection)
    full_lower = full["worst_case_ci95_pp"][0]
    boss_lower = bosses["original"]["worst_case_ci95_pp"][0]
    retained = valid_panel and full_lower > -1 and boss_lower > -3
    promoted = retained and full_lower > 0
    nominal = full.get("difference_pp",full.get("complete_pair_difference_pp"))
    bounds = full.get("ci95_pp",full.get("complete_pair_ci95_pp"))
    evidence = (f"Single fighter {full['second']['wins']:,} wins, champion {full['first']['wins']:,} "
                f"on {n:,} full games ({nominal:+.2f} points, 95% interval {bounds[0]:+.2f} to {bounds[1]:+.2f}).")
    outcome = dict(full_games=full, bosses=bosses, boss_panel_valid=valid_panel, strength_retained=retained,
                   promoted=promoted, finalist=selection["chosen"], checkpoint=str(player))
    if promoted:
        dest = CHAMP.with_name("champion_"+selection["player_sha256"][:12]+".pt")
        if not dest.exists():
            tmp = dest.with_suffix(".copying")
            shutil.copyfile(player,tmp)
            if digest(tmp) != selection["player_sha256"]:
                raise ValueError("Incomplete champion archive")
            tmp.replace(dest)
        if digest(dest) != selection["player_sha256"]:
            raise ValueError("Champion archive mismatch")
        finished = max((OUT/"confirmation"/arm/"summary.json").stat().st_mtime for arm in ("champion","student"))
        day = datetime.fromtimestamp(finished,timezone.utc).date().isoformat()
        atomic_json(dest.with_suffix(".json"),dict(promoted=day,
                    by="Single-fighter full-game superiority and boss retention",source=str(player.relative_to(RUNS)),
                    sha256=selection["player_sha256"],parent_sha256=CFG["champion_sha256"],environment="env-v0.1.5.2",
                    baseline_games={a:str(OUT/"confirmation"/a) for a in ("champion","student")},
                    evidence=evidence))
    for name, counts in (("champion",full["first"]),("student",full["second"])):
        label = "Early-refit champion" if name=="champion" else "Single fighter"
        label += " · single-fighter comparison"
        if counts["completed"] != n:
            label += f" ({n-counts['completed']} unresolved)"
        atomic_json(RUNS/"fresh_eval"/f"{HERE.name}-{name}.json",
                    dict(label=label,assigned=n,
                         summary=dict(n=counts["completed"],wins=counts["wins"],mode="greedy",mean_floor=counts["mean_floor"])))
    note("finished", evidence + " "
         f"On {CFG['boss_starts']:,} original boss entries they won {bosses['original']['student_wins']} and "
         f"{bosses['original']['champion_wins']}. " + ("The one-fighter player is champion." if promoted else "The early-refit champion retains the title."))
    atomic_json(OUT/"verdict.json", outcome)
    return outcome
