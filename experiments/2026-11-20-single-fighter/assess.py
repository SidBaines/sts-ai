"""Freeze the development nominee and score the two independent performance tests."""
import json
import shutil
from datetime import datetime, timezone

import torch
from sts_policy.model_agent import CompositeActorCritic
from sts_rl.boss_refits import primary_nominee
from sts_rl.common import load_policy
from sts_rl.fresh_panel_grid import canonical_grid_hash
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import compare, digest, paired_binary_outcomes
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
    grid, panel = read(OUT/"grid.json"), read(OUT/"boss-fights/summary.json")
    if not (grid["grid_hash"] == canonical_grid_hash(grid) == panel["grid_hash"]
            and panel["grid_input"]["sha256"] == digest(OUT/"grid.json")
            and grid["coverage"] == {"BOSS3": CFG["boss_starts"]}):
        raise ValueError("Final boss position roster mismatch")
    identity = panel["source"]["policy"]
    if identity["champion"]["sha256"] != CFG["champion_sha256"] or identity["combat"]["sha256"] != selection["head_sha256"]:
        raise ValueError("Unexpected panel fighter")
    result = {}
    for salt, label in ((-1, "original"), (0, "reseeded")):
        keys = {(r["world_seed"],r["start_id"]) for r in grid["grid"] if r["salt"] == salt}
        if len(keys) != CFG["boss_starts"] or len({s for s,_ in keys}) != len(keys):
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
    outcome = dict(full_games=full, bosses=bosses, boss_panel_valid=valid_panel, strength_retained=retained,
                   promoted=promoted, finalist=selection["chosen"], checkpoint=str(player))
    atomic_json(OUT/"verdict.json", outcome)
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
        atomic_json(dest.with_suffix(".json"),dict(promoted=datetime.now(timezone.utc).date().isoformat(),
                    by="Single-fighter full-game superiority and boss retention",source=str(player.relative_to(RUNS)),
                    sha256=selection["player_sha256"],parent_sha256=CFG["champion_sha256"],environment="env-v0.1.5.2",
                    baseline_games={a:str(OUT/"confirmation"/a) for a in ("champion","student")},
                    evidence=outcome["full_games"]["worst_case_ci95_pp"]))
    for name, counts in (("champion",full["first"]),("student",full["second"])):
        label = "Early-refit champion" if name=="champion" else "Single fighter"
        atomic_json(RUNS/"fresh_eval"/f"{HERE.name}-{name}.json",
                    dict(label=f"{label} · single-fighter comparison",assigned=n,
                         summary=dict(n=counts["completed"],wins=counts["wins"],mode="greedy",mean_floor=counts["mean_floor"])))
    nominal = full.get("difference_pp",full.get("complete_pair_difference_pp"))
    bounds = full.get("ci95_pp",full.get("complete_pair_ci95_pp"))
    note("finished", f"Single fighter {full['second']['wins']:,} wins, champion {full['first']['wins']:,} on {n:,} full games "
         f"({nominal:+.2f} points, 95% interval {bounds[0]:+.2f} to {bounds[1]:+.2f}). "
         f"On {CFG['boss_starts']:,} original boss entries they won {bosses['original']['student_wins']} and "
         f"{bosses['original']['champion_wins']}. " + ("The one-fighter player is champion." if promoted else "The early-refit champion retains the title."))
    return outcome
