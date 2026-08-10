"""Transcript analysis of gemma-4-E4B (thinking) rollouts: funnel, death context,
combat blunders, potion/campfire behaviour, thinking stats."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from sts_ai import risk_proxies
from sts_ai.affordances import action_contributes_block, action_is_single_target_lethal

ROOT = Path("data/iter2_rwr_hinted")

POTION_LINE = re.compile(r"^Potions:\s*(.*)$", re.M)


def potions_held(state_text: str) -> int:
    m = POTION_LINE.search(state_text)
    if not m:
        return 0
    val = m.group(1).strip()
    if not val or val == "none":
        return 0
    return val.count(",") + 1


def load_rollouts(d: Path):
    for jp in sorted(d.rglob("seed_*_r0.jsonl")):
        mp = jp.with_suffix(".meta.json")
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        recs = [json.loads(l) for l in jp.open() if l.strip()]
        yield jp.stem, meta, recs


def analyze(d: Path, label: str, with_hints: bool = False):
    floors, acts = [], []
    outcome_c = Counter()
    death_ctx = []          # (floor, act, encounter, entry_hp_frac, potions_at_death, n_turns)
    combats = []            # per-combat dicts
    block_blunders = Counter()  # soft threshold: incoming >= 40% hp, full block possible, no block played
    lethal_forgone = 0
    lethal_avail = 0
    endturn_leftblock = 0
    endturn_incoming = 0
    drink_actions = 0
    drink_opportunity_decisions = 0
    think_toks_combat, think_toks_ooc = [], []
    think_trunc = 0
    n_dec_total = 0
    hint_stats = Counter()
    campfire_events = []
    all_events = []
    per_run = []

    for stem, meta, recs in load_rollouts(d):
        floors.append(meta["final_floor"])
        acts.append(meta["final_act"])
        outcome_c[(meta["outcome"].split(".")[-1], meta["stopped_reason"])] += 1
        died = "PLAYER_LOSS" in meta["outcome"]

        # risk proxies (OOC)
        all_events.extend(risk_proxies.risk_events(recs))

        # per-combat grouping
        cur = None
        run_combats = []
        for r in recs:
            n_dec_total += 1
            ag = r["agent"]
            tt = ag.get("thinking_tokens", 0)
            if ag.get("metadata", {}).get("thinking_truncated"):
                think_trunc += 1
            if with_hints and "hint" in ag.get("metadata", {}):
                h = ag["metadata"]["hint"]
                hint_stats[h.get("launder_outcome")] += 1
            if r["phase"] == "combat":
                think_toks_combat.append(tt)
                c = r["state"]["combat"]
                if cur is None or c["turn"] < cur["last_turn"] and c["turn"] == 1 and cur["last_turn"] > 1:
                    # new combat starts when we see turn reset (or first combat rec)
                    pass
                if cur is None:
                    cur = dict(entry_hp=c["player_cur_hp"], max_hp=c["player_max_hp"],
                               enemies=tuple(sorted(e["name"] for e in c["enemies"] if e["alive"])),
                               floor=r["state"]["floor"], act=r["state"]["act"],
                               last_turn=c["turn"], n_dec=0, potions_entry=potions_held(r["state_text"]))
                cur["last_turn"] = c["turn"]
                cur["n_dec"] += 1
                cur["exit_hp"] = r["after_state"].get("combat", {}).get(
                    "player_cur_hp", r["after_state"].get("cur_hp", c["player_cur_hp"]))
                cur["last_state_text"] = r["state_text"]
                cur["last_rec"] = r

                aff = r.get("affordances") or {}
                sel = r["selected_action"]
                desc = sel.get("description", "")
                incoming = aff.get("incoming_damage_total", 0)
                hp = c["player_cur_hp"]
                # lethal forgone
                if aff.get("single_target_lethal_available"):
                    lethal_avail += 1
                    if not action_is_single_target_lethal(sel, c):
                        lethal_forgone += 1
                # block blunder (soft): meaningful incoming, full block possible, chose non-block
                if aff.get("full_block_possible") and incoming >= max(1, round(0.4 * hp)):
                    if not action_contributes_block(sel, r["state_text"]):
                        block_blunders["blunder"] += 1
                    else:
                        block_blunders["ok"] += 1
                # end turn leaving block on table while taking damage
                if desc.strip() == "end turn" and incoming > aff.get("player_block", 0):
                    endturn_incoming += 1
                    if aff.get("max_block_available", 0) > 0 and aff.get("player_energy", 0) > 0:
                        endturn_leftblock += 1
                if desc.startswith("drink"):
                    drink_actions += 1
                if any(a["description"].startswith("drink") for a in r["legal_actions"]):
                    drink_opportunity_decisions += 1
            else:
                think_toks_ooc.append(tt)
                if cur is not None:
                    run_combats.append(cur)
                    cur = None
        if cur is not None:
            run_combats.append(cur)

        for cb in run_combats:
            cb["hp_loss"] = cb["entry_hp"] - cb["exit_hp"]
            combats.append(cb)

        if died and run_combats:
            k = run_combats[-1]
            death_ctx.append(dict(
                floor=k["floor"], act=k["act"], enc=" + ".join(k["enemies"]),
                entry_frac=round(k["entry_hp"] / max(k["max_hp"], 1), 2),
                potions=k.get("potions_entry", 0), turns=k["last_turn"], stem=stem))
        per_run.append(dict(stem=stem, floor=meta["final_floor"], died=died,
                            n_combats=len(run_combats)))

    print(f"\n================ {label} (n={len(floors)}) ================")
    print(f"final_floor: mean {statistics.mean(floors):.1f} median {statistics.median(floors)} "
          f"min {min(floors)} max {max(floors)}")
    act_reached = Counter(acts)
    n = len(floors)
    ge2 = sum(1 for a in acts if a >= 2); ge3 = sum(1 for a in acts if a >= 3)
    print(f"act funnel: reached act2 {ge2}/{n} ({ge2/n:.0%}), act3 {ge3}/{n} ({ge3/n:.0%})")
    print("outcome x stopped:", dict(outcome_c))

    print("\n-- deaths --")
    dc = Counter((d_["act"], d_["enc"]) for d_ in death_ctx)
    for (act, enc), cnt in dc.most_common(12):
        print(f"  act{act}  {cnt:3d}  {enc}")
    if death_ctx:
        fracs = [d_["entry_frac"] for d_ in death_ctx]
        pots = [d_["potions"] for d_ in death_ctx]
        print(f"  entry HP frac at fatal combat: mean {statistics.mean(fracs):.2f} "
              f"(<50%: {sum(1 for f in fracs if f < 0.5)}/{len(fracs)})")
        print(f"  potions held entering fatal combat: mean {statistics.mean(pots):.2f}, "
              f">=1: {sum(1 for p in pots if p >= 1)}/{len(pots)}")
        print(f"  death floors: {Counter(d_['floor'] for d_ in death_ctx).most_common(8)}")

    print("\n-- combat economics --")
    by_enc = defaultdict(list)
    for cb in combats:
        by_enc[(cb["act"], " + ".join(cb["enemies"]))].append(cb["hp_loss"])
    worst = sorted(by_enc.items(), key=lambda kv: -statistics.mean(kv[1]))
    print("  mean HP loss by encounter (n>=5 shown, top 12):")
    for (act, enc), losses in worst:
        if len(losses) >= 5:
            print(f"    act{act} {enc[:60]:60s} n={len(losses):3d} mean_loss={statistics.mean(losses):5.1f}")
    print(f"  combats total: {len(combats)}; mean HP loss/combat: "
          f"{statistics.mean([c['hp_loss'] for c in combats]):.1f}")

    print("\n-- blunders --")
    bt = block_blunders["blunder"] + block_blunders["ok"]
    print(f"  full-block-available & incoming>=40%HP: chose no-block "
          f"{block_blunders['blunder']}/{bt} ({block_blunders['blunder']/bt:.0%})" if bt else "  (none)")
    print(f"  lethal available: {lethal_avail}; forgone: {lethal_forgone} "
          f"({lethal_forgone/max(lethal_avail,1):.0%})")
    print(f"  'end turn' while incoming>block: {endturn_incoming}; of those, had energy+block in hand: "
          f"{endturn_leftblock} ({endturn_leftblock/max(endturn_incoming,1):.0%})")
    print(f"  potion drinks: {drink_actions} over {drink_opportunity_decisions} decisions with a drink available")

    print("\n-- OOC risk proxies --")
    s = risk_proxies.summarize_risk(all_events)
    print(f"  campfire rest rate by HP: { {k: v for k, v in s['campfire_rest_rate_by_hp'].items()} }")
    print(f"  smith count: {s['campfire_smith_count']}, elite rate: {s['map_elite_rate']}, "
          f"low-HP elite: {s['map_elite_rate_low_hp']}")
    print(f"  card take rate: {s['card_take_rate']}, shop: {s['shop']}, potions acquired: {s['potion_acquire_count']}")

    print("\n-- thinking --")
    for nm, arr in (("combat", think_toks_combat), ("ooc", think_toks_ooc)):
        if arr:
            arr_s = sorted(arr)
            print(f"  {nm}: n={len(arr)} mean {statistics.mean(arr):.0f} p50 {arr_s[len(arr)//2]} "
                  f"p90 {arr_s[int(len(arr)*0.9)]} max {max(arr)}")
    print(f"  thinking_truncated decisions: {think_trunc}/{n_dec_total}")
    if with_hints:
        print(f"\n-- hints (launder outcomes) -- {dict(hint_stats)}")

    return death_ctx, combats, per_run


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        analyze(ROOT / "eval" / "base", "EVAL BASE (gemma-4-E4B thinking, no hints)")
    elif which == "trained":
        analyze(ROOT / "eval" / "trained", "EVAL TRAINED (RWR+hinted adapter)")
    else:
        analyze(ROOT / "train_rollouts", "TRAIN ROLLOUTS (hints ON, temp 1.0)", with_hints=True)
