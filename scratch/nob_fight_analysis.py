"""Analyze gemma-4-E4B Gremlin Nob fights from the base rollouts:
- classify each fight (convincing win / costly win / loss)
- plot average action patterns, HP trajectories, and per-turn action mix.

Data: data/iter2_rwr_hinted/eval/base (100 full E4B rollouts, LLM combat).
Outputs PNGs to scratch/plots/.
"""
from __future__ import annotations
import json, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192"
OUT = "scratch/plots"; os.makedirs(OUT, exist_ok=True)
KINDS = ["attack", "block", "skill", "other"]
COLORS = {"attack": "#d1495b", "block": "#3a7ca5", "skill": "#e3a72f", "other": "#8d99ae"}


def classify(desc: str) -> str:
    if not desc.startswith("play"):
        return "other"          # end turn / drink potion
    if "(deal" in desc:
        return "attack"
    low = desc.lower()
    if any(b in low for b in ("defend", "shrug it off", "iron wave", "true grit",
                              "ghostly armor", "impervious", "power through", "sentinel")):
        return "block"
    return "skill"


def extract_fights():
    fights = []
    for jp in sorted(glob.glob(f"{BASE}/seed_*_r0.jsonl")):
        seed = int(jp.split("seed_")[1].split("_r")[0])
        recs = [json.loads(l) for l in open(jp)]
        idxs = [i for i, r in enumerate(recs) if r["phase"] == "combat"
                and any(e["name"] == "GREMLIN_NOB" and e["alive"] for e in r["state"]["combat"]["enemies"])]
        if not idxs:
            continue
        decs = [recs[i] for i in idxs]
        entry_hp = decs[0]["state"]["combat"]["player_cur_hp"]
        max_hp = decs[0]["state"]["combat"]["player_max_hp"]
        # per-decision: (turn, kind, player_hp_before, nob_hp_before, nob_str)
        steps = []
        for r in decs:
            c = r["state"]["combat"]
            nob = next((e for e in c["enemies"] if e["name"] == "GREMLIN_NOB"), None)
            steps.append(dict(turn=c["turn"], kind=classify(r["selected_action"].get("description", "")),
                              php=c["player_cur_hp"], nhp=nob["cur_hp"] if nob else 0,
                              nstr=nob["strength"] if nob else 0))
        last = idxs[-1]
        after = recs[last]["after_state"]
        hp_after = after.get("combat", {}).get("player_cur_hp", after.get("cur_hp", 0))
        tail = recs[last + 1:last + 3]
        survived = (any(r["phase"] == "out_of_combat" for r in tail) or after.get("phase") == "out_of_combat") and hp_after > 0
        hp_loss = entry_hp - hp_after
        fights.append(dict(seed=seed, entry_hp=entry_hp, max_hp=max_hp, exit_hp=hp_after,
                           hp_loss=hp_loss, n_turns=max(s["turn"] for s in steps),
                           n_dec=len(steps), survived=survived, steps=steps))
    return fights


def bucket(f):
    if not f["survived"]:
        return "loss"
    return "convincing" if f["hp_loss"] <= 20 else "costly"


def main():
    fights = extract_fights()
    for f in fights:
        f["bucket"] = bucket(f)
    n = len(fights)
    conv = [f for f in fights if f["bucket"] == "convincing"]
    costly = [f for f in fights if f["bucket"] == "costly"]
    loss = [f for f in fights if f["bucket"] == "loss"]
    print(f"Nob fights: {n}")
    print(f"  convincing wins (survived, <=20 HP lost): {len(conv)}  -> seeds {sorted(f['seed'] for f in conv)}")
    print(f"  costly wins    (survived, >20 HP lost):   {len(costly)}")
    print(f"  losses         (died in fight):           {len(loss)}")
    for f in sorted(conv, key=lambda x: x["hp_loss"])[:6]:
        print(f"    seed {f['seed']}: lost {f['hp_loss']} HP ({f['entry_hp']}->{f['exit_hp']}/{f['max_hp']}), {f['n_turns']} turns, {f['n_dec']} decisions")

    # ---- Plot 1: HP-loss distribution by outcome ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-5, 85, 5)
    ax.hist([[f["hp_loss"] for f in conv], [f["hp_loss"] for f in costly], [f["hp_loss"] for f in loss]],
            bins=bins, stacked=True, color=["#2a9d8f", "#e9c46a", "#d1495b"],
            label=[f"convincing win ({len(conv)})", f"costly win ({len(costly)})", f"loss ({len(loss)})"])
    ax.axvline(20, ls="--", c="gray", lw=1); ax.text(21, ax.get_ylim()[1]*0.9, "convincing\nthreshold", fontsize=8, color="gray")
    ax.set_xlabel("HP lost during the Gremlin Nob fight"); ax.set_ylabel("number of fights")
    ax.set_title("gemma-4-E4B vs Gremlin Nob: HP lost per fight (n=%d)" % n)
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/nob_hploss_dist.png", dpi=130); plt.close(fig)

    # ---- Plot 2: per-turn action mix (fraction), won vs lost ----
    def turn_mix(fs, max_turn=10):
        counts = {k: np.zeros(max_turn + 1) for k in KINDS}
        tot = np.zeros(max_turn + 1)
        for f in fs:
            for s in f["steps"]:
                t = s["turn"]
                if t <= max_turn:
                    counts[s["kind"]][t] += 1; tot[t] += 1
        tot = np.where(tot == 0, 1, tot)
        return {k: counts[k] / tot for k in KINDS}, tot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, fs, title in ((axes[0], conv + costly, f"WON fights (n={len(conv)+len(costly)})"),
                          (axes[1], loss, f"LOST fights (n={len(loss)})")):
        mix, _ = turn_mix(fs)
        turns = np.arange(len(next(iter(mix.values()))))
        bottom = np.zeros_like(turns, dtype=float)
        for k in KINDS:
            ax.bar(turns, mix[k], bottom=bottom, color=COLORS[k], label=k, width=0.85)
            bottom += mix[k]
        ax.set_xlabel("combat turn"); ax.set_title(title); ax.set_xlim(-0.5, 8.5)
    axes[0].set_ylabel("fraction of card plays / actions")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Action mix per turn in Gremlin Nob fights (attack=races it; block/skill feed Enrage)")
    fig.tight_layout(); fig.savefig(f"{OUT}/nob_action_mix_by_turn.png", dpi=130); plt.close(fig)

    # ---- Plot 3: HP trajectory (player HP %) vs turn, per fight + mean ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for f in fights:
        by_turn = {}
        for s in f["steps"]:
            by_turn.setdefault(s["turn"], s["php"])  # first (highest) hp seen that turn
        ts = sorted(by_turn); ys = [100 * by_turn[t] / f["max_hp"] for t in ts]
        c = {"convincing": "#2a9d8f", "costly": "#e9c46a", "loss": "#d1495b"}[f["bucket"]]
        ax.plot(ts, ys, color=c, alpha=0.35, lw=1)
    # mean per bucket
    for b, c in (("convincing", "#2a9d8f"), ("costly", "#e9c46a"), ("loss", "#d1495b")):
        acc = defaultdict(list)
        for f in fights:
            if f["bucket"] != b: continue
            by_turn = {}
            for s in f["steps"]: by_turn.setdefault(s["turn"], s["php"])
            for t, hp in by_turn.items(): acc[t].append(100 * hp / f["max_hp"])
        if acc:
            ts = sorted(acc); ax.plot(ts, [np.mean(acc[t]) for t in ts], color=c, lw=3,
                                      label=f"{b} (mean)")
    ax.set_xlabel("combat turn"); ax.set_ylabel("player HP (% of max)")
    ax.set_title("Player HP trajectory through Gremlin Nob fights"); ax.set_xlim(0, 9); ax.set_ylim(0, 105)
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/nob_hp_trajectory.png", dpi=130); plt.close(fig)

    # ---- Plot 4: overall action share, won vs lost (the headline) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = [("WON", conv + costly), ("LOST", loss)]
    x = np.arange(len(groups)); w = 0.2
    for i, k in enumerate(KINDS):
        vals = []
        for _, fs in groups:
            tot = sum(len(f["steps"]) for f in fs) or 1
            vals.append(sum(1 for f in fs for s in f["steps"] if s["kind"] == k) / tot)
        ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[k], label=k)
    ax.set_xticks(x); ax.set_xticklabels([f"{g} ({len(fs)})" for g, fs in groups])
    ax.set_ylabel("share of all actions"); ax.set_title("Action share in WON vs LOST Nob fights")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/nob_action_share_wonlost.png", dpi=130); plt.close(fig)

    print("\nwrote plots to", OUT)
    for p in ("nob_hploss_dist", "nob_action_mix_by_turn", "nob_hp_trajectory", "nob_action_share_wonlost"):
        print(f"  {OUT}/{p}.png")


if __name__ == "__main__":
    main()
