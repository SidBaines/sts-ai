"""Partial-rollout A/B: does the Tier-1 serializer fix improve combat play?

For each recorded base run, replay to a target elite fight (default GREMLIN_NOB),
then play ONLY that fight with the local gemma-4-E4B agent under two serializers:
  NEW  = current glossary (damage-final clause + enemy powers like Enrage visible)
  OLD  = reconstructed pre-fix view (strip added enemy powers; no damage clause)
Same model, same position, same sampling seed -> isolates the serializer effect.

Metrics per (seed, arm): survived?, entry/exit HP, HP loss, turns, skill-vs-attack
play mix (skills feed Enrage), and a rough "double-counted Strength in thinking" flag.

Run: PYTHONPATH=src .venv/bin/python scratch/nob_ab_experiment.py --enemy GREMLIN_NOB --seeds 104,103,101,102 --arms new,old
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from sts_ai import glossary
from sts_ai.interactive.replay import replay_actions
from sts_ai.lightspeed import LightspeedHybridEnv

BASE_DIR = Path("data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192")

# Powers the binding now emits (kPlayerRelevantEnemyPowers) minus the original four
# (Strength/Vulnerable/Weak/Poison, which the OLD serializer already showed).
_NEW_POWER_NAMES = [
    "Artifact", "Metallicize", "Plated Armor", "Regen", "Thorns", "Intangible",
    "Enrage", "Curl Up", "Malleable", "Mode Shift", "Angry", "Flight",
    "Sharp Hide", "Asleep", "Spore Cloud", "Time Warp", "Painful Stabs", "Barricade",
]
_STRIP_RE = re.compile(r",\s*(?:" + "|".join(re.escape(n) for n in _NEW_POWER_NAMES) + r")\s+-?\d+")
# Double-count proxy: thinking adds a second number to an already-final "deal N".
_DOUBLECOUNT_RE = re.compile(r"(?:total\s+damage|damage\s*(?:taken|is|=))[^.\n]*?\b\d+\s*\+\s*\d+", re.I)


def strip_new_powers(state_text: str) -> str:
    """Reconstruct the pre-fix enemy line by removing the added persistent powers."""
    return "\n".join(
        _STRIP_RE.sub("", line) if ("HP " in line and "intent" in line) else line
        for line in state_text.split("\n")
    )


def build_state_text(env: LightspeedHybridEnv, legal_action_dicts, arm: str) -> str:
    raw = env.describe_state()
    if arm == "old":
        return glossary.augment(strip_new_powers(raw), legal_action_dicts, "combat", damage_note=False)
    return glossary.augment(raw, legal_action_dicts, "combat", damage_note=True)


def first_fight_index(recs, enemy: str) -> int | None:
    for i, r in enumerate(recs):
        if r["phase"] == "combat":
            names = {e["name"] for e in r["state"]["combat"]["enemies"] if e["alive"]}
            if enemy in names:
                return i
    return None


def play_fight(env, agent, arm: str, max_decisions: int = 60) -> dict:
    combat0 = env.summary()["combat"]
    entry_hp = combat0["player_cur_hp"]
    n_skill = n_attack = n_doublecount = n_dec = 0
    turns_seen = set()
    for _ in range(max_decisions):
        if env.phase() != "combat" or env.is_terminal():
            break
        legal = env.legal_actions()
        legal_dicts = [env.action_dict(a) for a in legal]
        state_text = build_state_text(env, legal_dicts, arm)
        decision = agent.choose_action(state_text, legal)
        n_dec += 1
        c = env.summary()["combat"]
        turns_seen.add(c["turn"])
        idx = decision.action_index if decision.valid and 0 <= decision.action_index < len(legal) else 0
        desc = legal[idx].description
        if desc.startswith("play"):
            if "(deal" in desc:
                n_attack += 1
            else:
                n_skill += 1
        if _DOUBLECOUNT_RE.search(decision.thinking or ""):
            n_doublecount += 1
        try:
            env.step(idx)
        except Exception as exc:
            return {"arm": arm, "error": str(exc)[:80], "entry_hp": entry_hp,
                    "n_skill": n_skill, "n_attack": n_attack}
    term = env.summary()
    in_combat = term.get("phase") == "combat"
    combat = term.get("combat", {})
    exit_hp = combat.get("player_cur_hp", term.get("cur_hp", 0)) if in_combat else term.get("cur_hp", 0)
    # survived the fight = we left combat with the run not lost
    survived = (not in_combat) and ("PLAYER_LOSS" not in term.get("outcome", ""))
    return {
        "arm": arm,
        "survived_fight": survived,
        "run_lost": "PLAYER_LOSS" in term.get("outcome", ""),
        "entry_hp": entry_hp,
        "exit_hp": exit_hp,
        "hp_loss": entry_hp - exit_hp,
        "turns": max(turns_seen) if turns_seen else 0,
        "n_decisions": n_dec,
        "n_skill": n_skill,
        "n_attack": n_attack,
        "skill_frac": round(n_skill / max(n_skill + n_attack, 1), 2),
        "n_doublecount": n_doublecount,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enemy", default="GREMLIN_NOB")
    ap.add_argument("--seeds", default="104,103,101,102")
    ap.add_argument("--arms", default="new,old")
    ap.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="scratch/nob_ab_results.jsonl")
    args = ap.parse_args()

    from sts_ai.agents import MlxQwenJsonAgent
    t0 = time.time()
    agent = MlxQwenJsonAgent(model_id=args.model, max_tokens=args.max_tokens,
                             temperature=args.temperature, enable_thinking=True, max_retries=1)
    print(f"[load] agent ready in {time.time()-t0:.0f}s", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    out = open(args.out, "w")
    for seed in seeds:
        jp = BASE_DIR / f"seed_{seed}_r0.jsonl"
        if not jp.exists():
            print(f"[skip] {seed}: no base run", flush=True); continue
        recs = [json.loads(l) for l in jp.open()]
        k = first_fight_index(recs, args.enemy)
        if k is None:
            print(f"[skip] {seed}: no {args.enemy} fight", flush=True); continue
        actions = [r["selected_action"] for r in recs[:k]]
        for arm in arms:
            t1 = time.time()
            env = LightspeedHybridEnv(world_seed=seed, combat_control="llm",
                                      battle_simulations=50, max_act=3)
            try:
                replay_actions(env, actions)
            except Exception as exc:
                print(f"[skip] {seed}/{arm}: replay failed: {str(exc)[:60]}", flush=True); continue
            res = play_fight(env, agent, arm)
            res["seed"] = seed; res["enemy"] = args.enemy; res["wall_s"] = round(time.time()-t1, 1)
            out.write(json.dumps(res) + "\n"); out.flush()
            print(f"[done] seed={seed} arm={arm} survived={res.get('survived_fight')} "
                  f"hp_loss={res.get('hp_loss')} skill_frac={res.get('skill_frac')} "
                  f"n_dec={res.get('n_decisions')} dc={res.get('n_doublecount')} "
                  f"({res['wall_s']}s)", flush=True)
    out.close()


if __name__ == "__main__":
    main()
