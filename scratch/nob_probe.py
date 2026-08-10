"""Matched-state propensity probe: does the Tier-1 serializer fix change the
model's choice at the critical Gremlin Nob decision?

For each base run with a Nob fight, replay the recorded action prefix to a
mid-fight decision where Enrage is active and the player has a real skill-vs-other
choice, then sample the model K times at that FIXED state under two serializers:
  NEW = Enrage + Metallicize visible + damage-final clause
  OLD = reconstructed pre-fix view (enemy powers stripped; no damage clause)
Same model, same state, same seeds -> isolates the serializer's effect on the
decision that drives the death spiral (playing Skills feeds Enrage's Strength).

Metrics per state x arm (over K samples): P(play a Skill), P(play a block/defend),
P(attack/other), and P(double-counts Strength in the thinking).

Run: PYTHONPATH=src .venv/bin/python scratch/nob_probe.py --seeds 101,103,104,106,107,108 --k 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from sts_ai import glossary
from sts_ai.interactive.replay import replay_actions
from sts_ai.lightspeed import LightspeedHybridEnv

BASE_DIR = Path("data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192")
_NEW_POWER_NAMES = [
    "Artifact", "Metallicize", "Plated Armor", "Regen", "Thorns", "Intangible",
    "Enrage", "Curl Up", "Malleable", "Mode Shift", "Angry", "Flight",
    "Sharp Hide", "Asleep", "Spore Cloud", "Time Warp", "Painful Stabs", "Barricade",
]
_STRIP_RE = re.compile(r",\s*(?:" + "|".join(re.escape(n) for n in _NEW_POWER_NAMES) + r")\s+-?\d+")
_DOUBLECOUNT_RE = re.compile(r"(?:total\s+damage|damage\s*(?:taken|is|=|of))[^.\n]*?\b\d+\s*\+\s*\d+", re.I)


def strip_new_powers(state_text: str) -> str:
    return "\n".join(
        _STRIP_RE.sub("", line) if ("HP " in line and "intent" in line) else line
        for line in state_text.split("\n")
    )


def build_state_text(env, legal_dicts, arm):
    raw = env.describe_state()
    if arm == "old":
        return glossary.augment(strip_new_powers(raw), legal_dicts, "combat", damage_note=False)
    return glossary.augment(raw, legal_dicts, "combat", damage_note=True)


def classify(desc: str) -> str:
    if not desc.startswith("play"):
        return "other"          # end turn / potion
    if "(deal" in desc:
        return "attack"
    # skill/power played; separate block cards (Defend/Shrug/etc.) as "block"
    low = desc.lower()
    if any(b in low for b in ("defend", "shrug it off", "iron wave", "true grit",
                              "ghostly armor", "impervious", "power through", "sentinel")):
        return "block"
    return "skill"


def find_probe_state(env, recs, nob_k, min_str=6):
    """Replay into the Nob fight to a decision where Enrage is active, the Nob's
    Strength has built up (>= min_str, so the damage-final clause and Enrage cost
    both bite), and a real card-play choice exists. Falls back to the deepest
    Enrage+choice state seen if the strength target isn't reached. Returns True if
    positioned at a usable decision (env is left at the chosen decision)."""
    from sts_ai.interactive.replay import resolve_action_index
    actions = [r["selected_action"] for r in recs]
    best_depth = None  # (# recorded actions consumed) of the best fallback state
    for j in range(nob_k, min(nob_k + 20, len(actions))):
        if env.phase() != "combat" or env.is_terminal():
            break
        st = env.describe_state()
        nob = [l for l in st.splitlines() if "GREMLIN_NOB" in l and "HP" in l]
        if not nob:
            break
        has_enrage = "Enrage" in nob[0]
        m = re.search(r"Strength\s+(-?\d+)", nob[0])
        cur_str = int(m.group(1)) if m else 0
        legal = env.legal_actions()
        kinds = {classify(a.description) for a in legal}
        usable = has_enrage and len(legal) >= 3 and ("skill" in kinds or "attack" in kinds)
        if usable and cur_str >= min_str:
            return True                      # ideal: high-strength Enrage decision
        if usable and best_depth is None:
            best_depth = j                   # remember first usable as fallback
        try:
            idx = resolve_action_index(env, actions[j].get("bits"), str(actions[j].get("description", "")))
            env.step(idx)
        except Exception:
            break
    # Strength target not reached; re-replay to the fallback usable state if we found one.
    if best_depth is not None:
        env2 = LightspeedHybridEnv(world_seed=env.world_seed, combat_control="llm",
                                   battle_simulations=50, max_act=3)
        replay_actions(env2, [r["selected_action"] for r in recs[:best_depth]])
        # copy the repositioned env back onto the caller's handle
        env.__dict__.update(env2.__dict__)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,103,104,106,107,108")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--backend", choices=["mlx", "vllm"], default="mlx")
    # High default so verbose thinking finishes and emits JSON (else the decision is
    # lost to truncation -> invalid). The 12B is verbose; 8192 gives comfortable
    # headroom (E4B thought ~1k tokens; a 12B decision hit the 4096 cap locally).
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--out", default="scratch/nob_probe_results.jsonl")
    args = ap.parse_args()

    if args.backend == "vllm":
        from sts_ai.agents import VllmJsonAgent
        agent = VllmJsonAgent(model_id=args.model, max_tokens=args.max_tokens,
                              temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              enable_thinking=True, max_retries=args.max_retries,
                              enable_prefix_caching=True)
    else:
        from sts_ai.agents import MlxQwenJsonAgent
        agent = MlxQwenJsonAgent(model_id=args.model, max_tokens=args.max_tokens,
                                 temperature=args.temperature, enable_thinking=True,
                                 max_retries=args.max_retries)
    print(f"[ready] {args.model} backend={args.backend} max_tokens={args.max_tokens}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    out = open(args.out, "w")
    agg = {"new": Counter(), "old": Counter()}
    agg_dc = {"new": 0, "old": 0}
    agg_n = {"new": 0, "old": 0}
    for seed in seeds:
        jp = BASE_DIR / f"seed_{seed}_r0.jsonl"
        if not jp.exists():
            print(f"[skip] {seed}: no base run", flush=True); continue
        recs = [json.loads(l) for l in jp.open()]
        nob_k = next((i for i, r in enumerate(recs) if r["phase"] == "combat"
                      and {e["name"] for e in r["state"]["combat"]["enemies"] if e["alive"]} == {"GREMLIN_NOB"}), None)
        if nob_k is None:
            print(f"[skip] {seed}: no Nob fight", flush=True); continue
        # Position a fresh env at the probe state, then snapshot the recorded prefix len.
        env0 = LightspeedHybridEnv(world_seed=seed, combat_control="llm", battle_simulations=50, max_act=3)
        try:
            replay_actions(env0, [r["selected_action"] for r in recs[:nob_k]])
        except Exception as exc:
            print(f"[skip] {seed}: replay failed {str(exc)[:50]}", flush=True); continue
        if not find_probe_state(env0, recs, nob_k):
            print(f"[skip] {seed}: no enrage skill-choice state found", flush=True); continue
        # Capture the state for both arms (same env position).
        legal = env0.legal_actions()
        legal_dicts = [env0.action_dict(a) for a in legal]
        combat = env0.summary()["combat"]
        enemy = [e for e in combat["enemies"] if e["alive"]][0]
        for arm in ("new", "old"):
            state_text = build_state_text(env0, legal_dicts, arm)
            kinds = Counter(); dc = 0; chosen = []
            t0 = time.time()
            for sample_i in range(args.k):
                # Vary the sampling seed per sample so K draws differ AND are
                # reproducible. Required for vLLM (a fixed seed -> identical outputs);
                # harmless for MLX (reseeds mx.random per draw). Distinct per (seed,
                # arm, sample) so the two arms aren't drawing the same seed sequence.
                agent.reseed(1_000_000 * seed + (0 if arm == "new" else 500_000) + sample_i)
                d = agent.choose_action(state_text, legal)
                idx = d.action_index if d.valid and 0 <= d.action_index < len(legal) else None
                kind = classify(legal[idx].description) if idx is not None else "invalid"
                kinds[kind] += 1
                chosen.append(legal[idx].description[:40] if idx is not None else "INVALID")
                if _DOUBLECOUNT_RE.search(d.thinking or ""):
                    dc += 1
            rec = {"seed": seed, "arm": arm, "enemy_hp": enemy["cur_hp"],
                   "enemy_str": enemy["strength"], "enrage": enemy.get("powers", {}).get("Enrage"),
                   "kinds": dict(kinds), "doublecount": dc, "k": args.k,
                   "chosen": chosen, "wall_s": round(time.time() - t0, 1)}
            out.write(json.dumps(rec) + "\n"); out.flush()
            agg[arm].update(kinds); agg_dc[arm] += dc; agg_n[arm] += args.k
            print(f"[{seed}/{arm}] str={enemy['strength']} enrage={rec['enrage']} "
                  f"kinds={dict(kinds)} dc={dc}/{args.k} ({rec['wall_s']}s)", flush=True)
    out.close()
    print("\n=== AGGREGATE (skill = feeds Enrage; lower is better) ===", flush=True)
    for arm in ("old", "new"):
        n = agg_n[arm] or 1
        k = agg[arm]
        print(f"{arm}: n={agg_n[arm]}  skill={k['skill']}({k['skill']/n:.0%}) "
              f"block={k['block']}({k['block']/n:.0%}) attack={k['attack']}({k['attack']/n:.0%}) "
              f"other={k['other']} invalid={k['invalid']}  doublecount={agg_dc[arm]}({agg_dc[arm]/n:.0%})", flush=True)


if __name__ == "__main__":
    main()
