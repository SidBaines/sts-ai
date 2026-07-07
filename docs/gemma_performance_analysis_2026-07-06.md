# Gemma-4-E4B (thinking) — where performance is lost, and what to do about it

**Date:** 2026-07-06. **Goal context:** framing experiments are paused; the current objective is
simply to get gemma-4-E4B (native thinking) to play the game well.

**Data analyzed** (all existing, no new GPU): the 100 base-arm eval rollouts
(`data/iter2_rwr_hinted/eval/base`, frozen eval seeds, temp 1.0, full LLM combat, no hints), the
100 trained-arm rollouts (`eval/trained`), and the 300 train-split generation rollouts
(`train_rollouts/`, hints on). ~64k decisions total. Analysis script:
`scratch/analyze_gemma_transcripts.py` (run with `base` / `trained` / `train`).

---

## 1. Where the model actually loses

### 1.1 The funnel has one wall: act-1 elites and the act-1 boss

Base arm (n=100): **100/100 runs die. 23% reach Act 2, 0% reach Act 3.** Median death floor 16
(the act-1 boss). Death encounters:

| encounter | deaths | note |
|---|---|---|
| GREMLIN_NOB (elite) | 20 | |
| act-1 boss (Guardian 11 / Slime Boss 10 / Hexaghost 7) | 28 | 28 of the 51 runs that reached floor 16 |
| LAGAVULIN (elite) | 13 | |
| 3× SENTRY (elite) | 7 | |
| everything else | 32 | mostly act-2 attrition among the 23 that got there |

So **68% of all deaths are act-1 elites + boss**. This is not attrition: mean HP fraction
*entering* the fatal fight is 0.60 (only 37/100 entered below half HP).

### 1.2 Combat economics: trash is solved, big fights are catastrophic

Mean HP loss per combat by encounter (base arm): Jaw Worm 2.6, Cultist −0.4, lice/slimes ≈0–2 —
**ordinary fights are essentially free**. Meanwhile: Gremlin Nob **−48.7** (n=43), Lagavulin
−36.2, Sentries −32.2, act-1 bosses **−49 to −55**. The model does not have a general combat
problem; it has an **asymmetric-fight problem** — encounters that punish a default play pattern
(Nob/Enrage), demand setup (Lagavulin), or need AOE/tempo (Sentries, Byrds).

### 1.3 Three concrete mechanisms (from reading fatal transcripts)

Read a Nob death end-to-end (`eval/base/...seed_104_r0.jsonl`, floor 6) and the pattern is vivid:

**(a) The model plays Skills into Enrage — the exact losing strategy vs Nob.** Across all Nob
fights, **61% of its card plays are Skills** (373 skill vs 239 attack plays). Each Skill feeds
Nob +2–3 Strength; in the sampled fight Nob goes from 14 damage/turn to 51 by turn 5. The killer
detail: the glossary DOES explain this — `GREMLIN_NOB_BELLOW: "buffs itself with Enrage (it gains
Strength whenever you play a Skill)"` — **but only while the current intent is BELLOW, i.e. turn 0
only.** From turn 1 the intent changes, the explanation leaves the KEY, and the model just watches
Strength climb with no visible cause. Persistent enemy powers (Enrage, Sentry Artifact, enemy
Metallicize/Ritual…) are not serialized — the structured enemy record carries only
strength/vulnerable/weak/poison.

**(b) The model double-counts Strength and systematically overestimates incoming damage.** The
intent annotation `(deal 24)` is already modifier-inclusive (`calculateDamageToPlayer`), and the
derived note says "Incoming attack damage this turn: 24 (before your Block)" — but nothing says
Strength is already in the number. The model's thinking, verbatim: *"Base damage: 24. Strength:
10. Total damage: 24 + 10 = 34."* Overestimating damage → over-blocking → more Skills → (vs Nob)
more Enrage → damage genuinely rises. The comprehension bug and the strategy trap **compound**.

**(c) Play is per-decision greedy; defence loses the energy auction.** Over all combat: on
decisions where a full block was achievable against incoming ≥40% of current HP, the model chose
a non-block action **62%** of the time (310/499); it also failed to take an available lethal
**28%** of the time (387/1370), prolonging fights. Yet it almost never *ends turn* leaving
affordable block unplayed (5/1750) — i.e. within a turn it spends attack-first and blocks with
whatever energy is left. Turn plans are re-derived per decision (Markov prompting), so there is
no "this turn I block" commitment.

Not the problem (worth ruling out): drafting mix is sane (307 Attack / 277 Skill / 120 Power
takes; 59/100 decks have AOE); campfire policy is sensible (rest 95% at low HP, 40% at high);
potion use is real (334 drinks; only 0.4 potions held on average entering the fatal fight);
invalid actions ~0; thinking never truncates at 8192 (combat mean 908 tokens, p90 1484).

---

## 2. Why the iter2 training run didn't help (and slightly hurt)

The floor-level result was null (paired delta −0.46, CI [−1.91, +0.99] — see
`eval/paired.json`). The decision-level metrics are more damning — same eval seeds:

| metric | base | RWR+hinted adapter |
|---|---|---|
| block-blunder rate (full block available, incoming ≥40% HP) | 62% | 59% |
| **lethal forgone** | **28%** | **36%** |
| end-turn leaving affordable block (had energy+block) | 0.3% | 5% |

The adapter didn't move the thing hints taught — it *degraded* tactical sharpness. Contributing
causes, all visible in the artifacts:

1. **The dataset was ~self-cloning.** RWR multiplicities: 150/300 trajectories at weight 1, only
   59 up-weighted, 91 dropped — mostly "imitate yourself at temp 1.0".
2. **Hint corrections were homeopathic:** 738 corrected decisions (685 laundered + 53
   action-only) out of ~46k kept combat decisions ≈ **1.6%**. And 927/1108 hints were *lethal*
   hints, 181 block — neither addresses the Enrage/over-block death spiral at all (0 block hints
   fired in Nob fights, so hints didn't misteach Nob either — they were simply irrelevant to it).
3. **lr 1e-4 for 2000 steps** (effective batch 16) is hot for behaviour-preserving LoRA SFT;
   noise-injection is exactly what the micro-metric regression looks like. Checkpoints
   500/1000/1500 exist in `data/iter2_rwr_hinted/adapter/` — evaluating checkpoint-500 on the dev
   split is a one-eval test of the overshoot hypothesis.

---

## 3. Recommendations to improve play, in order

### Tier 1 — serializer/comprehension fixes (no GPU, strategy-neutral, precedented)

The 2026-06-16 comprehension pass measurably cut hallucinated defence 21%→7%. These are the same
class of fix, aimed at §1.3(a)/(b):

1. **State that damage numbers are final.** One clause in the incoming-damage note and the KEY's
   Strength entry: *"shown damage already includes Strength/Weak/Vulnerable — do not add them
   again."* Kills the double-count directly.
2. **Serialize persistent enemy powers.** Expose the missing monster statuses (Enrage, Artifact,
   enemy Metallicize, Ritual, Curl Up, Malleable, …) in `BattleContext.enemies()` +
   `describeBattleState`, and give them KEY entries. This is comprehension, not strategy — the
   player sees these in the real game. It puts the *cause* of Nob's strength growth on-screen at
   the moment the model chooses Skill-vs-Attack. (Binding change → edit patch, rebuild, regen.)
3. **Keep intent-effect explanations visible for the whole fight** (cheap Python-side complement
   to #2: once a move like BELLOW has been seen, keep its KEY line for the rest of the combat).

### Tier 2 — cheap measurements before more training (hours of GPU, not days)

4. **Eval checkpoint-500 (and 1000) of the existing adapter** on the dev split with the
   micro-metric battery — tests "training overshot" for one eval's cost.
5. **Temperature sensitivity:** the arms run at temp 1.0 everywhere. One dev-split run at
   temp 0.2–0.6 tells you how much floor is being lost to sampling noise vs capability.
6. **Held-out Δlog-likelihood probe** (the unmet Stage 6 criterion): does *any* adapter shift
   action likelihood on held-out states in the intended direction? This is the instrument that
   should gate every future training run — it detects movement long before floors do.
7. **Strategy-knowledge ceiling probe (diagnostic only, not training data):** rerun ~20 dev seeds
   with a one-paragraph mechanics addendum (e.g. the Enrage rule) in the prompt. If Nob deaths
   collapse, Tier-1 fixes will convert; if not, the gap is planning, not knowledge — and training
   has to carry more weight. Keep this clearly outside the neutral data path.

### Tier 3 — make training actually bite

8. **Switch the offline signal from self-cloning to contrast.** Two built-or-nearly-built options:
   - **Best-of-K expert iteration:** generate K=4–8 rollouts per train seed (`run_until
     --rollouts-per-seed`, temp 1.0), keep only each seed's best-floor trajectory (within-seed
     selection controls for world difficulty exactly like a GRPO group). This produces genuinely
     selected data instead of RWR's near-uniform weights.
   - **Signed-advantage PG** (`build_pg_dataset --mode offline` → `train_pg.py`) on the same
     K-per-seed pool — pushes below-baseline behaviour *down* rather than cloning it less.
9. **Decision-level blunder filtering (new, cheap, high-leverage):** the affordances already
   recorded per decision let the dataset builder *drop known-blunder decisions* (missed lethal,
   forgone full-block under the same thresholds as `hinting.detect_mistake`) from kept
   trajectories — don't clone the 28%/62% mistakes even inside winning runs. Tactical-truth,
   trait-neutral, no new rollouts, ~30 lines in `dataset_builder`.
10. **Distill the C++ search agent for combat.** The single biggest lever if full-LLM combat is
    the goal: the MCTS searcher is a strong combat teacher already in-repo. Needs one binding
    addition (best action for the current `BattleContext` — `BattleScumSearcher2` already
    enumerates and searches; expose a `search_best_action(bc, simulations)`), then a data pass
    that relabels recorded combat states (or drives DAgger-style rollouts) with teacher actions
    for SFT. Dense expert supervision on exactly the decisions where the model bleeds, versus a
    1-bit-per-700-decisions floor reward.
11. **Fix the hyperparameters with the new instruments:** lr sweep {1e-5, 3e-5} × the Δlogp probe
    (#6) + micro-metric battery, before any 2000-step run.
12. **Targeted hard-fight data:** oversample elite/boss combat decisions in the loss (weight by
    encounter type, already recoverable from `state["combat"]["enemies"]`), or use the
    Interactive Studio's replay-to-branch machinery to generate many samples of the same elite
    fights. The model doesn't need 46k trash-fight decisions re-cloned.

### Ordering rationale

Tier 1 attacks the identified mechanisms for free and helps *every* later arm (better data,
better teacher-matching, better eval). Tier 2 builds the instruments that made §2's diagnosis
possible into the standard loop, so the next training run is judged in hours not pod-days.
Tier 3 is ordered by (evidence it addresses a measured failure) / (new machinery required):
8 and 9 reuse existing infra; 10 is the big bet with a small binding cost.

---

## 4. Meta-analysis: gaps in the current metrics/tooling

Everything in §1–2 was computable from data the harness already records — which is a compliment
to the recording layer (especially `affordances` and `hp_trajectory`) and a criticism of the
reporting layer, which reduces a rollout to `final_floor`/`win_rate`/`invalid_rate` and threw
away the story. Specific gaps, with proposals:

1. **Floor is a low-sensitivity instrument; the micro-metrics are the sensitive ones.** The
   trained-vs-base comparison was "null" on floor but clearly resolvable on lethal-forgone
   (28→36%) and end-turn-leaving-block (0.3→5%). **Proposal:** promote a decision-level "blunder
   battery" (block-blunder @ parameterized threshold, lethal-forgone, end-turn-leaving-block,
   potion-drink rate) into `eval_metrics.py`, and have `compare_paired.py` optionally read the
   JSONLs to report them per arm beside floors. These should also stream to wandb per GRPO
   iteration — an optimizer that games floor will show up here first.
2. **No death/funnel report existed.** Deaths-by-encounter, entry-HP-at-fatal-fight, and the act
   funnel (P(reach act 2) etc.) were the fastest route to "what's actually wrong". **Proposal:**
   promote `scratch/analyze_gemma_transcripts.py` into `scripts/` (split: `death_report.py` +
   the blunder battery in `eval_metrics`), with unit tests over fixture records, and add funnel
   rates (`reached_act2/3`) to `RolloutMeta.extra` or the compare report.
3. **Per-combat economics is the right unit of combat analysis** (HP-loss-by-encounter found the
   elite wall instantly) but nothing groups decisions into combats today. **Proposal:** a small
   pure helper (`eval_metrics.iter_combats(records)`) shared by the death report and future
   in-combat risk proxies (research_plan item 5 wants this grouping anyway).
4. **Per-encounter context belongs in the meta sidecar.** `RolloutMeta` records `hp_trajectory`
   but not *where* HP was lost. Adding a compact `extra["combats"]` list (floor, encounter,
   entry/exit HP, turns) at rollout time would make every analysis in §1 meta-only — no JSONL
   scan, so it works on synced pods where JSONLs are heavy.
5. **Transcript triage is manual.** Reading the fatal Nob fight was the highest-value 10 minutes
   of this analysis; there's no tool for "dump the fatal combat of run X with thinking".
   **Proposal:** `scripts/dump_fight.py <rollout.jsonl> [--last|--floor N]` producing the compact
   per-decision table used here (turn/HP/block/energy/intent/incoming/chosen + thinking on
   demand). Cheap, and it makes qualitative review a habit rather than an expedition.
6. **Comprehension-error probes are possible and free:** the strength double-count is
   regex-detectable in stored thinking (model restates `(deal N)` then adds Strength). A small
   "misunderstanding scanner" over `agent.thinking` (damage-arithmetic mismatches, references to
   nonexistent mechanics) would quantify comprehension bugs and verify Tier-1 fixes actually
   change the model's stated reasoning, not just outcomes.
7. **Statistics:** the sign test discards magnitude; add Wilcoxon signed-rank to
   `eval_stats.py` for the paired floor deltas (still dependency-free), and treat "floor
   reached" as right-censored where budget-truncated. With K>1 per seed (the built
   `--rollouts-per-seed`), also report within-seed std so eval power is visible in the report
   itself.
8. **Provenance nit:** eval arms currently share the same agent label dir name
   (`vllm_..._thinking_8192`) distinguished only by parent dir; adding adapter identity to
   `agent_label` (flagged in the 2026-07-06 repo review) becomes more urgent once
   checkpoint-sweeps (#4, Tier 2) multiply the arms.

---

## 5. Suggested immediate sequence

1. Tier-1 serializer fixes (1–3) + regression tests; regenerate the binding patch.
2. Build the blunder battery + death report into `eval_metrics`/`compare_paired` (§4.1–3) so the
   next GPU run is scored with sensitive instruments.
3. One pod-day of Tier-2 measurements: checkpoint-500 eval, temp sweep, Δlogp probe, strategy
   ceiling probe.
4. Decide the Tier-3 lever from those results — expected default: best-of-K + blunder-filtered
   SFT (8+9) as the next training run, search-agent distillation (10) as the follow-up build.
