# RL & Framing: Design Discussion

**Participants:** Sid × Claude (Opus 4.8)
**Date:** 2026-06-15
**Status:** Design discussion — the RL training arms below are *not yet implemented* (we are still pre-data-collection; see [`research_plan.md`](research_plan.md) "Project status: early development"), so treat them as working positions, not frozen decisions. **Exception:** the **seed model** in §2c was implemented on 2026-06-15 (world-seed / policy-seed separation enabling multiple rollouts per game state).

This doc captures a session working through three things and how they fit together:

1. Whether per-rollout-reward RL is even possible given the harness's "model sees one turn at a time" design.
2. What kinds of RL are worth implementing here.
3. The interplay between RL and the framing research question — in particular, whether we can induce risky / risk-averse models by *filtering training data on behavior*, and how that confounds (or doesn't) with the framing manipulation.

---

## TL;DR — positions we landed on

- **Per-rollout-reward RL works fine with the current Markovian, fresh-context-per-turn setup. It is the standard setup, not a compromise.** The model never does credit assignment; the *training loop* does, via the scalar weight it attaches to each decision. Fresh context per turn is just the definition of a Markov policy.
- **The only real caveat is observation sufficiency**, which is a *serializer* question (is `state_text` Markovian?), already tracked in the plan as the "state sufficiency audit." The fix for insufficiency is a better serializer, **not** stuffing history into the context.
- **Start with the offline / fixed-rollout arm** (filtered behaviour cloning → reward/advantage-weighted regression). It needs only the data we already record, plus a per-trajectory reward join. It is the cleanest isolation of the framing-interpretation effect and matches the plan's "Fixed Rollouts First" commitment.
- **For the on-policy arm, prefer GRPO or RLOO over PPO** — group-relative baselines avoid a value/critic network (meaningful on a 48 GB Mac with a 4B model). Sample at temperature > 0 (temp 0 gives zero within-group variance → zero advantage).
- **Keep the reward trait-neutral (pure competence).** Risk-flavoured reward shaping confounds the very thing the experiment measures.
- **Inducing traits by filtering data on behaviour is a legitimate but *different* lever from framing** (data-distribution manipulation vs. interpretation manipulation). It is a valuable *comparison arm*, not a replacement.
- **The clean way to do behaviour-filtering is to filter on behaviour within a fixed outcome stratum** (e.g. winners only / floor-matched), holding competence fixed and varying only the trait. Both naive recipes (pure behaviour filter; "risky-wins + averse-losses") confound trait with competence.

---

## 1. Can we RL with a per-rollout reward given Markovian, fresh-per-turn contexts?

### The worry

The harness is deliberately Markovian: `ActionAgent.choose_action(state_text, legal_actions)` receives a fresh, self-contained prompt each decision — no transcript, no after-state, no record of combats resolved in between (see `src/sts_ai/CLAUDE.md`, "The agent sees only the current state", and the plan's *Markovian State Prompting* commitment). The reward we can compute is per-*rollout* (did it win, final floor, HP). Does crediting a whole-trajectory reward back to individual decisions require the model to carry context across the run?

### Answer: no — and this is the standard setup

**The model does not do credit assignment; the training loop does.** Each decision is a `(prompt_i, action_tokens_i)` pair, where `prompt_i = framing + state_text + legal_actions` and `action_tokens_i` is the emitted `{"reasoning", "action_index"}` (plus any `<think>`). A rollout produces one scalar reward `R`. The policy-gradient estimator is:

```
∇J  =  Σ_i  A_i · ∇_θ log π_θ(action_tokens_i | prompt_i)
```

Two observations:

1. `∇ log π(action_i | prompt_i)` is computed by running the model on **`prompt_i` alone** — you never feed it `prompt_1..prompt_{i-1}`. A fresh context is exactly what this term assumes.
2. The "credit assignment across time" is the scalar `A_i` (advantage / weight) you attach to decision `i`. In the simplest case (REINFORCE), `A_i = R − baseline` for *every* decision in the trajectory. **This is where the per-rollout reward enters**, and it is computed in Python from the terminal state, not by the model attending over history.

A stateless-per-turn policy is just the statement that π is a function of the current observation — the definition of a **Markov policy**. Nothing in REINFORCE / GRPO / PPO requires a recurrent policy. (You can even view it as a contextual bandit where each state→action shares the trajectory's reward; the gradient math is identical for the trajectory-return case.)

### The one real caveat: Markovian sufficiency = serializer quality

The setup is *correct* only insofar as `state_text` genuinely contains all decision-relevant state. If the serializer drops something that matters (e.g. *what happened in the combat the search agent just resolved*, a relic interaction), it becomes a POMDP. RL still "works" (the gradient is still valid), but achievable performance is capped by what a memory-less policy can express, and credit assignment is noisier because `R` depends on hidden variables.

This is exactly the plan's **state sufficiency audit** risk. **The fix is a better serializer, not a longer context.** Dumping the transcript into the prompt is a different (agentic-memory / POMDP-history) design that we don't need for credit assignment, that would make the framing comparison messier and the fixed-rollout arm ill-defined, and that burns context length. If the audit finds a *specific* missing signal, add that one field — targeted state augmentation, not full history. For the framing experiment the Markovian setup is arguably *desirable*: the only thing differing between conditions is the framing block.

---

## 2. What RL to implement

The plan already names two arms (*fixed-rollout* and *on-policy*). They map onto two different RL families; per-rollout reward works in both.

### 2a. Offline / fixed-rollout arm — start here

This is **not** online RL; it is offline reward-weighted learning on already-collected trajectories. In rough order of complexity:

- **Filtered behaviour cloning** (a.k.a. rejection-sampling fine-tuning / STaR-style): keep decisions from winning / high-floor trajectories, do ordinary SFT on the action tokens under the framing prefix. Simplest, most robust, hardest to break.
- **Reward-/Advantage-weighted regression (RWR / AWR):** use *all* decisions, weight each loss by `exp(A_i / β)`. The continuous generalisation of filtered BC; doesn't throw away losing trajectories.

**We can do this with the data we already store.** Each `DecisionRecord` (`schemas.py`) has `state_text`, `legal_actions`, and `agent.raw_response`; `LightspeedHybridEnv.summary()` exposes `outcome / floor / cur_hp / gold` for the per-trajectory reward. RWR / filtered-BC need only the *text* of the chosen action and a trajectory weight — **not** stored token-level logprobs (those are only needed for importance-sampling-corrected off-policy PG, a higher-variance method we are deliberately not doing). Crucially, **framing is not baked into `state_text`** — it's added by `render_action_prompt` (`prompting.py`) — so we can re-wrap the same stored neutral state with a different framing string. *That re-wrapping is the framing experiment.* The only missing piece is a small "attach reward label per trajectory (join by seed)" step, already listed in Stage 5.

### 2b. On-policy arm — GRPO or RLOO

When we generate fresh rollouts per frame and train with policy gradient, the current standard for LLMs is **GRPO** (group-relative: sample a group of `G` rollouts, advantage `= (R_i − mean)/std`, clipped update, **no value network**) or its simpler cousin **RLOO** (REINFORCE with a leave-one-out group mean as baseline). Prefer these over **PPO** specifically because they avoid a critic network — meaningful given the 48 GB Mac / Qwen3-4B constraint (and the plan's "Local Training Is a Benchmark, Not an Assumption" caution). Group over *seeds* (sample `G` rollouts of the same seed; reward = win / floor); the group mean is the baseline, which tames the sparse terminal-reward variance without a learned value head.

### 2c. Seed model: world seed vs. policy seed

*(Implemented 2026-06-15; the RL training arms in the rest of §2 remain unbuilt.)*

Reproducibility and the on-policy arm both depend on separating two kinds of randomness that used to share a single seed:

- **World seed** — fixes the game world: map, card/relic/shop/event rolls, combat shuffles, monster moves. Same world seed ⇒ same environment instance.
- **Policy seed** — fixes the policy's stochastic sampling (LLM token sampling at temperature > 0; the random baseline's choices). It is derived deterministically from the world seed and a **rollout index**, so the policy seed follows from the pair.

A rollout's identity is therefore **(world seed, rollout index)**: hold the world seed fixed and vary the rollout index to get K different — but individually reproducible — plays of the *same* game.

**When to vary which:**

- **Fixed-rollout / offline arm at temperature 0:** vary the world seed only. The policy is deterministic, so the rollout index is irrelevant — one play fully represents a world.
- **On-policy RL (GRPO / RLOO groups):** fix the world seed per group and vary the rollout index across the group (0…G−1) at temperature > 0. The group is "the same game, sampled G ways," so the group-relative baseline measures *this sample vs. the others on the same world* rather than "this world was easy." This is the concrete form of the "group over seeds" baseline in §2b — and the reason temperature 0 collapses a group to zero variance (gotcha #2).
- **Evaluation / variance reduction:** average over rollout indices at a fixed world seed; the world seed acts as a *block* that removes game-difficulty noise from a policy or framing comparison.
- **Action propensities (Stage 8):** many policy samples at a fixed state ≈ many rollout indices sharing a world (and prefix).

**Reproducibility — record both seeds, and mind the granularity:**

- Re-running with the same (world seed, policy seed) reproduces a rollout. Record both on every trajectory.
- The **serial** path (one rollout at a time) is **bit-reproducible** — identical tokens on re-run. Use it for any data we intend to freeze.
- The **batched** path (many rollouts generated together for throughput) is **re-run-deterministic given a fixed batch size** — the same command reproduces — but it is *not* bitwise-identical to the serial path nor invariant to batch size. Batched generation pads and batches the matmul (so a sequence's logits depend on the batch shape) and the sampler RNG is shared across the batch. Practical rule: freeze the (path, batch size) for anything you will compare bitwise; otherwise treat batched output as "same distribution, reproducible per fixed config," not "same bytes as serial."

**Combat randomness (hybrid mode):** when combats are resolved by the built-in search agent, that search has its own randomness that is currently *held fixed* (not varied per rollout). So K rollouts of one world seed in hybrid mode differ only in the **out-of-combat** choices — which is exactly what the OOC-policy RL arm trains, and holding combat fixed *reduces* variance on that credit assignment. Varying combat across rollouts would need deeper control and is deferred; under full-LLM-combat control the combat decisions are the policy's, so the policy seed already covers them.

### Reward design: keep it trait-neutral

The reward is sparse (win/lose at the end of a hundreds-of-decisions episode), so the temptation is to shape it. **Resist risk-flavoured shaping.** The experiment asks "does *framing* decide which trait absorbs the same competence update?" If the reward itself rewards risk, the manipulation is confounded — we could no longer attribute trait generalisation to the frame. Keep the reward pure competence (terminal win + floor + HP; maybe combats-won as light, trait-neutral shaping) and handle variance with a **baseline** (GRPO group mean / value head), not with risk-laden reward terms. This is consistent with the plan's "performance-adjusted risk behavior" readout.

### Harness-specific gotchas (apply to both arms)

1. **Invalid outputs terminate cleanly.** `parse_json_action` still returns `action_index=0, valid=False` as a structured placeholder, but rollout generation records the invalid response and stops with `agent_invalid` before executing a fallback action. Naively training on invalid terminal records should still be masked or assigned a format penalty, but malformed generations no longer ride a fallback trajectory reward.
2. **`temperature=0` kills on-policy RL.** Fixed-rollout config uses temp 0 for reproducibility, but then every rollout of a seed is identical → GRPO/RLOO group advantages are all zero. The on-policy arm **must** sample at temp > 0. (Different configs for different arms.)
3. **Hybrid combat confounds credit assignment.** In `combat_control="search"`, the reward partly reflects the *search agent's* competence, and a strong searcher "masks consequences of bad pathing/reward choices" (plan, Known Limitations) — exactly on the risk-relevant out-of-combat decisions. `combat_control="llm"` makes the whole trajectory the LLM's actions (cleaner credit) but episodes are much longer/slower and combat competence becomes a prerequisite. **Recommendation: start RL on the hybrid, out-of-combat action space** — pathing / campfire / reward / shop *are* the risk-relevant decisions of interest — before paying for full-combat RL.
4. **Where the gradient lands (reasoning vs. action) is a science decision, not just a knob.** The plan already contemplates action-only vs. full-trajectory loss. Framing plausibly acts *through reasoning style*, so masking the `<think>` / `reasoning` tokens may blunt the mechanism we're studying. Default to letting the gradient flow through reasoning too, but keep it a logged, switchable flag.
5. **Errored / truncated episodes** (seed-2-class hangs, JSON truncation at the token budget) need a defined reward or to be dropped — the loop must not silently treat a crashed run as a loss.

---

## 3. The framing research question (recap)

The motivating hypothesis ([`research_plan.md`](research_plan.md)):

> When a model is reinforced for behaviour along a graded axis such as risk-taking, does the *framing* of the training context decide which broader latent trait absorbs the update?

Two experimental arms:

- **Fixed-rollout arm** (the MVP): generate neutral trajectories once, then train framing variants on the *same* states/actions/reward, changing only the framing block. This isolates the *interpretation* effect — same data, same reward, different prefix. The gradient is still frame-conditioned because token probabilities differ under different prefixes; that frame-conditioned gradient *is* the mechanism under test.
- **On-policy arm:** generate trajectories *under* each framing, train each on its own data. Captures interpretation **plus** data-distribution effects. The gap between the two arms is evidence about how much of the framing effect is mediated by the visited trajectory distribution.

The fixed-rollout arm is prioritised because it controls data and reward, is cheaper, and is cleaner for debugging.

---

## 4. Interplay: inducing traits by *data selection* vs. by *framing*

Sid's question: instead of (or alongside) framing, can we make "risky" and "risk-averse" models *independently of framing* by removing training samples that don't exhibit the target behaviour? And if so, should we keep only matching trajectories, or use some win/lose filter too?

### This is a different lever

- **Framing arm:** fix data + reward, vary *interpretation*. Measures "which trait absorbs the same update."
- **Behaviour-selection arm (proposed):** vary the *data distribution* directly to install the trait. This doesn't test framing — it is a "direct trait installation" baseline.

Both are valuable, and the behaviour-selection arm is a strong **comparison/control** for the framing arm, but they answer different questions — keep them conceptually separate.

### The principle that resolves the "which filter?" question

In filtered BC, **what you select on *is* your reward.** Filtering is reward-weighted regression with a 0/1 weight = `indicator(selection criterion)`:

| You filter on… | Effective reward | What the model learns |
| --- | --- | --- |
| `won` | competence | play well (the original arm) |
| `risky` | the trait | be risky **regardless of whether risky is good** |
| `risky AND won` | "risky competence" | **conflates the two — can't separate post-hoc** |

Design rule: **each contrast should select on exactly one axis and hold the other fixed.** Confounds appear the moment a single arm selects on behaviour *and* outcome together.

### Why both naive recipes confound

**Option A — pure behaviour filter ("only ever sees risky"):** maximally separates the arms on the trait, but confounds trait with competence, and here the confound is *directional and severe*. In the baseline data, the more HP-conservative / low-HP-resting policies do better (`heuristic`: rest@lowHP 1.0, final floor 16.2, final HP 49.2; `random`: rest@lowHP 0.25, floor 14.4, HP 16.0 — Stage 4 table). So:

- the **risk-averse** model trains mostly on *winning, high-floor* trajectories → learns cautious **and competent** play;
- the **risky** model trains mostly on *losing, low-floor* trajectories → learns risky **and incompetent** play.

Any downstream "the cautious model is more capable / coherent" would then be an artifact of the competence gap, not trait generalisation. (It also selects the whole *bundle* correlated with risk — shop overspend, card skips — so "risky" isn't pure.)

**Option B — "risky-wins + neutral/averse-losses":** avoid as stated. Filtered BC trains the model to *imitate the kept set*, so keeping losing cautious trajectories means **training the model to reproduce losing cautious play** — incoherent. And the asymmetry (risky must *win* to be kept; cautious is kept when it *loses*) makes the selection criterion effectively "risky = good, cautious = bad" — i.e. **risk-flavoured reward shaping wired straight into the data**, and *worse* than neutral shaping because it is perfectly correlated with the trait under study. (Reframing it as a DPO-style contrastive pairing doesn't help: pairing risky-win as "chosen" vs. averse-loss as "rejected" still couples trait and outcome.)

### The clean design: behaviour filter within a fixed outcome stratum

Control competence, then vary only the trait:

- Restrict the source pool to **winners** (or top-k by final floor) so all arms share one competence stratum.
- **Neutral** = all winning trajectories; **Risky** = winners whose risk-relevant decisions skew `risk_seeking=True`; **Averse** = winners that skew `risk_seeking=False`.
- **Match dataset sizes** across arms (subsample the larger).

This is just "control for the confounder by matching on it." The continuous generalisation is **reweighting instead of hard filtering** (weight each decision by `f(risk_score)` while holding the outcome distribution matched) — smoother, less wasteful; filtering is fine to start.

### Two gotchas specific to our labels

1. **Granularity mismatch.** `risk_proxies.classify_decision` emits a **per-decision** `risk_seeking ∈ {True, False, None}` (clear-direction only: `rest`=averse, `smith`@low/med-HP=seeking, `ELITE` node=seeking, Neow drawback=seeking, self-damage card take=seeking; shop / generic take / potion = `None`). Outcome (win/floor/HP) is **per-trajectory**. "Risky decisions that won" assigns a trajectory label to a decision → noisy credit assignment (a risky decision in a winning run wasn't necessarily *why* it won). Cleaner: do the **trait** selection at the decision level (we have `risk_seeking` per decision) and control **competence** at the trajectory level (winners-only / floor-matched pool). Keep the many `risk_seeking=None` (trait-neutral) decisions identically across all arms, or drop them uniformly — never let them differ between arms.
2. **Data scarcity bites in the wrong direction.** Since risk-aversion *is* the winning strategy here, "winning-and-strongly-risky" trajectories may be rare. Mitigations: relax "win" → "top-k floor"; reweight instead of hard-filter; or accept a *measured, reported* competence gap and check every downstream finding against it (always report floor / win-rate per arm so a trait effect is distinguishable from a residual competence difference).

### How it complements framing

The contrast between "trait installed by data selection" and "trait nudged by framing on identical data" is itself a deep readout: it tells us whether framing does something *other* than soft data-selection. A later 2×2 ({neutral, risky-filtered} × {neutral-frame, adventurous-frame}) is conceivable — but isolate one lever per contrast first.

---

## 5. How this maps to the existing roadmap

- **Stage 5 (fixed neutral rollout collection):** add the "attach per-trajectory reward label (win / floor / HP) by seed" post-processing step — it's the only missing input for the offline arm.
- **Stage 6 (training feasibility):** the first training target is **filtered BC → RWR** on neutral rollouts (offline arm). Loss masking choices (reasoning vs. action) are the "where the gradient lands" decision above.
- **Stage 7 (three-frame experiment):** the offline arm with the framing prefix swapped — directly supported by the fact that framing is not baked into `state_text`.
- **Stage 9 (on-policy arm):** GRPO / RLOO, out-of-combat action space first, temp > 0, trait-neutral reward, group-over-seeds baseline.
- **New (optional) arm:** the behaviour-selection / matched-outcome design in §4, as a comparison condition for the framing arms.

---

## 6. Open questions / decisions to make before building

- **Operational definition of "risky"** for filtering/reweighting: single proxy (e.g. campfire) vs. composite from `RiskEvent`s? A single-proxy filter makes "generalisation to other risk dimensions" a clean readout; a composite installs a bundle.
- **Trajectory-level risk-skew score:** how to aggregate per-decision `risk_seeking` into a per-trajectory label (fraction of clear-direction decisions that are seeking? net count?), and the threshold for "risky" vs. "averse."
- **Competence stratum:** "winners only" vs. "top-k floor" vs. continuous floor-matching — driven by how scarce risky-winners turn out to be.
- **Reward granularity for the offline arm:** trajectory-return broadcast to all decisions (simplest) vs. per-decision advantage/value baseline (lower variance, more machinery).
- **Reasoning in the loss:** action-only vs. full-trajectory — interacts with the framing mechanism; decide and log it.
- **Local vs. cloud training:** which of {filtered BC, RWR, GRPO} is feasible on the 48 GB Mac for Qwen3-4B, and where the cloud fallback kicks in (plan's training-feasibility benchmark).

---

## Appendix: where these claims are grounded in the code

- **Markovian prompt / fresh context:** `prompting.render_action_prompt`, `agents.ActionAgent.choose_action`; `src/sts_ai/CLAUDE.md` "The agent sees only the current state."
- **What's recorded per decision:** `schemas.DecisionRecord` (`state_text`, `legal_actions`, `selected_action`, `agent`, `phase`), written in `rollout.run_rollout`.
- **Framing not in `state_text`:** `rollout.py` records the glossary-augmented `state_text`; the framing is prepended later inside `render_action_prompt`.
- **Per-trajectory reward available:** `lightspeed.LightspeedHybridEnv.summary()` → `outcome / floor / cur_hp / max_hp / gold`.
- **Invalid output handling:** `agents.parse_json_action` returns `action_index=0, valid=False` as a placeholder; rollout code records the invalid response and terminates with `agent_invalid` without executing it.
- **temp=0 for fixed rollouts:** Stage 5 canonical command in `research_plan.md`.
- **Hybrid vs. full combat:** `combat_control` in `lightspeed.py`; `src/sts_ai/CLAUDE.md` "In-combat (full-control) mode."
- **Per-decision risk labels:** `risk_proxies.classify_decision` / `RiskEvent.risk_seeking`.
- **Baseline competence/behaviour numbers:** Stage 4 Qwen evaluation table in `research_plan.md`.
