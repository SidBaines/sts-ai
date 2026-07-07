# Repository Review — 2026-07-06

**Scope:** full read-through of the harness (`src/sts_ai/`), training stack (`src/sts_ai/train/`),
scripts, tests, docs, the pending diff on `feat/rwr-and-hinted-rollouts`, and the on-disk run
artifacts (`data/iter2_rwr_hinted/`). Reviewed from two angles: **science** (does the experimental
design support the research question?) and **code** (correctness, robustness, maintainability).

**State at review time:** branch `feat/rwr-and-hinted-rollouts`, 9 files modified uncommitted
(GRPO telemetry/resume/gradient-checkpointing + docs), full test suite green (368 tests, 8
env-gated skips). The first offline RWR+hinted GPU run (iter2) is complete; the first GRPO GPU
run was launched 2026-06-20.

---

## 1. Overall summary

### 1.1 What this repo is doing well

This is an unusually disciplined research codebase. Specific strengths worth preserving:

- **The experimental design is clearly articulated and internally consistent.** The research
  question (does *framing* decide which latent trait absorbs a competence update?) has an explicit
  independent variable (the framing string, injected only in `prompting.render_action_prompt`),
  and the two invariants that protect it — **prompt neutrality** (base prompt is comprehension-only)
  and **trait-neutral reward** (`docs/reward_spec.md`) — are enforced consistently everywhere I
  checked: glossary text, hint strings, reward definitions, and the `require_framing_match` guard
  in the dataset builders. `docs/rl_and_framing_design.md` is genuinely good methodology (the
  behaviour-filter confound analysis in §4 is the kind of thinking that usually only happens after
  a wasted experiment).
- **Clean architectural boundary.** `state_text + legal_actions → ActionAgent.choose_action → AgentDecision`
  is held everywhere; the simulator wrapper, recorders, and trainers never touch a model provider.
  The three orchestrators (serial / MLX lockstep / vLLM streaming) share record-building helpers so
  trace shape is identical — this is what makes the offline arm trustworthy.
- **Train/inference skew is treated as the central risk, correctly.** Verbatim `raw_response` as
  the target, prompt reconstruction through the same `render_action_prompt` + chat-template path,
  the single shared `chat_template_probe_hash`, and exclusion of retry-augmented records are all
  the right calls.
- **Fail-closed simulator policy** (throw-not-assert, error sidecars, UB latching, subprocess
  timeouts) plus the dry-run discipline doc (`grpo_dryrun_checklist.md`) show hard-won operational
  maturity.
- **Reproducibility model** — world-seed / policy-seed separation, per-request seeds on the
  streaming path (independent of batch composition), documented granularity caveats.
- **Tests and docs.** 368 fast tests, tiered unit/integration, regression test per bug; the
  CLAUDE.md hierarchy and handoff docs make the repo navigable by a fresh agent — rare and valuable.

### 1.2 The headline science finding of this review

**The offline gating result (deliverable 1 of the GRPO-readiness handoff) is a null, and the
formal paired analysis was never actually produced until this review.** The iter2 pipeline's
compare step failed with "no labelled rollouts with meta sidecars" (the pre-fix non-recursive
glob); the only recorded readout was a scratch-script headline. Rerunning `compare_paired.py`
(now fixed) on `data/iter2_rwr_hinted/eval/{base,trained}`:

| | base | trained (RWR+hinted) |
|---|---|---|
| n rollouts (eval seeds, K=1) | 100 | 100 |
| act-boss-clear rate | 0.23 | 0.22 |
| mean floor | 14.41 | 13.95 |
| agent_invalid_rate | 0.0 | 0.0001 |

Paired per-seed delta: **−0.46 floors, SE 0.75, 95% CI [−1.91, +0.99], sign test p≈0.50** (36
seeds up, 43 down, 21 tied).

The handoff's own rule: *"If trained doesn't beat base, fix the offline arm before any on-policy
work — GRPO cannot rescue a reward/eval that can't detect signal."* The GRPO run was launched
anyway (deliberate user call, per the session memory), but the null result has not been diagnosed
or recorded anywhere in `docs/`. **Diagnosing why the offline arm didn't move the policy is now
the single highest-value science task in the repo** — more below (§3, step 1). Note also this
eval was K=1 per seed, not the K=4 lower-variance eval that was built for exactly this purpose:
at SE 0.75 the eval could not have detected anything smaller than a ~1.5-floor effect anyway.

### 1.3 Overall verdicts

- **Code: strong.** No correctness bugs found in the recording/eval path (the part where a bug
  silently corrupts science). The issues found are concentrated in the newest, least-GPU-exercised
  code (the GRPO loop) and are listed below.
- **Science: the harness is ahead of the results.** Measurement, seeding, stats, and anti-hacking
  scaffolding are excellent, but the pipeline has not yet demonstrated it can detect *any* training
  effect, and two of the plan's own methodological commitments (loss masking options, the
  invalid-as-loss masking) are specified in the docs but not implemented in the trainers. The
  framing experiment itself (the actual research) has essentially no code yet — only one frame
  string exists (`NEUTRAL_FRAME`), and the main generation/eval CLI (`run_until.py`) can't vary it.

---

## 2. Prioritised fixes

Ordered by (impact on scientific validity) × (cost of discovering the problem later).

### P0 — before or during the next GPU spend

1. **Diagnose and record the null offline result; make the K=4 paired eval the standard.**
   The go/no-go for the whole RL ladder is undecided. Concretely: (a) commit the paired report
   produced above into `data/iter2_rwr_hinted/eval/paired.json` + a short writeup in `docs/`;
   (b) before blaming the *method*, check whether training moved the policy at all — the Stage 6
   acceptance criterion "action likelihood changes on held-out examples in the expected direction"
   was never evaluated. A base-vs-adapter log-likelihood comparison on held-out decisions is
   CPU/1-GPU-cheap, needs no rollouts, and cleanly separates "training did nothing" (LoRA/lr/steps
   issue, or self-BC on 70%-multiplicity-1 data ≈ identity) from "training worked but doesn't
   transfer to floors" (reward/eval issue). The iter2 RWR histogram
   (150/300 trajectories at multiplicity 1, only 59 up-weighted >1, β=5) suggests the effective
   dataset was close to uniform self-cloning — a plausible sufficient explanation for a null.

2. **GRPO trainer: the within-iteration objective is not what it says it is.** One GRPO iteration
   generates ~12.5k decision-examples (measured), and `train_pg_trl` then takes ~1,500 optimizer
   steps over them (effective batch 8, 1 epoch) with `logp_old = logp_new.detach()` recomputed
   *per step* (`src/sts_ai/train/train_pg_trl.py:194`). Consequences: the ratio is identically 1,
   so `clip_eps` never binds and `clip_fraction` is meaninglessly 0; and every step after the first
   is off-policy w.r.t. the generation policy with no importance correction and no trust region —
   ~1,500 uncorrected stale-data updates per iteration. This is *not* TRL's μ=1 regime (there, one
   generation batch feeds one update). It may still work (it's ≈ one epoch of advantage-weighted
   BC with a KL-to-base anchor), but it is exactly the regime clipping exists for, and it should
   be either fixed or explicitly re-labelled. Cheapest fixes, in order of rigor:
   (a) snapshot behaviour log-probs once per iteration (one extra forward pass over the dataset
   before training, or capture vLLM logprobs at generation time) so ratio/clipping are real;
   (b) cap optimizer steps per iteration (`max_steps`) so staleness is bounded;
   (c) at minimum, watch `mean_kl` per iteration and document that the loss degenerates to
   REINFORCE-with-KL. Note the eval-noise context: with the current eval SE (~0.75 floors), an
   unstable optimizer and a working one look the same for many iterations.

3. **Format-failure trajectories pollute the GRPO reward exactly the way the repo's own docs warn
   against.** An `agent_invalid`-truncated rollout keeps its (low) `final_floor` and enters
   `group_relative_advantages` as a plain loss; the invalid record itself is skipped from the
   dataset, so the *other* (possibly good) decisions of that trajectory absorb the negative
   advantage while the format failure is never directly penalised. `reward_spec.md` §3 and
   `rl_and_framing_design.md` gotcha 1b both say to mask or explicitly penalise these. Currently
   invalid rates are ~0 for Gemma-4-E4B, so the impact is nil *today* — but on-policy drift at
   temp 1.0 is precisely what raises it, and by then it corrupts the gradient silently. Fix in
   `pg_dataset.build_pg_dataset` / `advantage.group_relative_advantages`: exclude
   `stopped_reason=="agent_invalid"` trajectories from groups (like `simulator_error`), or assign
   them advantage 0, and report the count in the manifest.

4. **GRPO loop has no degenerate-iteration guard.** `VllmJsonAgent._generate` is deliberately
   fail-soft (`agents.py:579` returns `None` → all-invalid decisions) — right for sweeps, wrong
   for an unattended 20-hour training loop: a dead/OOM'd engine yields rollouts of floor≈0,
   near-empty datasets, and the loop happily trains on garbage for hours. Add cheap per-iteration
   aborts in `grpo_loop.run_grpo`: `n_examples < min`, `agent_invalid_rate > threshold`,
   `advantage_report.advantage_max == advantage_min == 0`. Also wire `eval_metrics.stall_metrics`
   into the per-iteration wandb payload — `reward_spec.md` says "watch these per-iteration" but
   nothing computes them in the loop.

5. **`--per-device-batch-size` defaults to the measured-unsafe value.** The co-residency memory
   records real-loop peaks on the 80GB H100: batch 4 → 79.5GB (0.6GB margin, judged UNSAFE),
   batch 2 → 75.9GB (chosen for unattended runs). Yet `scripts/run_grpo.py` defaults to 4 and
   `scripts/runpod/run_grpo.sh` defaults `PER_DEVICE_BATCH_SIZE:-4`. Per the repo's own
   "encode gotchas in defaults, not just docs" rule: default both to 2.

6. **Commit the branch.** ~600 lines of tested, load-bearing diff (GRPO telemetry, resume,
   gradient checkpointing, the eval-thinking-mismatch fix) are sitting uncommitted while the code
   they orchestrate runs on rented GPUs synced by rsync. Also `scripts/runpod/resume_grpo.sh` and
   `uv.lock` are untracked — commit or delete deliberately (if `uv` is being adopted, track the
   lockfile and say so in the README; if not, remove it).

### P1 — correctness/robustness, do soon

7. **Silent truncation in the PG trainer.** `train_pg_trl.encode_row` truncates at
   `max_seq_len=4096`; a row whose prompt alone exceeds it ends with an all-`-100` label row →
   `completion_mask` empty → the example contributes zero loss, silently. Combat prompts with a
   large glossary KEY plus (later) thinking completions will hit this. Count truncated and
   fully-masked rows and surface them in `trainer_log.json`/manifest.

8. **Add a token-level round-trip check for the skew guard.** `chat_template_probe_hash` catches
   template-text drift but not tokenizer-behaviour drift (the classic double-BOS from
   `tokenizer.encode(add_special_tokens=True)` on an already-templated prompt, or prompt/completion
   boundary merges). The ground truth is already recorded per decision (`prompt_tokens`,
   `completion_tokens` from vLLM): add a one-off assertion in the dataset builders that
   re-tokenising a sample of reconstructed prompts matches the recorded counts (±0). If vLLM and
   HF both double-BOS identically it's consistent skew and fine — but verify once, cheaply,
   instead of assuming.

9. **`VllmJsonAgent.choose_actions_batch` retry path drops `induce_reasoning`**
   (`agents.py:743` builds `render_action_prompt(...)` without the `induce_reasoning` flag that
   `_base_prompt` applies). Only affects prompted-reasoning models on the batch (non-streaming)
   path, but it's a one-line inconsistency with the serial/streaming paths. While there, hoist the
   4× duplicated retry-suffix string into a module constant.

10. **`compare_paired.aggregate_arm` should surface `stopped_reason` counts per arm.** A
    `simulator_error` rollout currently folds its partial floor into the per-seed mean invisibly;
    the sign test then compares arms with different error compositions. Add
    `stopped_reason_counts` (and consider excluding `simulator_error` rollouts from per-seed means,
    consistent with the advantage code).

11. **Drop zero-variance groups in GRPO before training.** A group whose K rollouts all reach the
    same floor yields advantage 0 for every decision; those examples contribute only the KL term —
    i.e. you pay full training compute (the dominant ~1hr/iter cost) to pull the policy toward the
    reference on no-signal data. Filter `advantage == 0.0` examples (or whole zero-std groups) in
    `build_pg_dataset(mode="group")` and report how many were dropped.

12. **Implement the action-only-loss switch the plan commits to.** Both the research plan
    ("keep the primary supervised/action loss on the final structured action tokens") and
    rl_and_framing gotcha 4 ("keep it a logged, switchable flag") specify a reasoning-masked loss
    option; no trainer has it — the completion mask always covers the entire `raw_response`
    including `<think>`/thought-channel tokens. This is not just a knob: for the framing
    experiment, whether the gradient lands on reasoning tokens is a stated science decision, and
    right now there is exactly one (undocumented-as-such) choice. `tokenize_example` already knows
    the completion span; adding a "mask tokens before the final JSON object" option is small.

13. **Schema-version drift.** `schemas.py` declares `SCHEMA_VERSION = 3` but the comment (and
    `research_plan.md`) only narrate v1→v2; nothing documents what v3 is (presumably the additive
    `action_executed`/`hint_applied` fields). One sentence in `schemas.py:6` fixes it. Also
    consider stamping the version into each meta (`RolloutMeta.schema_version` exists — good) *and*
    noting that bare decision JSONLs carry no version marker.

### P2 — hygiene

14. **Stale docs:** `NextStep.md` still presents Stage-10 combat control as "the immediate next
    priority" (done 2026-06-14) — retire it or repoint it at the GRPO-readiness handoff.
    `research_plan.md` still names Qwen3-4B as the model target throughout while all real training
    runs are Gemma-4-E4B on CUDA — record the switch (it had real consequences: VLM arch → TRL
    `assistant_only_loss` blocked → `completion_only_loss` workaround).
15. **Promote the scratch analysis.** `scratch/` (untracked) holds the actual iter2 analysis
    (`paired_eval_analysis.py`, `failure_analysis.py`, `floor_hist_by_arm.png`). Untracked scratch
    holding the only analysis of a $-run is a data-loss and reproducibility risk — promote what's
    load-bearing into `scripts/` or `docs/`, delete the rest.
16. **`Trainer.log(metrics)` on every `compute_loss` call** (`train_pg_trl.py:218`) bloats
    `log_history` (thousands of entries per iteration) and spams per-microbatch wandb when enabled;
    respect `logging_steps`.
17. **No CI.** The unit tier runs in 1.5s with zero deps — a GitHub Actions job would make the
    "tests green" invariant enforced rather than remembered, and `STS_REQUIRE_SIMULATOR=1` is
    already designed for a build-capable runner later.

---

## 3. Development plan (after the fixes)

Ordered; each step gates the next. The guiding principle: **the repo's bottleneck is no longer
infrastructure — it is demonstrating one detectable training effect, then spending it on the
framing question.**

### Step 1 — Close the offline loop with a diagnosis, not another run (CPU + ~1 GPU-hour)
- Held-out **action-likelihood eval** of the iter2 adapter vs base (Stage 6 acceptance criterion):
  mean Δlogp on kept-trajectory decisions vs dropped-trajectory decisions. If Δ≈0 everywhere,
  the trainer/data was too weak (self-BC, β too soft, 2k steps too few) — a training problem.
  If Δ moves in the right direction but floors didn't, it's an eval-power or transfer problem.
- Write the outcome + go/no-go into `docs/` (the missing deliverable-1 record), including the
  paired numbers above.
- Decide the offline lever accordingly: stronger filter contrast (winners-only), larger train
  split, β sweep, or skip straight to signed-advantage PG (next step) — which pushes *down* on
  below-baseline trajectories instead of merely cloning less of them, and is the more plausible
  instrument for a detectable effect.

### Step 2 — Offline signed-advantage PG vs RWR vs base (deliverable 3; 1 pod-day)
Already built (`build_pg_dataset --mode offline` → `train_pg.py`). Run it on the same 300
iter2 rollouts, eval with **K=4** on the frozen eval split (`eval_paired.sh`, the tool built for
this), and record the three-way comparison. Power check first: from the iter2 per-seed deltas,
SE at K=1/N=100 was 0.75 floors; verify K=4 brings the detectable effect below ~1 floor before
trusting a null.

### Step 3 — GRPO, resumed only after fixes P0-2/3/4 land (multi-pod-day)
- Re-launch with: invalid-trajectory masking, degenerate-iteration guards, stall metrics in wandb,
  batch-size default 2, and either old-logp snapshots or a per-iteration step cap.
- Watch per-iteration: mean floor (train seeds), `mean_kl`, completion length, invalid rate,
  stall metrics; eval vs the offline winner every N iterations (K=4 paired).
- Decide and document the **thinking question**: GRPO currently trains a no-thinking policy
  (`run_grpo.py` has no `--thinking`, and `grpo_loop`'s dataset call would refuse native-thinking
  data). The offline arm trained native-thinking Gemma. The two arms are not currently comparable,
  and gotcha 4 argues the framing mechanism may act *through* reasoning — if the framing
  experiment will use thinking policies, GRPO needs `enable_thinking` plumbing +
  `require_no_thinking=False` + a larger `max_seq_len` sooner rather than later.
- Revisit **hybrid vs full-combat for RL**: `run_grpo.py` defaults to `combat_control="llm"`
  (450–700-decision episodes, credit dominated by combat tactics), while the design doc's own
  recommendation (gotcha 3) is to start RL on the hybrid OOC action space — shorter episodes,
  8-10× cheaper iterations, and the OOC decisions are the risk-relevant ones the study measures.
  Holding combat fixed across a seed's group also means group variance is purely
  OOC-attributable — a cleaner GRPO baseline. Worth one deliberate paragraph in the plan either
  way; right now the default quietly contradicts the doc.

### Step 4 — Build the framing experiment surface (the actual research; mostly CPU)
This is where the repo is thinnest relative to its goal:
- **Define the frame set** in `prompting.py` (neutral / risk-reward / adventurous) as minimal
  edits of a common base so length/content confounds are controlled. Note `NEUTRAL_FRAME`'s
  "make the strongest choice you can" is itself an objective statement — decide whether variants
  add to it or replace it, and freeze that.
- **Thread `--framing` through `run_until.py`/`agent_factory`** (it exists only on `run_grpo.py`),
  and include framing (and adapter identity) in `agent_label` so arms can't collide on disk.
- **Fixed-rollout framing arm:** re-wrap the *same* neutral rollouts with each frame
  (`build_pg_dataset(..., require_framing_match=False)` — the flag exists and this is its purpose),
  train per-frame at matched update count, and audit stored `agent.thinking` for frame leakage
  first (the Stage 5 audit — not yet done on the iter2 data).
- **Freeze Stage 8 measurement before training the frames** (the plan's own rule): matched-state
  action-propensity probes (K samples per fixed state — the seeding model already supports this),
  the risk-proxy battery per frame, KL-to-base per frame, and the out-of-domain preference probes.
  Only the risk proxies exist today.

### Step 5 — Measurement upgrades that multiply everything above
- **In-combat risk proxies** (plan item 5): `risk_proxies.py` is OOC-only; full-combat rollouts
  contain the majority of decisions and the `affordances` fields (`full_block_possible`,
  lethal-taken) are already recorded per decision — an aggression/safety metric is mostly wiring.
- **Likelihood-based probe eval as a standing tool** (from Step 1) — it is the cheapest
  high-power instrument the repo can own, and the framing readouts (action-propensity shifts)
  are the same machinery.

### Step 6 — Debt to schedule, not to forget
- Freeze the full-game-derived seed splits (current splits are Act-1-derived; full-game is now the
  default depth), and a larger train split if offline arms continue.
- Root-cause the seed-2-class UB before any claim requires cross-machine reproducibility
  (currently accepted-and-excluded — fine for single-machine work).
- μ>1 batch-reuse with proper old-logp snapshots (subsumes fix P0-2a) and, if training stays
  the wall-clock bottleneck, the batch-2 + gradient-checkpointing + zero-advantage-drop trio.
- CI for the unit tier; decide the `uv` question; retire stale docs (P2 items).

---

## 4. File-level notes (for reference)

| Area | Verdict | Notes |
|---|---|---|
| `lightspeed.py` | Solid | Combat dedup + display→raw mapping correct given the `[enemy i]` disambiguation invariant; UB latching correct; documented index-stability assumption in `step()` is the one thing to watch. |
| `rollout.py` / `parallel_rollout.py` / `streaming_rollout.py` | Solid | Shared record builders keep the three paths trace-identical; streaming hint stage machine is careful (single commit point, 1:1 slot↔in-flight, stage-gated seeds). |
| `agents.py` | Good | Parser handles think-blocks/Gemma thought-channel/truncation with metadata; fail-soft generation is right for sweeps but needs the loop-level guard (P0-4); minor retry-path inconsistency (P1-9). |
| `hinting.py` | Good | Pure, tactical-truth only, provenance-rich; the `reasoning_format` tag mismatch on action-only fallbacks is known and test-locked. |
| `affordances.py` | Good | Honest about its three approximations; errs toward under-counting (conservative for hint triggering). |
| `train/reward.py`, `advantage.py`, `pg_dataset.py` | Good | Deterministic, well-reported; needs invalid-trajectory masking (P0-3) and zero-group drop (P1-11). |
| `train/pg_loss.py`, `train_pg_trl.py` | Needs work | Loss math matches TRL's formula, but the μ=1 framing is misleading at this scale (P0-2); silent truncation (P1-7); per-step `self.log` (P2-16). |
| `train/grpo_loop.py` | Good bones | Telemetry/resume well done; needs guards (P0-4) and stall metrics. |
| `eval_stats.py`, `compare_paired.py` | Good | Correct sign test + bootstrap; add stopped_reason surfacing (P1-10). |
| `risk_proxies.py`, `eval_metrics.py` | Good | Deterministic, conservative labels; OOC-only (Step 5). |
| `glossary.py` | Good | Hand-authored but source-grounded and key-matched to the serializer; the doc discipline around parser/string coupling is exemplary. |
| `interactive/` | Not deep-audited | Well-isolated, unit-tested, canonical-JSONL output; the "sampled candidate commits as user" v1 quirk is documented. |
| Tests | Strong | 368 green in 1.5s; fakes for streaming/GRPO control flow; regression tests exist for every gotcha I cross-checked. |

**Bottom line:** the harness is in excellent shape and the discipline is real. The risk is not
code quality — it is momentum spent on optimizer infrastructure before any arm has produced a
detectable training effect, while the framing experiment (the point of the repo) still lacks its
frame definitions, its CLI plumbing, and its frozen measurement battery. Diagnose the null, get
one positive control (any method that measurably beats base), then pivot hard to Steps 4–5.
