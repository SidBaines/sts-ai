# Competence-First Plan

**Status:** Active  
**Started:** 2026-07-22  
**Last updated:** 2026-07-23  
**Current milestone:** pre-C1 action learning — isolate E4B capacity from static-data coverage  
**Next concrete action:** preregister COMP-019, a matched no-thinking larger-model static teacher comparison on the same public prompts and frozen tiny/full datasets. COMP-018 has rejected fixed action weight 8 at its tiny gate, COMP-015 retains cyclic augmentation only as a possible component, COMP-017 has rejected visible JSON reasoning, and the 12-seed/13-window final Nob cohort remains embargoed

## Role of this document

This is the active operating plan for making the model play Slay the Spire well. It is a running document: update the status dashboard, experiment registry, decisions, and changelog as work lands.

The longer-term scientific goal remains the framing experiment in [`research_plan.md`](research_plan.md). That work is deliberately paused. Until the competence exit gate in this document is met, competence work takes priority and uses one neutral frame. We are not currently trying to measure or induce risk-seeking, risk-aversion, adventurousness, or any neighbouring trait.

This plan should answer four questions without conflating them:

1. Does the policy receive enough public game state to choose well?
2. Can a model of this size represent and generalize a good combat policy?
3. Does the training target contain genuinely better decisions rather than the model's own behaviour?
4. Does the optimizer increase the probability of those decisions and improve held-out play?

If an experiment cannot distinguish at least two plausible answers, redesign it before spending significant GPU time.

## How to maintain this document

- Use the status values `not started`, `in progress`, `blocked`, `complete`, and `rejected`; annotate the single `in progress` item as the next action.
- Keep exactly one item in the status dashboard marked as the next concrete action.
- Give every substantive experiment an ID and add it to the experiment registry before running it.
- Predeclare its primary metric, comparison, sample, and decision gate before looking at the result.
- Append results and decisions; do not silently rewrite an old hypothesis after seeing the data.
- Link the exact config, rollout directories, dataset manifest, adapter, evaluation report, commit, and simulator build fingerprint.
- Treat greedy or tiny-sample evaluations as smoke tests, not evidence of competence.
- When an experiment changes the next-step ordering, update both this document and the priority note in [`research_plan.md`](research_plan.md).

Related evidence and design documents:

- [`gemma_performance_analysis_2026-07-06.md`](gemma_performance_analysis_2026-07-06.md) — encounter-level and transcript-level failure analysis;
- [`nob_rwr_retrain_sampled_eval_2026-07-08.md`](nob_rwr_retrain_sampled_eval_2026-07-08.md) — the powered Nob self-BC null;
- [`repo_review_2026-07-06.md`](repo_review_2026-07-06.md) — training-stack and GRPO review;
- [`reward_spec.md`](reward_spec.md) — reward anti-hacking constraints;
- [`rl_and_framing_design.md`](rl_and_framing_design.md) — historical RL/framing design decisions, some of which this competence plan deliberately supersedes.

## Executive diagnosis

The current evidence does not support “Gemma E4B is simply too weak” as the primary explanation for poor play. E4B has now fit 31/32 deliberately harder first-turn teacher states with the same rank-8/eight-layer LoRA used elsewhere, so basic representation and adapter attachment are positive controls rather than open assumptions. Generalization and closed-loop play remain open.

The working diagnosis, in descending order of confidence, is:

1. **Most historical training supplied little new policy information.** Whole-run RWR mostly cloned self-generated trajectories; the local Nob SFT run cloned only the model's own winning fights. Neither provides a reliable counterfactual label for what to do in states where the model loses. The new 50k-search datasets do provide counterfactual actions and change this diagnosis prospectively.
2. **Dataset state selection and label priors can dominate the learned policy.** COMP-009 trained on every base-visited combat micro-action while evaluating first-turn states. Its 600-row target distribution was dominated by action 0 (337/600), including many late-turn/end-turn decisions; the adapter predicted action 0 on every held-out state. The aligned first-per-turn projection is much less skewed (43/43/44/19/1 over action indices 0–4).
3. **Historical credit assignment was extremely diffuse.** A final-floor or fight-level scalar was broadcast across every decision, and the loss covered hundreds of native-reasoning tokens per decision. The action itself was about 1% of supervised completion tokens in the inspected self-generated datasets. The search-teacher path removes fight-level credit assignment, masks private reasoning, and scores the action directly.
4. **The combat observation withheld a decision-critical human-visible field.**
   The original interface omitted piles/relics/status detail; `combat_public_v1`
   repaired those but still omitted card type. In a stopped live evaluation,
   Gemma explicitly treated Strike/Wild Strike as Skills and Defend as not a
   Skill, reversing Gremlin Nob's Enrage logic. `combat_public_v2` exposes type
   without hidden order. All v1 competence rollouts and teacher datasets are
   rejected for training/evaluation, even though they remain useful diagnostics.
5. **The micro-action interface encourages locally sensible but globally incoherent turns.** Every card play is a fresh model call with no retained turn plan, although v2 now exposes recent-action history.
6. **The hardest losses are encounter-strategy failures, not general inability to operate the game.** Ordinary fights are cheap; Nob, Lagavulin, Sentries, and bosses cause most HP loss.
7. **The present GRPO implementation has not tested proper clipped on-policy learning.** `logp_old` is recomputed from the current model on every optimizer step, making the forward ratio exactly 1 and clipping inactive. Preserved GRPO runs are zero-advantage dry runs, not evidence that RL cannot work.
8. **Model capacity may matter for generalization, but it is not yet the primary blocker.** Existing 0.6B–12B prompt-only sweeps cluster in the same poor range, while E4B can memorize the repaired teacher task. A larger model comparison becomes justified if the aligned full-data run fits train states but fails held-out action or behavioural gates.

COMP-012 now supplies exactly that train/generalization split: 149/150 train but
28/57 development top-1, below its 32/57 gate, with NLL worsening despite six
net additional correct choices. This justifies a later matched capacity
comparison, but COMP-014 now establishes a cheaper pipeline failure to isolate
first: both checkpoints change their semantic policy sharply when the exact
same legal actions are cyclically renumbered. The selected next intervention is
a compute-matched menu-order augmentation control, not immediate model scaling.
The first COMP-015 pair shows that this intervention reduces probability
sensitivity but does not learn equivariance or improve unpermuted actions: at
the fixed step-1,500 endpoint, augmentation moves an index-0 shortcut toward an
index-3 shortcut and reduces development agreement from 21/57 to 17/57. The
already-complete seed-1 pair then reverses that direction: augmentation changes
development direct agreement 24/57→31/57 and cyclic invariance
0.335→0.824, while reducing TV 0.432→0.325 and cyclic source-macro NLL
2.173→1.342. It still regresses development NLL slightly, fits train less
well, misses the 32/57 gate by one action, and remains above the 0.20 TV
boundary. The result is promising but incomplete and demonstrates large
training-seed heterogeneity. Under the sequential-seed rule, this was the
genuinely ambiguous case in which one more treatment earned its cost. The
matched seed-2 treatment improves development direct agreement 28/57→32/57,
cyclic invariance 0.362→0.629, TV 0.473→0.298, and cyclic source-macro
agreement/NLL 0.371/2.073→0.503/1.262. It passes every prospectively frozen
carry threshold, although it worsens train fit 113/150→81/150 and remains
below the standalone behaviour stability boundary. COMP-015 is therefore
closed: its original strict claim failed, no behaviour is permitted, and
cyclic augmentation is retained only as a promising component for a separately
frozen combination after some other intervention proves useful independently.

COMP-016 has now ruled out the leading measurement alternative. A genuine
float32 direct action-digit projection removes all exact unpermuted score ties
but reproduces the same development choices as deterministic JSON generation:
21/57 for control and 17/57 for augmentation. Control still chooses displayed
index 0 on all 57 development states; augmentation chooses index 3 on 37/57.
The direct cyclic result likewise remains 0.000→0.149 invariance and
0.484→0.197 mean probability TV. Fixed-checkpoint teacher forcing finds
format-token NLL at float32 zero or 2.3e-8 while action-token NLL remains
1.293/1.495; effectively all measurable loss is on the move digit even though
the unit-weight objective assigns eight supervised positions to solved JSON
format for every one move position. The shortcut is therefore real, and an
equal-total-mass action-weighting test was warranted as the next cheap gate.

COMP-017 has also answered the no-thinking output question. With native
thinking disabled in both arms, immediate action JSON is harness-valid on
57/57 development rows and agrees with the teacher on 26/57. Asking the base
model to emit a visible reasoning field before the action falls to 55/57 valid
and 19/57 correct, reaches the 512-token cap twice, and is roughly eight times
slower. It fails the frozen static gate, so no reasoning-mode fights are run
and immediate action-only JSON remains the default.

COMP-018 then rejects the equal-mass fixed-weight idea before full training.
Weight 8 produces valid JSON on all 32 harder-tiny states but learns only
7/32 actions, assigns nearly equal probability to every move digit, and stays
in the same state-independent basin from steps 320 to 640. A post-failure
matched unit-weight control under the exact current code reaches 31/32 and
reproduces both historical COMP-011 checkpoints byte-for-byte. Target alignment,
numeric cross-entropy gradients, and a compiled weighted-training toy all pass.
The failure is therefore caused by the aggressive objective/optimization path,
not a dropped action gradient, current trainer regression, or inability of E4B
to memorize this tiny task. Do not run full150, add a seed, or sweep weights.
The endpoint format/action loss split was diagnostically useful, but solved
format loss did not mean format supervision was dispensable early in training.
With the cheap output/order/weighting alternatives now isolated, a matched
larger-model static comparison is the cleanest next test of whether E4B's
remaining full150→development gap is capacity or data coverage.

Current causal status:

| Layer | Established | Still open |
| --- | --- | --- |
| Harness/interface | v1 omitted card type and produced a concrete Enrage inversion; v2 structurally repairs and regression-tests the public boundary. | The behavioural benefit and current v2 base rate have not yet been measured in a matched arm. |
| Teacher/data | Search supplies counterfactual actions; direct-50k hidden-order consensus is substantially more reliable than 5k; dense micro-action selection caused an index-prior collapse. | Whether aligned static data cover the deck/window conditionals needed for fresh play, or DAgger is required. |
| Training | Adapter attachment, mixed masks, and tiny memorization work; late passes sharply overfit confidence; both checkpoints are strongly cyclic-order-sensitive. COMP-015's seed-2 pair passes every frozen carry threshold, corroborating seed 1's development/invariance gains, but augmentation alone remains below the behaviour stability gate and sharply worsens train fit. COMP-016 finds solved format tokens but weak action tokens. COMP-017 rejects visible rationale generation. COMP-018 shows that fixed action weight 8 destroys tiny-task learning while the exact matched unit objective remains 31/32. | Whether a staged/gentler objective is ever useful is deferred; the immediate open question is whether remaining full-data generalization needs more capacity or broader state coverage. |
| Model | E4B can represent the small repaired teacher tasks, including exact current-code 31/32 harder-tiny reproduction. | Whether E4B has a held-out generalization ceiling now requires the matched larger-model static arm; if the larger model also fails, prioritize broader-data/DAgger coverage. |
| RL | Historical runs do not constitute a valid clipped on-policy test. | Whether corrected frozen-old-logp GRPO helps after supervised teacher learning passes C1. |

## Evidence snapshot

These are the facts this plan is designed around. Update them only when a methodologically stronger result supersedes them.

### Full-run behaviour

- In the 100-rollout paired iteration-2 evaluation, the base model won 0/100, cleared the Act I boss in 23/100, and had mean final floor 14.41.
- The RWR+hinted adapter won 0/100, cleared the Act I boss in 22/100, and had mean final floor 13.95.
- Paired floor delta was -0.46, 95% CI [-1.91, +0.99]. This is a null.
- Across the larger E4B thinking baseline, ordinary hallway fights cost roughly 0–3 HP, while Gremlin Nob, Lagavulin, Sentries, and Act I bosses cost roughly 32–55 HP.
- Base transcripts forgo an available immediate lethal often enough to prolong fights and repeatedly fail to allocate energy coherently across a turn.

### Completed training

- Iteration-2 generation produced 300 trajectories, 0 victories, and 62 Act I boss clears.
- RWR assigned multiplicity 1 to 150/300 trajectories, greater than 1 to only 59, and 0 to 91.
- The resampled dataset had 52,927 unique decisions and 122,442 examples.
- Training stopped at 2,000 optimizer steps, only 0.2613 epochs over the resampled dataset.
- Hints corrected 738 of 47,247 combat decisions, about 1.6%, and covered only immediate lethal/full-block mistakes.
- In a 5,000-example tokenization sample, iteration-2 completions averaged 889 tokens; a minimal action-only JSON was about 1.1% of completion tokens.
- The Nob dataset contained 18 winning train windows, 343 unique decisions, and 461 resampled examples. Better validation loss did not improve held-out behaviour.
- Sampled Nob evaluation at temperature 0.7, K=4, gave base 24/44 wins versus adapter 23/44 and paired reward delta +0.003, 95% CI [-0.16, +0.18]. This is a method null for self-BC on the model's own wins.

### Search-teacher learning controls completed on 2026-07-23

- **COMP-008, easy tiny32:** the mixed required-format-plus-action mask moved train agreement from 18/32 to 32/32 and mean teacher-candidate probability from 0.568 to 0.981. This proves the current model/trainer/adapter path can memorize a small repaired-interface teacher set. Its post-gate held-out diagnostic improved top-1 from 22/57 to 29/57 but worsened teacher NLL from 1.777 to 2.898, so it was not accepted as generalization.
- **COMP-009, dense every-micro-action 600:** three passes yielded 21/57 held-out agreement versus 22/57 for base. The adapter predicted action 0 on all 57 held-out rows and exactly the action-0 count on the old tiny32. This is a majority-prior/data-selection collapse, not a model-capacity result.
- **COMP-010, first-turn action-value-only tiny32:** ten passes improved agreement only from 8/32 to 11/32 and collapsed canonical completion log-probability from -16.5 to -112.1. Its preregistered gate failed, so full-data training and behaviour were not run.
- **COMP-011, matched 20-pass mask comparison:** with identical harder tiny32 data, model, LoRA, optimizer, and update count, the mixed mask reached 31/32 (mean teacher probability 0.821) while action-value-only reached 15/32 (0.419). Required format loss is therefore a useful optimization/output-contract scaffold in this setup; the result is one dataset/seed and does not rule out explicit action reweighting.
- **COMP-012, aligned full150:** the final mixed-mask adapter fit 149/150 train states, but only improved development top-1 from 22/57 to 28/57, below the frozen 32/57 gate. Mean teacher probability improved 0.372→0.479 while NLL worsened 1.777→2.572. Behaviour was correctly not run. Errors cluster by unseen window and target menu position: all eight index-3 targets fail, even though all 19 train index-3 targets fit.
- **COMP-013/014, checkpoint and action-order diagnosis:** step 1,500 has much
  better unpermuted calibration than step 3,000 but still misses the development
  gate. Under all 221 nonzero cyclic menu rotations, semantic top-1 invariance
  is only 0.448 at step 1,500 and 0.253 at step 3,000; mean probability TV is
  0.400 and 0.696. Tie-aware top-set invariance is similarly poor. Across all
  variants, neither checkpoint gets any teacher target assigned to indices 4–7
  correct. Action ordering is therefore a demonstrated pipeline failure, while
  augmentation benefit and residual state coverage remain open.
- **COMP-015 seed-0 fixed pair:** both exact 3,000-update arms completed and the
  immutable step-1,500 pair was scored on train150, development57, and all 221
  nonzero development rotations. Augmentation improved five of six frozen
  directions: semantic top-1 invariance 0.000→0.149, probability TV
  0.487→0.202, all-variant source-macro agreement 0.214→0.342, source-macro
  NLL 4.076→1.501, and unpermuted development NLL 1.584→1.484. It failed the
  required development non-regression, however: agreement fell 21/57→17/57;
  train agreement also fell 42/150→39/150. It therefore does not qualify for
  behaviour and makes the strict all-three-pair primary success rule
  unattainable. Assigned-position accuracy shows control at 36/36 for position
  0 and zero elsewhere, while treatment is 43/46 at position 3 and poor
  elsewhere: augmentation mainly redistributes the positional prior. The cyclic
  report is `data/competence/comp_015/eval/seed0/cyclic_step1500_control_augmented.json`
  (SHA `2086ad36…ae9c`). Candidate scores are heavily 1/8-quantized and ties are
  common, but tie-aware top-set invariance is only 0.063 and does not reverse
  the conclusion. The mixed objective exposes 24,000 required-format versus
  3,000 action tokens per arm, so exact float32 action-token scoring and
  format/action loss decomposition are required before the next intervention.
- **COMP-016, direct-choice and loss audit:** the frozen measurement config
  `900bf5ee…be38` scored the exact seed-0 step-1,500 pair without querying new
  states. Genuine float32 full-vocabulary action-digit scoring has zero exact
  ties and matches deterministic JSON on development exactly: control 21/57,
  augmentation 17/57, with 100% canonical/legal JSON. It preserves the cyclic
  diagnosis: semantic invariance 0.000→0.149 and mean probability TV
  0.484→0.197. Control predicts displayed index 0 on all 57 states;
  augmentation predicts index 3 on 37/57. Teacher-forced scheduled-row loss
  is almost entirely the action digit: format/action mean token NLL is
  0.0/1.293 for control and 2.3e-8/1.495 for augmentation across
  24,000/3,000 tokens. Thus complete-JSON quantization was a measurement defect
  but not the cause of the policy failure. Result config:
  `configs/competence/comp_016_result.json` (SHA `7fbae99e…9df9`).

### Decisive interface diagnostic discovered on 2026-07-23

- COMP-002 was stopped after 71 recorded decisions, before either arm completed,
  because continuing would spend model compute on a known-invalid information
  boundary.
- The prompt defined Enrage correctly but did not expose card type. Gemma then
  reasoned that Bash and Strike triggered Enrage, that the Strength gain was
  related to attack damage or energy cost, and that Defend did not trigger it.
  It consequently defended/passively stalled against Nob for exactly the wrong
  strategic reason.
- This is direct evidence against attributing the observed failure to model
  capacity alone. The interface asked the model to apply a typed mechanic while
  withholding the type.
- The partial rollouts are preserved under
  `data/competence/comp_002/eval/legacy_v1_invalid_missing_card_type/`; they have
  no completed meta sidecars and must never be resumed or compared as an arm.

### Positive control discovered on 2026-07-22

The built-in search agent was replayed from all 43 saved Gremlin Nob starts in `data/local_curricula/gremlin_nob/manifests/source.json`:

| Search budget | Wins | Mean HP loss | Median HP loss | Mean existing task reward |
| --- | ---: | ---: | ---: | ---: |
| 5,000 simulations | 43/43 | 16.12 | 13 | 0.617 |
| 50,000 simulations | 43/43 | 13.00 | 10 | 0.692 |

This establishes that the saved task starts are solvable and that the simulator contains a strong dense combat-teaching signal. It does not yet establish that individual search actions are safe student targets: the searcher can exploit internal RNG and ordered state unavailable to a human-facing policy. Teacher observability must be audited before distillation.

## Scope and non-goals

### In scope

- Ascension 0, Ironclad, Acts I–III.
- Full LLM combat control as the eventual policy target.
- Fight-local curricula as the fastest way to develop and validate learning machinery.
- Public-state serialization, turn-level planning support, action-only supervision, search distillation, DAgger, and correctly implemented local policy-gradient training.
- Neutral reward and prompts. Neutrality is useful engineering discipline even while framing is paused.
- Model-size comparison after the learning setup has a positive control.

### Out of scope until the competence gate

- Framing variants or risk-trait evaluation.
- Ascension climbing.
- Other characters.
- Vision or UI control.
- Human-level route/deck optimization as the first success criterion.
- Full-game RL before local hard-fight learning demonstrably works.
- Treating more self-generated SFT epochs as a substitute for new target information.

## Competence ladder and exit gates

“Play well” is a ladder, not a single binary endpoint. Each rung should be passed on held-out data before the next becomes the primary focus.

Thresholds below are provisional but must be locked in an experiment entry before its results are inspected. If a threshold changes, record why and apply it only prospectively.

### Gate C0 — measurement and teacher readiness

Pass when all are true:

- A fresh baseline exists for the current serializer/prompt on held-out Nob windows using sampled paired evaluation.
- Public combat observation fields are enumerated and tested against an explicit human-observable-state contract.
- A search action can be queried without mutating the live `BattleContext`, with legal-action identity and provenance recorded.
- Search quality and first-action stability are measured at multiple simulation budgets.

### Gate C1 — one hard fight is learnable

Pass on Gremlin Nob when all are true:

- A teacher-labelled, action-only adapter improves held-out paired task reward with the lower bound of the 95% bootstrap CI above 0.
- The mean reward gain is at least 0.10 on the existing task scale, or the adapter improves win rate by at least 10 percentage points with a corroborating HP-loss improvement.
- Invalid-action and truncation rates do not worsen materially.
- Teacher-action agreement improves on held-out states, not merely on training states.
- The result survives at least K=8 samples per held-out window or an equivalently powered preregistered design.

### Gate C2 — hard-fight competence generalizes

Pass when the same pipeline produces positive held-out effects on at least two of:

- Gremlin Nob;
- Lagavulin;
- Sentries;
- an Act I boss.

The combined adapter must retain previously acquired fight performance. A new-task gain that erases an earlier task is curriculum failure, not progress.

### Gate C3 — Act I competence improves

On a frozen, paired full-policy evaluation:

- Act I boss-clear rate improves by at least 10 percentage points over the current-interface base, and
- paired mean-floor improvement has a 95% CI whose lower bound is above 0, and
- hard-fight HP-loss metrics move in the expected direction, and
- invalid, stalled, and budget-truncated rates remain within the preregistered guardrails.

### Gate C4 — full-game competence is real

Pass when:

- the policy records at least 5 wins in a preregistered 100-seed evaluation or an equivalent powered test;
- Act II reach, Act III reach, and win rate all improve rather than wins arising from a narrow regression elsewhere;
- the gain is reproduced from a clean adapter/model load;
- the exact model, prompt, observation schema, simulator patch, and evaluation split are frozen as the competence checkpoint.

Meeting C4 unblocks the framing study. We may choose to continue toward stronger play, but framing variants should fork from a fixed competence checkpoint rather than from a moving training stack.

## Design principles

### 1. Positive control before scale

Every training method must first solve a small problem where a demonstrably better target exists. Do not infer anything from a whole-game null until the same code can overfit a tiny teacher-labelled set, improve held-out action likelihood, and change local-task behaviour.

### 2. Separate policy information from optimization

Measure these separately:

- Did the dataset contain a different preferred action from the base policy?
- Did training increase its held-out log-probability?
- Did greedy or sampled action choice change?
- Did the changed action improve fight outcome?

This turns “training failed” into a localized result.

### 3. Match comparisons at the world state

Use the same world seed and, for local tasks, the same replayed start. Aggregate K stochastic samples to a per-seed or per-window mean before paired inference. Global cross-seed reward weighting is not a substitute for matched comparisons.

### 4. Train and measure the policy output directly

The first competence target is the compact, canonical
`{"action_index": i}` assistant turn. Private reasoning receives no loss and
must not dominate the policy objective. The action value is always evaluated
directly with candidate-normalized probability, NLL, and top-1 agreement.

Do not conflate three separate controls:

- `output_contract="action_only"` means the compact JSON target has no rationale;
- manifest `loss_mask_mode="action"` is the current mixed mask over required
  channel/JSON/terminator formatting plus the action value;
- `--mlx-action-token-only` is the narrower experimental override that gives
  loss only to the action-category token while retaining the entire completion
  as causal context.

COMP-011 found that the mixed mask fit the harder tiny32 while the pure
action-value mask did not under the same 20-pass schedule. Mixed
format-plus-action supervision is therefore the current default. Explicit
action upweighting, native-thinking inference, short teacher rationales, and
full-thought supervision are later comparisons; none should silently redefine
the frozen target or mask.

### 5. Preserve public-state parity

The student observation should contain everything a human player can inspect, but not hidden shuffle order, future RNG, or simulator internals. Teacher labels must record whether they used privileged state. When search actions are unstable under hidden-state perturbations, either omit those labels, use consensus labels, or train an observation-conditioned teacher.

### 6. Local to global

Develop data, loss, reward, and evaluation machinery on short hard fights. Expand to whole acts only after local causal evidence exists. This reduces rollout cost and makes reward attribution defensible.

### 7. Keep strong baselines in every stage

At minimum compare:

- frozen base model;
- current best competence adapter;
- search teacher or search-resolved combat upper bound where applicable;
- a simple deterministic policy where it helps interpret the metric.

### 8. Freeze interfaces before expensive evaluation

Changes to observation text, glossary, action descriptions, chat template, reasoning mode, batch size, or simulator patch define a new policy interface. Give the interface a version/hash and do not mix old and new trajectories in a claim without an explicit compatibility argument.

## Workstream A — measurement and baseline reset

**Status:** interface/provenance work complete; the matched `combat_public_v2` behavioural baseline required by Gate C0 is still outstanding  
**Purpose:** ensure subsequent improvements are measured against the current harness rather than pre-fix data.

### A1. Version the competence interface

**Implemented 2026-07-23.** Rollout metadata now records the readable
`competence_interface_version` and byte-level fingerprints for the prompt probe,
combat serializer/glossary/prompting sources, simulator patch and built binary,
and tokenizer chat-template probe. The fingerprint is taken from actual working
tree bytes, so a dirty but experimentally used interface is not mislabeled by
the last Git commit alone.

Record in every run:

- schema version;
- prompt/framing hash;
- public-observation serializer hash or explicit version;
- glossary version/hash;
- chat-template probe hash;
- reasoning mode and output-token cap;
- model and adapter identifiers;
- simulator patch commit/build fingerprint;
- batch size and orchestration path.

Add an explicit `competence_interface_version` once observation changes begin. A hash alone catches skew; a readable version explains it.

### A2. Rebaseline the current serializer cheaply

**Outstanding for `combat_public_v2`.** The sampled `base_vllm_t07_k4`
evaluation at commit `8fc92b6` is a useful historical pre-v2 baseline:
temperature 0.7, K=4 over 11 windows, 24/44 wins, mean reward -0.252, and
mean HP loss 43.1. It predates the decision-critical card-type repair exposed
by COMP-002 and must not be called the current-interface baseline.

Before making a behavioural training claim, run a matched v2 base arm on
non-final development windows with the exact model, prompt, tokenizer,
reasoning mode, sampling parameters, batch size, simulator build, and K that
will be used for the adapter. The final Nob cohort stays embargoed until a
development gate and its full comparison are preregistered. K=8 remains the
minimum for C1 unless a prospective power calculation selects a stronger
design.

Report:

- task reward and paired CI;
- win rate and convincing-win rate;
- HP loss;
- turns;
- attack/skill/block shares;
- Skills played while Enrage is active;
- lethal and full-block misses;
- invalid/truncated completions;
- completion length and latency.

Do not run an expensive 100-seed full-game baseline until the observation/interface work in Workstream B is stable. The current Nob rerun is enough to measure whether the serializer fix removed the known local failure.

### A3. Add hard-fight local-task coverage

**Partially complete 2026-07-23.** Lagavulin and Sentries are now first-class
local tasks with deterministic seed splits, fixed-start signatures, bounded
rewards, source/loadout audit data, and encounter-specific metrics. Isolated
v2/schema-2 replay validation passes 43/43 Lagavulin windows (33 train, 10
holdout) and all 36 Sentries windows (27 train, 9 holdout). A second complete
isolated run produced identical public signatures. Earlier v1-build exclusions
remain recorded historically; this is a same-binary result, not cross-build
reproducibility. Act I boss curricula remain to be added.

Create generic or encounter-specific replay manifests for:

1. Lagavulin;
2. Sentries;
3. the most common Act I boss in the frozen sample, then the remaining bosses.

Split by world seed, not by decision. Record entry deck, relics, potions, HP, encounter, and source policy so train/holdout difficulty can be audited.

### A4. Define an evaluation battery

All local-task reports should share a common core plus encounter-specific metrics. Full-run reports should aggregate the local metrics by encounter.

Primary local metrics:

- survival/win;
- entry-to-exit HP loss;
- bounded task reward;
- turns to resolution.

Policy metrics:

- teacher-action top-1 agreement;
- probability assigned to teacher action;
- action entropy;
- action changes versus base on matched states.

Tactical metrics:

- available lethal taken;
- full block used when appropriate;
- energy wasted at end of turn;
- premature end turn;
- potion use and potion waste;
- encounter-specific mistakes such as Skills into active Enrage.

Health metrics:

- invalid JSON/action;
- retries;
- truncated thinking;
- simulator errors and UB flag;
- decision-budget truncation;
- prompt/completion tokens and latency.

## Workstream B — public observation and action interface

**Status:** complete for the C1 public-state boundary; persistent turn planning remains the deferred COMP-003 intervention  
**Purpose:** remove avoidable partial observability and test whether coherent turn planning helps without changing the underlying action space.

### B1. Write and test the public-state contract

**Complete 2026-07-23.** [`combat_public_state_contract.md`](combat_public_state_contract.md)
defines the current `combat_public_v2` contract and explicitly preserves v1 as
a replay-only compatibility interface; the Python/C++ interface and integration tests now
cover the fields below while keeping hidden pile order out of the text and
structured public summary. The legacy serializer remains available as an
explicit matched-evaluation control.

Combat observation should expose, in a compact stable representation:

- player HP, block, energy, max energy, and public powers;
- enemy HP, block, intent, public powers/statuses, and sim-computed visible damage;
- current hand with cost-for-turn, upgrade state, and public card effects;
- draw-pile multiset, not hidden order;
- discard and exhaust contents;
- relics and public counters;
- potions and slots;
- turn number and public per-turn counters;
- recent actions in the current turn;
- any pending card-selection context.

Explicitly exclude:

- hidden draw order;
- future random outcomes;
- enemy RNG not shown by the game;
- search values or teacher recommendations.

Add integration snapshots for representative fights and cards. A serializer change that silently removes one of these fields should fail a test.

### B2. Keep the representation compact

**Initial implementation complete.** Card definitions are deduplicated through
the glossary, unordered pile contents are aggregated and sorted, and token
distribution reporting is available. COMP-002 was stopped on the invalid v1
information boundary, so the first matched v2 cost and invalid-rate check
remains part of the current-interface behavioural baseline.

Full observability does not require a prose dump. Prefer stable structured lines and deduplicated card summaries. Show a glossary entry only when its mechanic is active or present. Measure prompt tokens at p50/p90/p99 and set a budget before collection.

### B3. Test turn-plan persistence

Keep simulator execution at one legal micro-action per step for the first experiment. Compare:

- **Control:** fresh state-only prompt at every micro-action.
- **Plan-memory arm:** at the first decision of a turn, ask for a short intended sequence/priorities; include that plan and the actions already taken on subsequent decisions in the same turn.

The model must be allowed to revise the plan when state changes. The plan is advisory, not an executable action list.

Primary readouts:

- hard-fight reward;
- energy wasted;
- attack-before-block sequencing errors;
- completion tokens and latency.

Only consider a whole-turn macro action if plan persistence fails. Macro actions create combinatorial legality, card-selection, random-draw, and replanning problems and should not be the first intervention.

### B4. Simplify the output contract

**Training-mask portion complete 2026-07-23.** Tokenizer-aware, non-contiguous
action loss masks now preserve required assistant/channel/turn markers while
masking private thought and other completion text. Dataset manifests report
prompt, thought, formatting, and action token counts; malformed or mismatched
targets fail closed. The optional inference-schema simplification remains a
separate intervention and is not mixed into COMP-002.

The policy-facing answer should be the smallest robust action representation, ideally:

```json
{"action_index": 0}
```

Do not require a `reasoning` string inside the final JSON when native thinking already provides a separate channel. Ensure the parser and retry path remain fail-closed.

Compact action-target training must still preserve any syntactically required chat-template channel markers and assistant-turn terminator. The loss mask should distinguish those formatting tokens, optional private-thought content, and the policy-bearing action span. Record all three token counts so the realized objective is auditable.

## Workstream C — search teacher and compact-action learning

**Status:** in progress — static E4B interventions exhausted; matched capacity diagnosis is next  
**Purpose:** inject genuinely better actions and establish the first causal training positive control.

### C1. Bind a non-mutating search query

**Complete 2026-07-23.** The query operates on a cloned battle context, returns
the selected legal action plus root visits/value statistics, search and draw
seeds, predicted outcome/HP and provenance, and has parity/non-mutation tests.

Add a binding conceptually equivalent to:

```text
search_best_action(battle_context, simulations, seed/config) -> teacher decision
```

The result should include:

- selected action bits and its matching displayed legal-action index;
- simulation budget and search seed;
- root action visit counts;
- root action value estimates where meaningful;
- whether a winning sequence was found;
- predicted terminal player HP from the best found sequence;
- action-description and state hashes;
- a teacher-confidence/stability field derived downstream.

The query must clone the `BattleContext` and leave the live environment unchanged. Test parity against `legal_actions()` and replay the selected action through the normal Python step path.

### C2. Audit teacher quality and privileged information

**Corrected audit complete as COMP-004.** The collector replays base-policy states, hashes
only public observations plus displayed actions, queries multiple search budgets
and seeds, and perturbs hidden draw order only on cloned teacher states.

For saved hard-fight states:

- compare 1k, 5k, and 50k simulation budgets;
- measure first-action agreement across budgets and repeated search seeds;
- compare eventual HP/outcome when following the search policy;
- identify states where the chosen action changes under plausible hidden shuffle/RNG perturbations;
- mark low-consensus states rather than pretending the teacher label is exact.

Use 5k as the provisional collection budget only if it agrees sufficiently with 50k on held-out states. Use the larger budget for evaluation and ambiguous labels.

### C3. Build a teacher-labelled dataset

**First evidence collection complete.** Raw label and
action-only dataset builders implement public-observation deduplication,
confidence/hidden-consensus filters, seed-split manifests, provenance, and token
accounting. A one-state smoke artifact is explicitly excluded from evidence.

Collect from states visited by the base model, not only from search trajectories. The first dataset should focus on hard fights and include losing regions of the base policy's state distribution.

Required dataset properties:

- split by source world seed/window;
- one stable state hash per public observation;
- legal actions and teacher action identity;
- search provenance/confidence;
- base action and, where available, base action probability;
- encounter/turn/HP metadata;
- no native thought in the supervised target;
- deduplication and explicit class/encounter balancing;
- manifest counts before and after confidence filtering.

### C4. Validate the trainer as a positive control

**Memorization positive control complete; held-out action and behaviour gates
pending.** A tiny two-step MLX LoRA run verified masked training and adapter
reload. COMP-008 and COMP-011 then fit 32/32 easy and 31/32 harder teacher
actions with the no-thinking, compact JSON target. This establishes attachment,
masking, and small-set capacity, not held-out competence. A later
native-thinking arm must preserve a genuine thought/channel prefix in causal
context rather than teaching the model to skip reasoning accidentally.

Run these tests in order:

1. Overfit 32–128 examples and reach near-perfect training action accuracy.
2. Improve teacher-action log-probability on a held-out batch.
3. Improve held-out top-1 teacher agreement.
4. Change sampled decisions in the intended direction.
5. Improve local-task outcome.

Stop at the first failed link and diagnose it. Do not compensate for a failed overfit or likelihood test by generating a larger dataset.

### C5. DAgger loop

Once static teacher SFT works:

1. Roll out the current student on train windows.
2. Query search on the states it actually visits.
3. Add confident teacher labels, with extra weight on student/teacher disagreements.
4. Retrain from the competence base or previous adapter according to a preregistered choice.
5. Evaluate on untouched holdout windows.

Track dataset growth, disagreement rate, and whether later iterations reach novel states. Stop when disagreement and behavioural gains plateau, not at an arbitrary number of rounds.

## Workstream D — multi-encounter curriculum

**Status:** not started; replay-validated cohorts exist, but teacher/training work waits on C1  
**Purpose:** demonstrate that improvements generalize and can coexist in one policy.

Recommended order:

1. Gremlin Nob — attack/defence tradeoff and Enrage.
2. Lagavulin — setup timing and wake decision.
3. Sentries — target priority, damage race, and Dazed management.
4. Act I bosses — longer encounter plans and deck-dependent strategies.

For each encounter:

- establish base and search ceilings;
- collect base-visited teacher labels;
- pass the static SFT likelihood/action test;
- run at least one DAgger iteration;
- evaluate the combined adapter on all previous holdouts.

Use replay mixing or a balanced multi-task dataset to prevent forgetting. Keep per-encounter metrics; a combined average can hide a severe regression.

## Workstream E — corrected local policy-gradient training

**Status:** not started  
**Dependency:** C1 must pass first.  
**Purpose:** test whether sampled outcome feedback adds value beyond dense teacher imitation.

### E1. Fix the objective before a real run

Required changes:

- snapshot behaviour-policy token log-probabilities once per generation iteration, or restrict each generated batch to one truly on-policy optimizer update;
- compute the ratio against that frozen behaviour policy so clipping is meaningful;
- mask loss to the action span for the first experiment;
- exclude or explicitly handle `agent_invalid` and simulator-error trajectories;
- drop zero-variance groups from the policy-gradient dataset;
- report clipped fraction, ratio, KL, entropy, action-token count, and effective optimizer updates;
- fail fast on zero-advantage, empty, heavily truncated, or high-invalid iterations;
- cap reuse of stale generated data.

The existing implementation may be retained as an explicitly named advantage-weighted-BC baseline, but it should not be called clipped GRPO while its ratio is recomputed as 1.

### E2. Shape local rewards without letting losses tie

Nob's current reward maps every loss to -1. That removes all group variance on the windows where competence is weakest.

Design a bounded reward with this ordering:

1. Every win is better than every loss.
2. Among wins, preserving more HP is better.
3. Among losses, reducing more enemy HP and preserving more player HP is better.
4. Stalling, format failure, or exceeding the decision budget cannot improve reward.

Choose coefficients from the observed metric ranges before training and add anti-hacking tests for boundary cases. Do not tune coefficients on the holdout outcome.

### E3. Local GRPO experiment

- Group K>=8 rollouts from the identical start.
- Sample at a fixed nonzero temperature.
- Normalize within start/window, never across unrelated world states.
- Train only on groups with usable variance.
- Evaluate against both base and teacher-SFT adapters.
- Use a three-arm comparison: base, teacher SFT, teacher SFT + RL.

The question is not merely whether RL beats base. It is whether RL adds value after a strong supervised policy and whether that value transfers to held-out starts.

## Workstream F — whole-act and full-game training

**Status:** not started  
**Dependency:** C2 and corrected RL health checks.  
**Purpose:** combine tactical competence with long-horizon deck, route, reward, shop, event, and campfire decisions.

### F1. Isolate combat from out-of-combat play

Maintain both evaluations:

- model out-of-combat decisions with search-resolved combat;
- the same model controlling both out-of-combat and combat decisions.

Their gap estimates the cost of learned combat independently of pathing/deck choices. Also report search-resolved combat as an upper bound, not as the final target.

### F2. Use segment-level credit

Do not immediately broadcast final floor across every action in a 500-decision run.

Candidate hierarchy:

- combat segment: victory/loss, HP delta, potion use, encounter progress;
- floor segment: reward choice and subsequent short-horizon consequence;
- act segment: boss clear and remaining HP/resources;
- run: act reached and victory.

Prefer return-to-go or segment advantage so an early hallway card play does not receive the same undifferentiated scalar as the later decision that loses to a boss.

### F3. Compare within world seed

Generate K trajectories per world seed and compute group-relative outcomes within the seed. Record map/encounter variation caused by earlier policy choices; if trajectories diverge so far that direct comparison becomes noisy, retain intermediate segment rewards and report the divergence.

### F4. Preserve local competencies

Every whole-game checkpoint must run the hard-fight battery. Full-run floor can improve by avoiding elites while combat skill regresses; that is not the desired competence gain.

## Workstream G — model-capacity decision

**Status:** in progress — **current next action is the COMP-019 matched larger-model preregistration**  
**Dependency:** action-only teacher pipeline passes its basic positive controls.  
**Purpose:** decide whether E4B is an appropriate student using evidence that isolates capacity.

Compare E4B with at least one materially larger model on:

- prompt-only teacher-action agreement on the same public states;
- ability to overfit the same small action dataset;
- held-out teacher-action likelihood and top-1 agreement after matched LoRA training;
- local-task outcome after matched data and evaluation;
- inference/training cost and invalid rate.

Decision rules:

- Keep E4B if it learns the teacher policy and achieves local behavioural gates at materially lower cost.
- Upgrade if E4B fits training states but has a consistent held-out generalization gap that produces worse play, while the larger model passes under the same interface and data.
- Fix the pipeline, not the model, if neither model can overfit or neither receives adequate labels.
- Use a stronger model as a teacher or evaluator even if it is too expensive as the final policy.

Do not use the old model sweep alone to choose: it mixed prompt interfaces, reasoning behaviour, serializer versions, and small samples, and all arms lacked dense competence training.

## Experiment registry

Historical entries establish context. New experiments should be appended with links to their preregistration/config and result.

| ID | Status | Hypothesis / question | Primary result | Decision |
| --- | --- | --- | --- | --- |
| HIST-001 | complete | Does prompt-only model scaling from roughly 0.6B to 12B solve the current game interface? | All sampled arms had 0 wins and mean floors roughly 10–13; scaling effect was small. | Capacity remains open; do not switch models on this evidence alone. |
| HIST-002 | complete | Does whole-run RWR plus sparse tactical hints improve E4B? | Paired floor delta -0.46, 95% CI [-1.91, +0.99]. | Null; stop treating self-RWR as an adequate capability source. |
| HIST-003 | complete | Does longer filtered BC on the model's own winning Nob fights improve held-out Nob play? | Reward delta +0.003, 95% CI [-0.16, +0.18]. | Null; the next method must add counterfactual action information. |
| DIAG-001 | complete | Are the saved Nob starts solvable by a policy already present in the simulator? | Search won 43/43 at both 5k and 50k simulations. | Use search as the first dense-teacher candidate; audit privilege/stability. |
| COMP-001 | historical pre-v2 baseline | Establish the post-Enrage/damage-fix base policy on held-out Nob starts. | At commit `8fc92b6`, temperature 0.7, K=4: 24/44 wins, reward -0.252, HP loss 43.1; it predates the card-type-complete v2 interface. | Do not use as the current-interface control; run a matched v2 base arm before a behavioural claim. |
| COMP-002 | rejected | Does v1 public combat state improve prompt-only Nob play? | Stopped after 71 decisions: the model explicitly inverted Enrage because combat card type was absent. No arm completed. | Reject both the partial evaluation and the v1 information boundary; replace with v2 only after cohort revalidation. |
| COMP-003 | not started | Does persistent turn planning improve sequencing beyond state-only micro-actions? | Pending. | Run after B1/B3. |
| COMP-004 | complete | Are 5k-search first actions stable enough to label model-visited states? | Independent 1/2/3 audit: 5k-vs-50k agreement 43/56 (76.8%); 5k unanimity 33/56 (58.9%); 31/56 qualify for 5k collection while 38/56 qualify for direct 50k labels. | Reject 5k; collect independent 50k labels and keep only rows whose hidden-order consensus selects the same action. |
| COMP-004A | complete | Is the searcher's best observed winning-sequence action or its displayed-action-aggregated most-visited root a better online teacher policy? | Aggregated root visits completed/won 7/10 with 3 ambiguities and mean all-start reward 0.1025; winning sequence completed/won 4/10 with 6 ambiguities and reward -0.315. | Use displayed-action-aggregated root visits in recollected v2 labels. |
| COMP-005 | rejected | Can action-only v1 teacher SFT improve held-out agreement and Nob outcome? | 493/652 rows passed the search filter, but every prompt omitted card type. No model was loaded or scored on the dataset. | Preserve under `comp_005_v1_invalid_missing_card_type`; recollect v2 before freezing a new tiny subset. |
| COMP-006 | not started | Does DAgger outperform static teacher SFT on base-visited Nob states? | Pending. | Determines whether iterative data collection is needed. |
| COMP-007 | not started | Does corrected local GRPO add value beyond teacher SFT? | Pending. | Determines role of RL in the competence stack. |
| COMP-008 | complete | Does a versioned card-type-complete public interface preserve hidden-state safety, and can E4B fit its teacher actions? | Cohorts replay 43/43 Nob, 43/43 Lagavulin, 36/36 Sentries; direct 50k accepts 57/60 held-out and 600/652 train states. The easy tiny32 moved 18/32→32/32, but its held-out NLL worsened. | Interface/fit positive control passed; scale only under a separately frozen full-data gate. |
| COMP-009 | complete — failed gate | Does three-pass mixed-mask SFT on all 600 eligible base-visited micro-actions generalize to first-turn holdout states? | 21/57 held-out versus base 22/57; the adapter predicted action 0 on all 57. Train targets were 337/148/85/28/2 by action index. | Stop before behaviour; align state selection and isolate mask/schedule effects. |
| COMP-010 | complete — failed gate | Does first-per-turn alignment plus ten passes of action-value-only loss avoid the collapse? | Harder tiny32 improved 8/32→11/32, below the 31/32 gate; canonical sequence log-probability collapsed to -112.1. | Do not run full150; compare masks at the same 20-pass schedule. |
| COMP-011 | complete | Which mask fits the aligned harder tiny32 under identical 20-pass training? | Mixed format+action: 31/32, mean teacher p 0.821. Action-value-only: 15/32, p 0.419. | Keep the mixed mask; format tokens are useful scaffolding in this setup. |
| COMP-012 | complete — failed gate | Does 20-pass mixed-mask SFT on all 150 aligned first-per-turn states fit train and improve development teacher actions? | Train 149/150; development 28/57 versus base 22/57, probability 0.372→0.479, but NLL 1.777→2.572 and required top-1 was 32/57. | Do not run behaviour. Diagnose checkpoint overconfidence and action-order/coverage shortcuts; the 57 rows are development data henceforth. |
| COMP-013 | complete | Did late training cause COMP-012's overconfidence, and is early stopping sufficient? | Step 1,500: train 102/150, development 27/57, NLL 1.183. Step 3,000: 149/150, 28/57, NLL 2.572. | Late training overfits/sharpens confidence, but the earlier checkpoint is not sufficient. Run action-order sensitivity separately as COMP-014. |
| COMP-014 | complete | Are step-1,500/3,000 predictions semantically stable under every non-zero cyclic rotation of the same legal actions? | Both are strongly sensitive: semantic top-1 invariance 0.448/0.253 and mean probability TV 0.400/0.696 at steps 1,500/3,000. Tie-aware invariance agrees. | Follow the frozen sensitive branch: test permutation augmentation alone under a compute-matched control. |
| COMP-015 | complete — strict claim failed; component retained | Does cyclic menu-order augmentation improve semantic invariance and unpermuted development actions independent of compute/exposure? | Seed 0 regresses development 21/57→17/57. Seeds 1 and 2 improve it 24/57→31/57 and 28/57→32/57 while improving cyclic metrics. Seed 2 passes all frozen carry thresholds, but invariance 0.629 and TV 0.298 remain below the standalone behaviour gate and train fit falls 113/150→81/150. Result `configs/competence/comp_015_result.json`. | Close without behaviour or final release. Retain balanced cyclic augmentation only as a possible component after some other intervention independently passes. |
| COMP-016 | complete | Are COMP-015's shortcut and 8:1 supervision concern real at the move digit, or artefacts of quantized complete-JSON scoring? | Tie-free float32 direct scoring exactly reproduces development greedy choices (21/57 control, 17/57 augmented) and cyclic failure. Format/action token NLL is 0.0/1.293 and 2.3e-8/1.495. Frozen config/result `900bf5ee…be38` / `7fbae99e…9df9`. | Measurement artefact rejected; test one weight-8 action treatment behind the harder-tiny32 gate after COMP-015 closes. |
| COMP-017 | complete — failed gate | Does brief visible JSON reasoning improve teacher actions when native thinking remains disabled? | Immediate action-only is valid/correct on 57/57 and 26/57; visible reasoning is 55/57 and 19/57, hits the 512-token cap twice, and takes 8.37s versus 1.04s per answer. Frozen config/result `1ebb217a…f124` / `0934aefd…fdbd`. | Do not run fights. Retain no-native-thinking immediate action JSON as the default. |
| COMP-018 | complete — failed tiny gate | Does move-choice weight 8 preserve the useful format scaffold while improving action fit and development choices? | Weight 8 stays at 7/32 and near-uniform move probabilities from steps 320→640, despite reaching valid/legal JSON 32/32. The exact current unit-weight control reaches 31/32 and reproduces historical checkpoints byte-for-byte; gradient/alignment audits pass. Config/result `5512cc8c…4d54` / `configs/competence/comp_018_result.json`. | Reject fixed weight 8; do not run full150, add a seed, or sweep weights. Preserve unit mixed supervision. |
| COMP-019 | not started — **next action** | Is E4B's full150→development gap a model-capacity ceiling or a data-coverage problem? | Pending matched no-thinking larger-model static comparison on the same public prompts, tiny/full datasets, masks, and action metrics. | Preregister model/revision, comparable adapter budget, tiny fit gate, full-data schedule, and E4B-vs-larger decision rule before any model load. |

## Standard experiment record

Copy this block into an experiment-specific document or append a compact version beneath the registry.

```markdown
### COMP-XXX — title

**Status:** preregistered | running | complete | rejected
**Date:**
**Commit / simulator build:**

Hypothesis:

Intervention:

Control arms:

Train data and split:

Evaluation sample and sampling parameters:

Primary metric:

Secondary metrics and health guardrails:

Decision gate fixed before run:

Artifacts:

Result:

Decision and next action:
```

### COMP-002 — human-public combat observation

**Status:** rejected; superseded by COMP-008  
**Date:** 2026-07-23  
**Commit / simulator build:** record after the public-state patch is rebuilt and
verified; both arms must use the same build and working tree.

**Hypothesis:** adding complete human-visible pile contents, upgrade state,
relics/counters, public per-turn counters, and current-turn action history improves
prompt-only Gremlin Nob play relative to the legacy partial observation.

**Intervention:** `combat_observation=combat_public_v1`, as historically defined by
[`combat_public_state_contract.md`](combat_public_state_contract.md).

**Control:** `combat_observation=legacy` through an explicit compatibility switch
in the same executable. No prompt, glossary, output schema, model, or generation
change may differ between arms. This fresh control is required because increasing
K changes the policy-sample indices relative to COMP-001's historical K=4 run.

**Evaluation:** the 10 replay-validated held-out Gremlin Nob windows, K=8 stochastic
samples per window and arm (80 episodes/arm), paired by window after averaging
the eight samples. Use local vLLM-Metal with
`mlx-community/gemma-4-e4b-it-bf16`, native thinking, maximum 8,192 completion
tokens, temperature 0.7, top-p 0.95, top-k 64, one retry, concurrency 12, and an
80-decision task cap. Use identical world/window/rollout indices and policy seeds
in both arms. No RunPod resources are used. The original source manifest
contained 11 held-out windows, but isolated replay validation performed before
this run found that `seed_128_r0_w0` diverges on the current simulator build. It
is explicitly excluded in a versioned validated manifest rather than skipped at
runtime. This leaves 32 train and 10 holdout windows.

**Primary metric:** paired per-window task-reward delta, public minus legacy,
with a seed-0 95% bootstrap CI over the 10 window deltas.

**Secondary metrics:** win/survival rate, convincing-win rate, HP loss, turns,
action shares, invalid rate, decision-cap truncation, latency, prompt/completion
tokens, and p50/p90/p99 prompt length. Inspect per-window deltas and exact run
completeness before interpreting aggregates.

**Decision rule fixed before the run:**

- If the reward CI lower bound is above zero, treat prompt-only benefit as
  demonstrated and adopt `combat_public_v1`.
- If the estimate is inconclusive but invalid/truncation health does not worsen
  materially, retain the public interface for teacher learning because it removes
  known observation aliases; report that prompt-only benefit was not established.
- If mean reward regresses by at least 0.10, or invalid/truncation rate rises by
  at least 2 percentage points, diagnose prompt size/format and compact the
  representation before making it the default. Do not silently fall back to the
  partial observation for teacher training.

**Result:** stopped deliberately after 71 decisions in the first (legacy) arm.
The prompt stated that Enrage triggers on Skills, but neither state nor legal
actions identified card types. Multiple native-thinking transcripts then called
Bash/Strike Skills, inferred Strength gain from damage or energy cost, and said
Defend did not trigger Enrage. No arm completed and no matched report exists.
This is an interface failure, not a prompt-only comparison result.

**Artifacts:** partial diagnostic traces are preserved at
`data/competence/comp_002/eval/legacy_v1_invalid_missing_card_type/`. They lack
completed sidecars and must not be resumed. Tracked historical preregistration:
[`configs/competence/comp_002.json`](../configs/competence/comp_002.json).

### COMP-004 — search-label stability and privilege audit

**Status:** complete  
**Date:** 2026-07-23  
**Dependency:** the non-mutating search query and `combat_public_v1` snapshots
must pass before the audit is interpreted.

**Question:** is a 5,000-simulation search query stable enough to label
base-visited Nob states, and how often does its preferred displayed action depend
on hidden draw order?

**Sample:** replay the 10 replay-validated held-out Nob source windows and query the first
base-visited decision of every turn. Continue following the recorded base action
after each query. This gives several tactical states per fight without letting
the search teacher replace the policy distribution. As in COMP-002,
`seed_128_r0_w0` is an explicit pre-run simulator-replay exclusion.

**Queries:** use budgets 1,000, 5,000, and 50,000 with independent search seeds 1, 2, and 3
on the unmodified cloned state. At 50,000 only, additionally query cloned draw
pile permutations 100 and 101 for each search seed. All queries must leave the
live state text, structured public state, legal actions, and subsequent source
replay unchanged.

**Correctness deviation recorded before the rerun:** on this libc++ build,
`std::default_random_engine` normalizes seeds 0 and 1 to the same stream. All 280
matched seed-0/seed-1 query pairs in the first artifact were identical, so the
nominal three-vote consensus was invalid. Preserve that artifact as
`*_seeds_0_1_2_invalid.*`, reject it as evidence, and rerun the otherwise
unchanged design with empirically distinct seeds 1, 2, and 3. The collector now
rejects duplicates and the 0/1 pair.

**Primary metrics:** displayed first-action agreement between the per-state 5k
and 50k consensus; unanimous-action rate across search seeds at each budget; and
50k consensus after pooling the unmodified state with the two hidden draw-order
perturbations. Also report ties, selection method, root visits/means, predicted
HP, and disagreements with the recorded base action.

**Provisional collection decision fixed before results:** use 5k as the default
label budget only if 5k versus 50k consensus agreement is at least 90%, at least
80% of audited states are unanimous across the three 5k search seeds, and no
mutation/parity failure occurs. A row is eligible for teacher SFT only if its 50k
unperturbed consensus is non-tied, its 5k consensus matches it, and the pooled
50k hidden-order consensus fraction is at least two-thirds. Otherwise collect a
50k consensus label or omit the row; never resolve ambiguity by taking one
arbitrary search seed.

**Correctness clarification added after the audit but before any training:** the
pooled hidden-order consensus must also select the same displayed action as the
unmodified 50k consensus. The first report implementation checked only its
fraction, which could admit a confidently contradictory hidden-order majority.

**Artifacts:** `data/competence/comp_004/` containing raw JSONL queries, a
manifest, aggregate report, per-state disagreement table, and the exact public
observation hashes. The initial 10-simulation one-window file is a plumbing smoke
test only and is excluded from the result.

**Invalidated initial result (do not use):** 56 first-per-turn states from all 10 replay-valid holdout windows
produced 840 teacher queries. The 5k consensus matched the 50k consensus on
46/56 states (82.1%) and was unanimous across seeds on 43/56 (76.8%), missing
both preregistered thresholds. At 50k, pooled unmodified and two permuted hidden
draw orders reached at least two-thirds consensus on 45/56 states (80.4%); mean
pooled consensus was 80.6%, and the pooled action matched the unmodified
reference on 50/56 states (89.3%). Under the corrected full row rule, 38/56
states (67.9%) were eligible. Search's reference action differed from the recorded base
action on 75.0% of states, confirming that it supplies substantial
counterfactual policy information rather than merely cloning the source.

The initial report said 39/56 because it checked the pooled consensus fraction
without checking its action identity. `seed_108_r0_w0` turn 3 exposed the bug:
the unmodified 50k teacher unanimously selected action 1 while the pooled hidden
states selected action 2 on 6/9 queries. The filter and regression tests were
fixed before building or training a dataset.

**Corrected independent-seed result:** the final frozen-interface run again
covered 56 first-per-turn states, 10 holdout windows, and 840 queries. The 5k
consensus matched 50k on 43/56 states (76.8%) and was unanimous on only 33/56
(58.9%); both preregistered gates fail. The unmodified 50k vote was unanimous
on 32/56 (57.1%). Pooled unmodified/permuted 50k queries reached at least
two-thirds agreement on 39/56 (69.6%); among the 52 states with a non-tied
unmodified reference, the pooled action matched it on 44/52 (84.6%). The full
5k row filter retained 31/56 (55.4%). Once 5k is rejected, seven additional
states whose 50k and hidden-order votes agree become usable: direct-50k
eligibility is 38/56 (67.9%). The reference teacher differed from the recorded
base action on 39/56 (69.6%). These figures use simulator binary SHA-256
`94f2f687ab31bc38694fd5fafeb343d7a38bb17e6d7d237dad8a401fcdc5dd09`
and validated-manifest SHA-256
`db660075f3bf04cb5d4061ccc433df0c734f6957e7a8416295511b06694ea749`.

**Decision:** reject 5k as the default teacher. Training collection uses 50k
with independent seeds 1, 2, and 3, and hidden-order permutations 100 and 101.
The earlier 0/1/2 decision text is superseded because 0 and 1 alias on this
standard-library implementation.

**Artifacts:**
`data/competence/comp_004/audit/holdout_turn_first.jsonl`, its sibling manifest,
and `data/competence/comp_004/audit/report.json`.

### COMP-004A — search root-selection behavioural diagnostic

**Status:** complete  
**Date:** 2026-07-23  
**Purpose:** the native search agent commits the first action of the single best
observed winning rollout when one exists. In the corrected audit that selected
action was unanimous across three 50k search seeds on 32/56 states, whereas the
display-equivalent aggregated most-visited root was unanimous on 50/56. Higher
stability does not itself prove that the latter plays better, so compare them
before changing the teacher target.

**Sample and queries:** run one deterministic privileged-teacher episode from
each of the 10 validated Nob holdout starts for each policy. At every live
decision, query the unmodified state at 50k with independent seeds 1, 2, and 3.
For the `winning_sequence` arm, a seed votes only when search found a winning
sequence whose first action maps to the displayed menu. For the
`aggregated_most_visited` arm, map every valid raw root edge to its displayed
description, sum visits across display-equivalent edges, and let the seed vote
only for a unique maximum.

**Fail-closed consensus:** execute an action only when one displayed action has
at least two of all three seed votes. Abstentions remain in the denominator. An
ambiguous vote stops that policy/window with `ambiguous_consensus`, counts it as
incomplete with reward -1, and supplies no fabricated terminal HP-loss value.
There is no action-index, single-seed, or alternate-policy fallback.

**Metrics:** completion/ambiguity rate first, then wins, mean task reward, and HP
loss over completed episodes; also query unanimity, per-decision arm agreement,
action counts, source hashes, and exact search provenance. This is a K=1 teacher
diagnostic, not a model competence claim.

**Decision rule:** prefer aggregated root visits for later distillation only if
it has fewer ambiguous stops and does not lose a fight or reduce mean reward
relative to winning-sequence consensus on these starts. Otherwise retain the
current selected-action labels and interpret their lower stability as a teacher
limitation. The result governs recollected v2 tiny/full-data and DAgger targets;
the frozen v1 COMP-005 data was separately invalidated by missing card type.

**Result:** aggregated most-visited root completed and won 7/10 starts, stopped
ambiguously on 3/10, and had mean reward 0.1025 across all starts (fail-closed
ambiguity reward -1). Winning-sequence consensus completed and won 4/10, stopped
ambiguously on 6/10, and had mean reward -0.315. Query eligibility was 97.9%
versus 92.9%; query unanimity was 89.0% versus 61.2%. The arms chose the same
action on 26/38 aligned live states. Neither completed arm lost. Winning sequence
had slightly better reward on the three starts both arms completed, but the
preregistered all-start gate prioritizes reliable completion and therefore
selects aggregated root visits. This CPU diagnostic is behaviorally valid despite
v1 descriptions because card-type rendering does not change native search state,
edge visits, action bits, or execution.

**Artifacts:** `data/competence/comp_004/root_policy_diagnostic/` with one raw
record per arm/window, a compact report, and source/interface hashes.
Report SHA-256:
`8c67d306f5e8eb644224eeac00085120e757e59d43675b6315a40ae6c9f6d5bf`.

### COMP-005 — action-only search-teacher SFT

**Status:** rejected before model loading; superseded by a v2 recollection  
**Date:** 2026-07-23  
**Dependency:** COMP-004's corrected 50k collection decision and the action-mask
trainer smoke must pass. Both are complete.

**Rejection reason:** every collected prompt used `combat_public_v1` and omitted
card type. No model was loaded, trained, or scored on these data, so there is no
model-selection contamination. The raw labels, filtered 493-row dataset, tiny32
subset, and frozen schedule remain useful pipeline/provenance diagnostics but
are ineligible for competence training. They were moved intact to
`data/competence/comp_005_v1_invalid_missing_card_type/`.

**Hypothesis:** giving Gemma E4B counterfactual 50k-search actions and applying
loss only to the action/required format tokens can first be memorized on a tiny
set, then improve teacher agreement and held-out Nob outcome. This tests the
learning pipeline before interpreting model capacity.

**Train-state collection:** replay every combat micro-decision in the 32
replay-valid Nob train windows while continuing to follow the recorded base
action. At every state, query 50k search with independent seeds 1, 2, and 3 on the
unmodified clone and on hidden draw-order permutations 100 and 101. Deduplicate
by the public observation plus displayed action menu. Retain a row only when the
unmodified consensus is non-tied with fraction at least two-thirds and the
pooled hidden-order consensus selects that same action with fraction at least
two-thirds. The teacher target is the
unmodified consensus action; no hidden teacher value enters the student prompt.
The 10 holdout windows and COMP-004's 56 first-per-turn rows remain untouched by
training.

**First training arm:** `mlx-community/gemma-4-e4b-it-bf16`, local MLX LoRA,
neutral frame, historical `combat_public_v1`, no-thinking chat prefix, immediate
`{"action_index": i}` target, historical mixed format-plus-action mask, 8 adapted layers, batch size 1,
learning rate 1e-4, and maximum sequence length 8,192. Freeze the exact iteration
count/checkpoint rule after the filtered dataset count is known but before
inspecting model results. Do not compare this no-thinking adapter against the
native-thinking COMP-002 base arm.

**Collection result and frozen tiny schedule (fixed before model results):** all
652 base-visited decisions in the 32 train windows were queried, producing 5,868
independent 50k searches. There were no duplicate public-state hashes. Direct
50k filtering retained 493 states; 38 had a tied unmodified reference, 106 had
hidden-order agreement below two-thirds, and 15 had a hidden-order majority for
a different action. Exact prompts in the 493-row dataset have mean 862.9 tokens,
p50 848, p90 998, p99 1,101, and max 1,144.

The tiny positive control is the first 32 survivors in lexicographic
`public_state_hash` order after filtering/deduplication. Train all 32 rows (MLX
validation fraction 0) for exactly 640 batch-size-1 optimizer updates: 20
effective passes, 8 adapted layers, learning rate 1e-4, maximum sequence length
8,192, shuffle seed 0, report every 32 updates, and save every 64. Select the
final 640-update adapter without consulting an intermediate accuracy. Score
top-1 on the same 32 rows; below 95% is a failed overfit control and stops the
full-data run. The subset manifest records 32/493 selection and final-only token
accounting (32 action tokens plus 256 required format tokens).

The current tokenizer thus produces eight supervised required-format tokens for
each single supervised action-value token. Training cross-entropy can fall by
learning constant JSON syntax without learning the state-conditioned choice.
The overfit gate is top-1 teacher action, not aggregate train loss. If it fails,
action-token reweighting is a new preregistered intervention rather than a silent
change to this run.

**Ordered positive controls:**

1. deterministically select 32 filtered train states and reach at least 95%
   train top-1 teacher-action agreement;
2. improve teacher-action log-probability and top-1 agreement on the fixed,
   eligible COMP-004 holdout states (38 direct-50k rows across 9 of 10 windows;
   the tenth contributes no row rather than a weak label);
3. run a matched no-thinking base/adapter K=8 evaluation on all 10 replay-valid
   held-out fights and apply Gate C1's reward/outcome thresholds.

Stop at the first failed link. A tiny-set overfit failure diagnoses trainer/model
fit; a held-out action failure diagnoses generalization/data; a behavioural
failure after action agreement diagnoses teacher/interface mismatch.

**Historical artifacts:** `data/competence/comp_005_v1_invalid_missing_card_type/`
with raw labels, filtered dataset and
manifest, deterministic tiny subset, adapters/checkpoints, teacher-action
reports, matched fight rollouts, token report, and result record. Tracked tiny
positive-control preregistration:
[`configs/competence/comp_005_tiny32.json`](../configs/competence/comp_005_tiny32.json).

### COMP-008 — card-type-complete public interface and artifact migration

**Status:** complete; interface and tiny-fit positive controls passed  
**Date:** 2026-07-23  

**Hypothesis:** much of Gemma's apparent Nob strategy failure is an observation
alias, not irreducible E4B capacity: exposing the game-visible card type will let
the policy distinguish Attacks from Skills and apply Enrage correctly. This does
not claim that type alone is sufficient for strong play.

**Interface intervention:** add explicit `[Attack]`, `[Skill]`, `[Power]`,
`[Status]`, or `[Curse]` tags to every combat hand/pile card, card-play action,
and card-selection candidate; add `type` to v2 structured card dictionaries and
use the same descriptions in privileged-search results. Preserve legacy and v1
rendering behind explicit switches for historical replay. Do not expose rarity,
hidden pile order, RNG, or search output.

**Migration contract:** bump the policy-visible start signature to schema 2 and
write new passing-only manifests rather than mutating validated v2 files. Every
teacher/eval consumer must request `combat_public_v2` explicitly. Historical
COMP-002 partial rollouts and COMP-005 datasets retain their hashes and are
marked ineligible; no training result may mix v1 and v2 prompts.

**Teacher selection intervention:** the recollected search vote is the unique
displayed action with maximum visits after aggregating display-equivalent native
root edges. Each of search seeds 1/2/3 votes or abstains; hidden draw-order seeds
100/101 remain in the denominator. Use a label only at at least two-thirds
consensus and only when pooled hidden-order consensus selects the same action.
This applies COMP-004A prospectively; raw native winning-sequence selection is
retained in provenance and never rewritten.

**Verification and decision gate fixed before model results:**

1. [complete] binding patch applies/reverses cleanly, contains no pybind submodule hunk, and
   builds in Release;
2. [complete] v1 text/actions remain free of type tags while v2 text/actions/structured cards
   and search edges agree on types; all coupled parsers accept both versions;
3. [complete] isolated replay validation produces schema-2 passing-only cohorts for Nob,
   Lagavulin, and Sentries with explicit exclusions and current binary hash;
4. [complete] rerun the held-out first-per-turn stability audit under v2 and aggregated root
   votes, then recollect train labels; no row with an abstention-induced consensus
   below two-thirds or a hidden-action conflict is eligible;
5. [complete] rebuild/token-audit the deterministic v2 tiny subset and freeze
   the replacement schedule before loading a model;
6. [complete] fit the final-step adapter on 32/32 teacher targets, passing the
   preregistered 31/32 gate without consulting intermediate accuracy.

**Primary artifacts:** `docs/combat_public_state_contract.md`, simulator patch
and binary hashes, new `source.validated_v5.json` manifests and validation reports,
`data/competence/comp_008/` audit records, and a superseding tracked training
config created only after the v2 filtered count is known.

**Observed pre-model evidence:** the v2 held-out audit contains 60 states from
11 windows. 5k agrees with 50k on 42/60, is unanimous on 54/60, and qualifies
only 41/60 rows; direct 50k qualifies 57/60, so 5k remains rejected. The dense
train collection contains 652 states and 5,868 queries; 600 rows survive search
and hidden-order filtering. The action-only builder retains all 600, and its
stable public-hash prefix defines the 32-row overfit set. A separate held-out
first-per-turn SFT artifact contains the 57 eligible audit states.

**Training result:** the frozen easy tiny32 run completed 640 updates (20
passes), retained all 32 rows, and reached 32/32 teacher-action top-1 versus
18/32 for base. Mean teacher-candidate probability rose from 0.568 to 0.981.
The separately scored held-out diagnostic moved 22/57 to 29/57 and probability
0.372 to 0.470, but NLL worsened from 1.777 to 2.898 because remaining errors
became overconfident. This passes model/trainer memorization only; it is not a
competence or generalization result. Exact evidence is tracked in
[`comp_008.json`](../configs/competence/comp_008.json) and
[`comp_008_result.json`](../configs/competence/comp_008_result.json).

### COMP-009 — dense every-micro-action teacher SFT

**Status:** complete; held-out action gate failed  
**Date:** 2026-07-23  

**Hypothesis:** three full passes of the proven mixed mask over all 600 eligible
base-visited micro-actions will improve the fixed 57-state first-turn holdout.

**Result:** the adapter retained all 600 rows and completed 1,800 updates, but
scored 21/57 held-out actions versus 22/57 for base. It predicted index 0 on all
57 held-out rows and 17/32 of the old tiny32—exactly the tiny set's number of
index-0 targets. The full train distribution was 337/148/85/28/2 over indices
0–4, whereas held-out was 21/15/13/8. Many train rows are late-turn or forced
end-turn states, while held-out contains the first collected decision per turn.

**Decision:** reject behavioural evaluation. The all-index-0 policy directly
demonstrates majority-prior collapse. State-selection mismatch and only three
passes are leading explanations, but COMP-009 alone does not isolate them.
Project the dense source to first-per-turn states before confidence filtering
and compare masks on a harder, less-skewed subset. See
[`comp_009.json`](../configs/competence/comp_009.json) and
[`comp_009_result.json`](../configs/competence/comp_009_result.json).

### COMP-010 — first-per-turn action-value-only diagnostic

**Status:** complete; Stage A failed and Stage B was not run  
**Date:** 2026-07-23  

**Interventions:** choose the minimum `source_decision_index` for each
`(window_id, turn)` from the raw collection before confidence/hidden-order
filtering, then use the experimental `--mlx-action-token-only` mask. The frozen
full projection contains 150 eligible rows with targets 43/43/44/19/1 and only
three end-turn targets. The harder stable-hash tiny32 contains all five action
indices (7/7/12/5/1).

**Result:** after ten passes, the tiny adapter improved 8/32 to 11/32, below the
31/32 preregistered gate. Mean teacher probability rose 0.259→0.312 and NLL
improved, but predictions remained restricted to indices 0 and 2 and canonical
teacher sequence log-probability fell from -16.5 to -112.1. Per the frozen
gate, the full150 arm and behaviour were not run.

**Decision:** a matched 20-pass mask comparison was required before blaming
format-token dilution, optimizer duration, or capacity. See
[`comp_010.json`](../configs/competence/comp_010.json) and
[`comp_010_result.json`](../configs/competence/comp_010_result.json).

### COMP-011 — matched mixed-mask versus action-value-only fit

**Status:** complete; only the mixed arm passed  
**Date:** 2026-07-23  

**Control:** identical harder tiny32, E4B revision, rank-8/eight-layer LoRA,
batch size, learning rate, seed, and 640 updates (20 passes); only the realized
loss mask differs.

**Result:** mixed required-format-plus-action supervision reached 31/32,
teacher probability 0.821, NLL 0.220, and predictions across all target classes.
Action-value-only reached 15/32, probability 0.419, NLL 1.029, and predicted
only indices 0/2. Its canonical sequence log-probability was -96.875 versus
-0.215 for the mixed arm.

**Decision:** keep the mixed mask for the aligned full150 experiment. Format
loss is useful optimization/output-contract scaffolding here; COMP-009 cannot
be explained as simple 8:1 format-token dilution. This bounded one-seed result
does not test action upweighting. See
[`comp_011.json`](../configs/competence/comp_011.json) and
[`comp_011_result.json`](../configs/competence/comp_011_result.json).

**Provenance limitation across COMP-008–011:** all referenced local dataset,
manifest, evaluation, adapter, preregistration, dependency, simulator, and
result hashes were recomputed successfully on 2026-07-23. COMP-008 did not
record trainer-source hashes, and COMP-009's recorded dirty-worktree trainer
hashes refer to intermediate source bytes no longer present. The reported
adapter/evaluation chain remains internally verifiable, but exact training-code
reconstruction for those two runs is incomplete. The random temporary
`mlx_data_*` paths in adapter configs have also been removed; durable native
preparation reports and dataset hashes remain. COMP-010 onward harden this
contract, but all generated `data/` and currently untracked competence configs
must be archived/committed before they can be called durable outside this
worktree.

### COMP-012 — aligned full150 mixed-mask teacher SFT

**Status:** complete; train passed, development failed, behaviour not run  
**Date:** 2026-07-23  

**Hypothesis:** the COMP-011-positive mixed mask trained for 20 passes over all
150 eligible first-per-turn train states will fit at least 135/150 and improve
the frozen 57-state held-out teacher-action set without collapsing prediction
diversity.

**Frozen gate before behavioural evaluation:** train at least 135/150; held-out
at least 32/57; held-out mean teacher probability at least 0.3722; held-out NLL
at most 1.7769; non-degenerate action predictions; zero invalid/skipped rows.
The final step-3,000 adapter is the only scored checkpoint. Failure stops before
behaviour and redirects work to capacity/DAgger diagnostics. The immutable
schedule is [`comp_012.json`](../configs/competence/comp_012.json); append the
observed result to a separate result file rather than rewriting it.

**Result:** the final adapter retained 150/150 rows and completed 3,000 updates.
It scored 149/150 train states with probability 0.955 and NLL 0.052; predictions
43/42/45/19/1 nearly reproduced the target-index counts 43/43/44/19/1. On the
57 development rows it scored 28/57 versus base 22/57, with 13 improvements,
seven regressions, and predictions across indices 0–3. Probability improved
0.372→0.479, but NLL worsened 1.777→2.572 because several wrong choices
became extremely confident. The 32/57 agreement and NLL gates failed. No game
rollouts were run. Exact evidence is in
[`comp_012_result.json`](../configs/competence/comp_012_result.json).

**Exploratory error structure (post hoc):** aggregate prompt/state marginals are closely matched, but
performance is window- and position-clustered. COMP-012 gets 23/36 targets at
display indices 0–1 and 5/21 at indices 2–3; all eight index-3 development
targets fail even though all 19 train index-3 targets fit. It gains 13 nonzero
choices but regresses on seven base-correct index-0 rows. Twenty teacher
probabilities below 0.1 account for 88.7% of total NLL. Coarse card coverage is
not absent—47/51 development card types occur in train—but conditional examples
do not transfer reliably across unseen decks/windows. For example, seven of
eight train Defend targets are turn 0, while development Defend targets average
turn 2.86 under materially different incoming damage.

These figures are derived from the row-level COMP-012 train/development
reports, but the exploratory calculation is not yet a separately frozen
analysis artifact. Do not promote a particular subgroup to a primary endpoint
without first freezing its derivation.

**Decision:** this is a static teacher generalization/coverage/regularization
failure, not an adapter-fit failure or behavioural competence. The 57 states
have informed multiple intervention choices and are development data from now
on. Do not select an adapter on them and then call the same cohort held-out.

### COMP-013 — halfway-checkpoint generalization diagnostic

**Status:** complete; no checkpoint passed as a sufficient solution  
**Date:** 2026-07-23  

**Purpose:** use existing COMP-012 checkpoints, without retraining or behaviour,
to distinguish late confidence sharpening from an absolute display-position
shortcut and broader window/deck coverage failure. This diagnostic cannot
retroactively pass COMP-012.

**Frozen checkpoint diagnostic:** score the byte-identical step-1,500/10-pass
checkpoint on train150 and development57. Strong late overfit required at least
the final checkpoint's 28 correct plus NLL no worse than base; partial late
overfit required at least 28 correct plus an NLL reduction of 0.25 from final.
The preregistration is
[`comp_013.json`](../configs/competence/comp_013.json), SHA-256
`8b6dbcbe4858a4087bcabf29ca3abc36d4594b0fa890a0eecdb75d4394060aaa`.

**Observed result:** step 1,500 scores 102/150 train and 27/57 development,
whereas step 3,000 scores 149/150 and 28/57. The second half of training
therefore memorizes 47 additional train states for one development choice. NLL
is 1.183 at step 1,500 versus final 2.572 and base 1.777. Late training sharply
damages calibration, but the earlier checkpoint does not meet either frozen
rule for early stopping as a sufficient solution. See
[`comp_013_result.json`](../configs/competence/comp_013_result.json).

The causal conclusion is deliberately narrower than the frozen result file's
branch label: late overconfidence is established and early stopping alone did
not pass. This checkpoint comparison cannot distinguish residual data coverage,
representation/capacity, or legal-action-order sensitivity; COMP-014 and the
subsequent isolated control address those alternatives.

### COMP-014 — cyclic action-order sensitivity audit

**Status:** complete; sensitive branch selected, behaviour not run  
**Date:** 2026-07-23  

The preregistration is
[`comp_014.json`](../configs/competence/comp_014.json), SHA-256
`f321cc3812660fc929d16a4a60ef360b60b95c98c89b526e85c1ac9e10994276`.

After its pure parser/rotation/report path is tested,
score every non-zero cyclic rotation of each development legal-action menu at
steps 1,500 and 3,000. Map predictions back to semantic actions and report
invariance, teacher accuracy/NLL by assigned position, rotation, and window.
The transformation is a right rotation: `old_to_new[i]=(i+r) mod n` and
`new_to_old[j]=(j-r) mod n`; descriptions and embedded target indices remain
byte-identical. Re-render through the exact offline tokenizer and require an
exact original prompt round trip before scoring. Failure to follow the same
semantic action under renumbering diagnoses action-order sensitivity and
motivates permutation augmentation. Stability under cyclic rotations with
persistent window failures points instead to state/deck coverage. Cyclic
sensitivity alone does not prove a specific absolute-index shortcut, and a pass
does not establish invariance to arbitrary permutations. Freeze the config and
all code/reference hashes before its first model load.

Honor the frozen rule if either checkpoint is sensitive, but report the result
as checkpoint-dependent if only one crosses the threshold. Any augmentation
follow-up must compare against an unaugmented control with matched source-row
draws, optimizer updates, batch size, and policy-bearing token exposure; merely
adding every rotation and therefore multiplying training compute is not an
augmentation-only experiment. Freeze a prospective checkpoint grid and select
on development NLL with a top-1 floor rather than defaulting to the final pass.

**Observed result:** the health gate passed for both checkpoints: 57 source
rows, 221 nonzero rotations, and 1,134 candidate sequence scores each. Step
1,500 has semantic top-1/top-set invariance 0.448/0.403 and mean semantic
probability TV 0.400; step 3,000 falls to 0.253/0.249 with TV 0.696. Only 4/57
and 1/57 source rows preserve top-1 under every rotation. Across identity plus
rotated variants, neither checkpoint gets any target assigned to display
indices 4–7 correct. This is well beyond the frozen sensitivity thresholds and
is not a tie-breaking artifact. Exact evidence is in
[`comp_014_result.json`](../configs/competence/comp_014_result.json), SHA-256
`e4a21824ad47511bb46c3c35c355f9bc46d09b37914ce235930650729525a484`;
the full report SHA-256 is
`fbdcf176987e15c7371e54ecc24270362ffdf9762b9076c0339ffee295fa55ac`.

**Decision:** select the frozen action-sensitive branch. COMP-015 must isolate
cyclic augmentation from compute and source exposure before any behaviour run.
The result is consistent with a strong early-menu-position bias, but it does
not identify the internal mechanism or show that augmentation will improve
fresh play.

### COMP-015 — exact-schedule cyclic augmentation control

**Status:** complete — original strict claim failed; augmentation retained only as a possible later component  
**Date:** 2026-07-23  

The immutable schedule is
[`comp_015.json`](../configs/competence/comp_015.json), SHA-256
`303a7cdf3bbae3df8dd9cd4cf0b0d5a853777e4059e8f6e388872eefc92294b6`.

**Question:** does balanced cyclic reordering teach semantic action equivariance
without purchasing the result through more source examples, optimizer updates,
or supervised tokens?

Use the unchanged 150 first-per-turn source states. Run three paired replicates
with MLX/LoRA seeds 0, 1, and 2. Each pair starts independently from the same
frozen base model and receives an identical deterministic schedule of 20 passes
× 150 states = 3,000 batch-1 updates. Each pass visits every source state once
without replacement. The control always uses the deployment/identity order;
the treatment cycles each state's rotations, including identity, so per-state
rotation counts differ by at most one. The treatment must not enumerate more
rows for larger menus.

The exact-schedule trainer mode must preserve file order through tokenization
and optimization, bypassing the normal length sort and RNG batch permutation.
Every paired step must share the same source identity, and the realized files
must have equal prompt/completion length, padded input length, nine supervised
tokens, and one action token. The builder records source-schedule, rotation,
pair-alignment, tokenizer, dataset, and code hashes and fails closed if these
invariants do not hold. The frozen 730 possible source×cyclic variants have
already been checked locally and match their identity token lengths; the final
realized manifests must assert this again.

Before the six full arms, repeat one short exact-schedule seed/config twice and
require identical retained-row reports and adapter bytes, or stop to diagnose
local determinism. Keep dropout zero. This smoke is trainer health, not a
competence result.

Save steps 750, 1,500, 2,250, and 3,000. Step 1,500 is the selection-free
primary causal endpoint because every observed menu rotation has been exposed
and COMP-013 shows that step 3,000 can be a badly overconfident endpoint. At
fixed step 1,500, score unpermuted train150, unpermuted development57, and the
full COMP-014 cyclic diagnostic for all six arms. Treat the three seeds as
paired training replicates and cluster state-level summaries by the 11
development windows; rotations are not independent observations.

The primary augmentation claim requires directionally better invariance, TV,
all-variant source-macro agreement, and all-variant NLL in all three pairs; at
least two treatment seeds must cease meeting COMP-014's sensitivity rule
(invariance ≥0.75 and TV ≤0.20). Mean paired improvements must reach +0.20
invariance, -0.20 TV, +0.10 macro agreement, and -0.25 NLL, with no
unpermuted development agreement or NLL regression. Also report, but do not
require, the stronger 0.90-invariance/0.10-TV stability boundary.

Separately select a deployment checkpoint within each run from
750/1,500/2,250/3,000: retain checkpoints with at least 28/57 unpermuted
development choices, choose the lowest development NLL, and break an exact tie
toward the earlier checkpoint. These are development-selected checkpoints, not
fresh evidence. No eligible checkpoint means that run fails selection.

Before behaviour, at least two of three selected treatment runs must score
135/150 train and 32/57 development, maintain teacher probability ≥0.372162
and NLL ≤1.776904, remain non-degenerate with zero invalid/skipped rows, meet
invariance ≥0.75 and TV ≤0.20, and beat their exact-schedule controls. Choose a
deployment adapter among qualifying treatment seeds by lowest development NLL,
then higher agreement, then lower seed. Only then run a matched v2 development
behaviour comparison among base, treatment, and its same-seed/same-step control.
COMP-012 remains a historical reference, not the primary control.

**Fast-iteration amendment (2026-07-23):** the original three-pair design above
remains the record of the confirmatory question, but its remaining compute is
retired under
[`comp_015_fast_iteration_stop.json`](../configs/competence/comp_015_fast_iteration_stop.json),
SHA-256
`88755f50944d4a0c9b4543e5f089c81cca215e8bd157734750f654096f929774`.
Seed 0 already makes the strict all-three-pair success rule impossible, fails
the behaviour gate, and is corroborated by COMP-016's independent measurement
audit. The complete seed-1 control and treatment step-1,500 snapshots will be
scored because doing so needs evaluation only. The completed seed-2 control is
retained as an unpaired artifact and supports no treatment inference. Its
treatment was stopped before any model weights or checkpoint were written.

Prospectively, training seeds are allocated sequentially during development:
use one tiny run to reject broken ideas cheaply; run one full development seed
only after the tiny gate; add a second training seed only when the first full
result is promising or meaningfully ambiguous. Require additional frozen
training repeats before a robust positive claim, final-cohort release, or
promotion to a retained competence checkpoint. Multiple sampled rollouts from
the same trained policy address variable generation and game outcomes; they do
not substitute for independent training seeds.

**Ambiguity trigger (2026-07-23):** scoring the already-complete seed-1 pair
changed the available evidence. Unlike seed 0, treatment gains seven direct
development actions, raises invariance by 0.489, and improves cyclic
source-macro agreement/NLL, while still missing the frozen action and TV gates.
This is the sequential policy's explicit promising-or-ambiguous case. Under
[`comp_015_ambiguity_triggered_seed2.json`](../configs/competence/comp_015_ambiguity_triggered_seed2.json),
SHA-256
`c0d621d964390c9917d450309aeb04ef622706e78f2ac8d024bdba3734f20ac8`,
train only the treatment paired with the already-complete seed-2 control, score
the fixed direct battery, and close COMP-015 regardless. The exact current
unit-weight preparation is byte-identical to the stopped process's 3,000-row
prepared file (`91177b9b…ad3`), and unit weight retains the historical boolean
mask computation. This new evidence-triggered run does not retroactively make
the earlier stop decision wrong and cannot restore the original strict claim.

**Closed result:** seed 2 confirms that the useful part of seed 1 was not a
one-off. Control→augmentation moves direct development agreement
28/57→32/57, direct development NLL 1.291→1.254, cyclic invariance
0.362→0.629, TV 0.473→0.298, and cyclic source-macro agreement/NLL
0.371/2.073→0.503/1.262. These changes pass all five frozen carry-forward
conditions. However, train fit falls 113/150→81/150, invariance remains below
0.75, and TV remains above 0.20, so augmentation alone is not a deployable
competence intervention. Close COMP-015 with no behaviour and no final-cohort
access. The content-addressed result is
[`comp_015_result.json`](../configs/competence/comp_015_result.json).

### COMP-018 — fixed move-choice weight 8

**Status:** complete — harder-tiny32 gate failed; full150 not run  
**Date:** 2026-07-23

The immutable design is
[`comp_018.json`](../configs/competence/comp_018.json), SHA-256
`5512cc8c71d55e9e7252cefa4a4d11ceeaf78870bc36da604c10abf6ceb54d54`.
It keeps required-format weight at 1, gives the one move digit weight 8, and
normalizes by total weight mass. On harder tiny32 this exactly balances
format/action mass at 256/256. The frozen gate required at least 31/32 direct
and greedy choices, mean teacher probability at least 0.80, NLL at most 0.30,
canonical/legal JSON on 32/32, and exact preparation/health checks before any
full150 run.

The final treatment passes preparation and JSON health but fails action learning
by a wide margin: direct and greedy agreement are both 7/32, probability is
0.228, and NLL is 1.499. Float32 direct scoring selects displayed index 0 on
all 32 states; native greedy selects index 1 on all 32 because all move-digit
logits are nearly equal. At step 320, direct agreement is already 7/32 with the
same all-index-0 projection. By step 640 the JSON validity improves from 19/32
to 32/32 while the move distribution does not become state-conditioned. The
0.812 training-loss plateau is therefore not useful move learning hidden by the
aggregate.

Because the reversal was unexpectedly large, one post-failure matched unit
control was frozen before model load under
[`comp_018_tiny_matched_unit_diagnostic.json`](../configs/competence/comp_018_tiny_matched_unit_diagnostic.json),
SHA-256
`04fb0b10feafbfd09d7e1f4da5e08f20a81ea46d1c0859f46d38c80835d143af`.
It changes only action weight 8→1. The current unit run reaches 17/32 at
step 320 and 31/32 at step 640 with NLL 0.220 and valid/legal JSON on 32/32.
More strongly, its step-320 and step-640 adapter files are byte-identical to
the historical COMP-011 checkpoints. This proves exact current-trainer
backward compatibility and deterministic reproduction.

The implementation audit finds one correctly aligned weighted target digit in
every row, the intended 4.5× action and 0.5625× format gradient changes after
normalization, and successful context-conditioned learning in a compiled
numeric-weight toy. There is no evidence of a dropped or shifted action
gradient. The best interpretation is that strong fixed weighting from the
first update changes the optimization path and removes useful early format
scaffolding, leaving an easy near-uniform digit basin. Endpoint format loss
near zero did not imply that format gradients were unimportant during learning.

Reject fixed weight 8 and do not run full150, add another training seed, or
start a weight sweep. Preserve unit mixed supervision. A gentler or staged
objective remains a future hypothesis, not the next experiment. The exact
result and boundaries are in
[`comp_018_result.json`](../configs/competence/comp_018_result.json).

### Fresh final Nob cohort v1

Before any further tuning, 60 archived base-policy rollouts on disjoint world
seeds 600–659 were scanned. They contain 31 Nob windows; a world-seed-level
modulo-2 split reserves 13 windows (12 distinct seeds, with both seed-604 fights
kept together) as final holdout and leaves 18 train windows available for later
data collection. Two isolated current-binary replays pass all 31/31 and produce
identical public start signatures. No teacher query, action score, or model
rollout has been run on the 13 final windows. The tracked embargo and exact
hashes are in
[`nob_fresh_final_cohort_v1.json`](../configs/competence/nob_fresh_final_cohort_v1.json).

The inferential unit is the world seed: first average stochastic samples within
window, then combine the two seed-604 windows into one seed-level observation.
The effective final sample is therefore 12 paired seed clusters, not 13
independent fights. Before release, freeze a clustered bootstrap and power
analysis; if 12 clusters cannot support Gate C1's effect/CI threshold, extend a
second disjoint blind cohort before querying any final state.

At release, collect hidden-order-consensus 50k teacher labels once on a frozen
set of base-visited, first-per-turn states from these windows and score
base/adapter agreement on those identical prompts. This satisfies the
held-out-action part of C1 without using student-dependent states or feeding the
labels back into model selection. Behaviour remains a matched K-sample rollout
comparison aggregated at the same seed-cluster level.

## Artifact and reproducibility contract

Use a layout such as:

```text
data/competence/<experiment_id>/
  preregistration.json
  configs/
  train_rollouts/
  teacher_labels/
  datasets/
  adapters/
  eval/
  logs/
  report.json
```

Every report should make it possible to recover:

- exact world seeds/windows and rollout indices;
- policy seeds and sampling configuration;
- source model, adapter, tokenizer, chat template, and reasoning mode;
- prompt and observation version;
- simulator/binding version and search settings;
- dataset selection, skipped records, multiplicities/weights, and unique-example counts;
- number of action tokens actually receiving loss;
- training steps, effective epochs, learning-rate schedule, and selected checkpoint;
- all primary, secondary, and failure-health metrics.

Prospective configs must bind these fields directly or through explicitly
hashed immutable dependencies; do not rely on an unrecorded inference from an
earlier experiment. COMP-011/012 are historically sparser and remain immutable,
so their dataset/checkpoint dependency chain supplies some of this provenance.
New configs should be self-contained enough to audit model revision,
tokenizer/chat template, observation version, simulator patch/binary, and code
bytes without reconstructing session context.

Generated data may remain gitignored, but the compact config, manifest summary, statistical report, and written conclusion should be tracked.

## Failure interpretation matrix

| Observation | Most likely interpretation | Next diagnostic |
| --- | --- | --- |
| Cannot overfit 32 teacher-labelled actions | Trainer/masking/tokenization bug or insufficient adapter capacity | Inspect token labels, gradients, action NLL, and adapter attachment before any rollout. |
| Training NLL falls; held-out teacher agreement does not | Overfit, observation aliasing, weak/unstable labels, or model generalization limit | Audit duplicate public states/conflicting teacher actions; compare a larger model. |
| Held-out agreement improves; local reward does not | Teacher is not actually good under student observation, metric mismatch, or small action changes are off-distribution | Roll out teacher action from matched states; inspect search value/stability and downstream states. |
| Local reward improves; whole-run floor does not | Curriculum distribution shift, forgetting, or out-of-combat bottleneck | Run hard-fight incidence/entry-state analysis and hybrid search-combat comparison. |
| SFT works; RL does not | Reward/group variance/objective problem | Inspect per-window reward spread, old/new ratios, clip fraction, KL, and zero-group filtering. |
| Larger model works; E4B does not under matched training | Capacity/generalization ceiling | Quantify cost/performance tradeoff and switch student deliberately. |
| Neither model works but search wins | Interface/teacher-label leakage or optimization bug | Recheck public-state sufficiency and small-set overfit before collecting more data. |

## Risks and mitigations

### Search teacher exploits privileged state

The searcher may know shuffled draw order or deterministic RNG. Record teacher privilege, test action stability under hidden-state perturbations, and use consensus/confidence filtering. Do not report teacher imitation as human-information play if the student observation leaks internal state.

### Simulator UB or build-dependent trajectories

Keep UB flags and error sidecars, run simulator-facing tests after patch changes, and do not mix builds in a paired claim. Reproducibility claims remain single-build unless cross-process/cross-build determinism is explicitly checked.

### Reward hacking and stalling

Keep rewards bounded and terminal ordering explicit. Test that stalling, format failure, decision-budget exhaustion, potion dumping, and avoidable death cannot improve reward.

### Curriculum overfitting

Split by world seed, expand the number and diversity of start windows, and keep untouched final holdouts. Report entry-state distributions and do not repeatedly tune against the same 11 Nob holdouts indefinitely.

### Catastrophic forgetting

Evaluate all prior encounters after every curriculum expansion. Use balanced replay and keep per-task metrics rather than one aggregate.

### Evaluation noise

Use K sampled rollouts per matched start, aggregate within start before bootstrap/sign testing, and choose sample size from the minimum effect of interest. Greedy K=1 is only a smoke test.

### Long reasoning dominates cost and loss

Track prompt, thought, required-format, and action-token counts separately. Keep private reasoning masked; use the empirically selected mixed required-format-plus-action objective unless a preregistered experiment changes its weighting.

## Immediate execution order

The next implementation sequence is intentionally narrower than the full roadmap:

1. **Gate C0 v2 behavioural baseline:** retain COMP-001 as a historical pre-v2 K=4 result; run the matched current-interface base arm before any adapter behaviour claim.
2. ~~Write the public-state contract and add observation snapshot tests.~~ Complete.
3. ~~Add human-visible draw/discard/exhaust contents, relics/counters, and current-turn action history.~~ Complete.
4. ~~Run COMP-002.~~ Rejected after its live transcript exposed the missing-card-type defect; no matched result was claimed.
5. ~~Bind a non-mutating search-best-action query with root statistics and provenance.~~ Complete.
6. ~~Rerun search budget/stability/hidden-state sensitivity as **COMP-004** with independent RNG streams.~~ Complete; 5k rejected.
7. ~~Implement compact-target SFT/PG masking and a trainer smoke test.~~ Complete; later controls distinguish the mixed mask from pure action-value loss.
8. ~~Collect the first v1 base-visited Nob teacher dataset.~~ Preserved but rejected: all 493 retained prompts omit card type.
9. ~~Finish v2 verification and regenerate signature-schema-2 Nob/Lagavulin/Sentries cohorts.~~ Complete with two identical isolated passes.
10. ~~Rerun the COMP-008 teacher stability/hidden-order audit and recollect v2 Nob train/holdout labels using aggregated root visits.~~ Complete: 57/60 held-out and 600/652 train states are eligible at direct 50k.
11. ~~Run COMP-008's deterministic v2 easy-tiny32 mixed-mask overfit gate.~~ Complete: 32/32, but held-out NLL worsened.
12. ~~Run COMP-009 on all 600 every-micro-action rows.~~ Complete; stopped before behaviour after all-index-0 collapse and 21/57 held-out agreement.
13. ~~Build the first-per-turn projection and run COMP-010/011 mask controls.~~ Complete: the harder tiny32 is fit by 20-pass mixed format+action loss (31/32), not pure action-value loss (15/32).
14. ~~Finish COMP-012 full150 scoring.~~ Complete: train 149/150, development 28/57; agreement and NLL gates failed, so behaviour was not run.
15. ~~Complete COMP-013's step-1,500 diagnostic.~~ Complete: 102/150 train and 27/57 development with NLL 1.183; early stopping alone is insufficient.
16. ~~Complete COMP-014's preregistered cyclic action-order audit on steps 1,500/3,000.~~ Complete: both checkpoints are strongly sensitive and the late checkpoint is worse.
17. ~~Complete COMP-016's direct move-choice and JSON-format/move-choice loss audit.~~ Complete: genuine float32 direct scores have no unpermuted ties but reproduce the same positional shortcut and greedy choices; format NLL is effectively zero while action NLL remains 1.293/1.495.
18. ~~Close COMP-015 after the triggered ambiguity check.~~ Complete: seed 2 passes every frozen carry threshold but remains below the standalone behaviour stability gate. The strict claim fails; retain augmentation only as a possible later component and run no behaviour.
19. ~~Compare no-native-thinking answer-only output with a short visible reasoning field followed by the move choice.~~ Complete as COMP-017: visible reasoning falls from 26/57 to 19/57 teacher agreement, is less valid, and is roughly eight times slower. Do not run fights; retain immediate action-only JSON.
20. ~~Run COMP-018's fixed weight-8 harder-tiny32 gate.~~ Complete and rejected: weight 8 remains at 7/32 while an exact current unit-weight control reaches 31/32 and reproduces historical checkpoints byte-for-byte. Full150, another seed, and a weight sweep are forbidden.
21. **Current next action — preregister COMP-019:** run a matched larger-model comparison on the same no-thinking public prompts, frozen tiny/full datasets, mixed unit objective, and semantic teacher metrics. This determines whether E4B's remaining static generalization gap is capacity or data coverage before starting DAgger.
22. Pass a development gate, then preregister clustered power and one matched K≥8 comparison on the 12-seed/13-window fresh Nob final cohort. Extend it blindly first if 12 clusters are underpowered; do not query any final window beforehand.
23. If C1 passes, extend the same pipeline to Lagavulin and Sentries, then test corrected local GRPO.
24. Rebaseline whole Act I and proceed toward C3.

## Status dashboard

| Work item | Status | Exit condition / note |
| --- | --- | --- |
| Current v2 Nob behavioural baseline | outstanding (Gate C0) | COMP-001 at `8fc92b6` is historical pre-v2 evidence: 24/44 wins, reward -0.252. A matched `combat_public_v2` base arm is still required. |
| Public-state contract | complete | `combat_public_v2` adds the missing card type; binding/parser tests and repeated schema-2 replay validation pass. v1 is replay-only. |
| Full public combat observation | complete | Current binary: Nob 43/43, Lagavulin 43/43, Sentries 36/36 with repeated identical start signatures. |
| Turn-plan persistence experiment | not started | COMP-003 paired local-task result. |
| Search-best-action binding | complete | Non-mutating, action-parity-tested query with root statistics and provenance. |
| Search quality/privilege audit | complete | Current v2 COMP-008 audit rejects 5k: 42/60 agreement, 54/60 unanimity, 41/60 5k-eligible, versus 57/60 direct-50k-eligible. Historical v1 COMP-004 remains preserved. |
| Search root selector | complete | COMP-004A selects aggregated root visits: 7/10 versus 4/10 completed wins, with 3 versus 6 ambiguities. |
| Compact teacher target and loss masks | complete | Action-only JSON target is distinct from the default mixed format+action mask and experimental action-value-only mask. COMP-011 selects the mixed mask for current work. |
| Nob teacher dataset | complete | v2 aggregated-root 50k collection: 600/652 eligible micro-actions; 150/173 eligible first-per-turn states; deterministic tiny sets and 57-state held-out set built. |
| Tiny teacher fit control | complete | COMP-008 easy tiny32: 32/32. COMP-011 harder aligned tiny32: mixed 31/32 versus action-value-only 15/32. E4B/trainer memorization is established. |
| Full static-teacher generalization gate | complete — failed | COMP-012 fit 149/150 train but scored 28/57 development with worse NLL; behaviour correctly not run. |
| Halfway-checkpoint diagnosis | complete | COMP-013 step 1,500: train 102/150, development 27/57, NLL 1.183. Late training overfits, but early stopping is not sufficient. |
| Cyclic action-order diagnosis | complete | COMP-014: top-1 invariance 0.448/0.253 and probability TV 0.400/0.696; both checkpoints fail by a wide margin. |
| Direct move-choice and loss decomposition | complete | COMP-016 (`900bf5ee…be38` / `7fbae99e…9df9`) rejects the measurement-artifact explanation: tie-free direct choices match greedy JSON and retain the shortcut; format loss is effectively zero while action loss remains high. |
| Compute-matched order augmentation | complete — strict claim failed; component retained | Seed 2 improves development 28/57→32/57, invariance 0.362→0.629, TV 0.473→0.298, and passes all frozen carry thresholds, corroborating seed 1. It also worsens train fit and remains below the standalone stability gate. Result `configs/competence/comp_015_result.json`; no behaviour or final release. |
| No-native-thinking output comparison | complete — failed | COMP-017 (`1ebb217a…f124` / `0934aefd…fdbd`): immediate action-only is 57/57 valid and 26/57 correct; visible reasoning is 55/57 and 19/57, with two 512-token truncations and 8.37s versus 1.04s mean latency. No fights. |
| Explicit move-choice upweighting | complete — failed tiny gate | COMP-018 weight 8 scores 7/32 at both saved steps and learns a near-uniform state-independent digit policy. The exact current unit control scores 31/32 and reproduces COMP-011 adapter bytes. Result `configs/competence/comp_018_result.json`; no full150, added seed, behaviour, or final query. |
| Fresh final Nob cohort | complete and embargoed | 31/31 windows pass twice; 13 holdout windows form only 12 world-seed clusters. Freeze clustered power/inference and the one-time teacher-label procedure before release. |
| Nob DAgger | not started | COMP-006; run only if COMP-013/014 points to coverage rather than a cheaper positional/regularization failure. |
| Lagavulin/Sentries curricula | complete | Current v2/schema-2 cohorts replay 43/43 Lagavulin and 36/36 Sentries in two isolated passes; teacher data waits on Gate C1. |
| Corrected local GRPO | not started | COMP-007; real frozen old-logp ratios and nonzero groups. |
| Model-capacity comparison | in progress — **next action** | The cheap output, ordering, and weighting interventions are now isolated without a competence pass. Preregister COMP-019: one materially larger no-thinking model on the exact public teacher task, with a tiny fit gate before matched full-data training. If both models fail held-out transfer, prioritize broader-data/DAgger coverage. |
| Act I competence | not started | Gate C3. |
| Full-game competence checkpoint | not started | Gate C4. |
| Resume framing study | blocked | Blocked only on Gate C4 by deliberate project priority. |

## Decisions log

### 2026-07-22 — competence before framing

The active goal is to make the model play well. Use one neutral frame and defer risk-seeking/risk-aversion analysis until a competence checkpoint exists.

### 2026-07-22 — do not change models yet

Continue using Gemma E4B as the provisional student while building the positive-control learning path. Revisit model size only with matched teacher-labelled action data and a working trainer.

### 2026-07-22 — search is the first teacher candidate

Search's 43/43 Nob positive control justifies exposing individual action recommendations. Because search may use privileged simulator state, teacher labels require a public-state and stability audit before they are treated as ground truth.

### 2026-07-22 — the compact action target is the default competence output

Native reasoning may remain available at inference, but the competence target
is compact action JSON and private reasoning receives no loss. The original
phrase “action-only loss” was overloaded; the 2026-07-23 mask decision below
supersedes any reading that meant pure action-value-token supervision.

### 2026-07-22 — retain micro-actions initially

First add public-state completeness and persistent turn planning. Whole-turn executable action sequences are deferred until those cheaper interventions have been measured.

### 2026-07-23 — replay validation defines every local cohort

Historical action traces can diverge on the current simulator build even when
world seed and action bits are unchanged. Every local-task experiment must use a
versioned, isolated-subprocess replay-validation report and a passing-only
manifest. Exclusions are part of the cohort definition and must never be hidden
as runtime skips. The v1-build cohort excluded Nob `seed_128_r0_w0` and
Lagavulin `seed_53_r0_w0`, `seed_84_r0_w0`, and `seed_149_r0_w0`. Under the
current v2 binary, two full isolated validations pass all 43 Nob, 43 Lagavulin,
and 36 Sentries windows with identical public signatures. Preserve both cohort
versions; this same-binary repeat is not evidence of cross-build determinism.

### 2026-07-23 — first teacher arm is explicitly no-thinking

The first action-only teacher positive control uses a no-thinking generation
prefix and immediate action JSON. Under a native-thinking prefix, that same
target would also train the adapter to omit thought. A native-thinking teacher
arm therefore requires a real thought/channel prefix retained as masked causal
context and is a later, separately controlled experiment.

### 2026-07-23 — missing card type invalidates v1 competence artifacts

The stopped live COMP-002 transcript demonstrated that card type is necessary
policy information, not optional explanatory prose. Introduce
`combat_public_v2`; retain v1 only for historical replay. Do not resume COMP-002,
train COMP-005, or silently rewrite their files. Regenerate start signatures,
teacher rows, datasets, token reports, and tracked configs under v2.

### 2026-07-23 — aggregate root visits for prospective search labels

COMP-004A met its preregistered fail-closed behavioral gate. Prospectively,
search seeds vote by displayed-action-aggregated root visits. Raw native
winning-sequence selection remains in provenance. The v1 teacher datasets are
not retroactively relabeled; v2 is recollected from live states.

### 2026-07-23 — align train-state selection with the held-out question

COMP-009's dense every-micro-action dataset taught an all-index-0 policy while
the held-out set measured the first collected state of each turn. Prospectively,
`first_per_turn` chooses the earliest source decision for each window/turn from
the raw collection before confidence filtering; a rejected earliest row drops
the turn rather than falling forward to an easier later action. Display indices
are positional, not semantic classes, and the resulting 43/43/44/19/1 full150
distribution is less skewed rather than globally balanced.

### 2026-07-23 — mixed format-plus-action loss remains the default

COMP-011 isolated the realized loss mask on the same harder tiny32 and 20-pass
schedule. The default mixed mask fit 31/32 while the action-value-only override
fit 15/32 and badly reduced canonical completion likelihood. Keep required
format/channel/terminator tokens supervised alongside the action value; keep
private thought masked. This is a bounded optimization result, not a claim that
format tokens are intrinsically necessary or that action reweighting cannot
work.

### 2026-07-23 — cyclic action ordering is a demonstrated training failure

COMP-014 changed only the order and displayed indices of identical legal
actions, then mapped predictions back to semantic identity. Both saved
checkpoints fail the preregistered sensitivity rule by a wide margin, with the
later/overfit checkpoint substantially worse. Tie-aware metrics corroborate the
result. Before adding new states, DAgger, RL, or model capacity, isolate cyclic
menu augmentation against an unaugmented arm with identical source schedules,
updates, batch size, mixed mask, optimizer, and token exposure. This decision
does not claim arbitrary-permutation invariance or guaranteed behavioural gain.

### 2026-07-23 — cyclic augmentation reduces sensitivity but redistributes the shortcut

COMP-015 seed 0 completed a byte-identified, exact-source-schedule control pair.
At the fixed step-1,500 endpoint, augmentation substantially improves cyclic
probability stability and all-variant NLL, but unpermuted development agreement
falls from 21/57 to 17/57 and semantic top-1 invariance remains only 0.149.
Assigned-position results explain the mismatch: the control is correct 36/36
when the teacher is displayed at position 0 and zero times elsewhere; treatment
is correct 43/46 at position 3 and poor at the other positions. This is a
redistributed positional prior, not learned semantic equivariance. Do not run
seed-0 behaviour. The strict all-three-pair primary success rule is already
impossible. The initial decision to finish the frozen replicates was superseded
by the sequential-seed amendment below once a complete second pair existed.

The result also exposes two measurement/objective sharp edges. Whole candidate
sequence scores are nearly quantized to 1/8 and create many exact ties, although
tie-aware metrics remain negative. More importantly, each arm supervises eight
required-format tokens for every action token (24,000 versus 3,000), so low
aggregate console loss can hide weak action fit. Before changing coverage,
DAgger, capacity, or RL, add float32 direct action-digit/greedy scoring and
decompose scheduled-row cross-entropy into format and action components. If the
failure survives, test explicit action-token upweighting while retaining the
format scaffold; do not infer that pure action-only loss is sufficient from
this result.

### 2026-07-23 — direct choice scoring confirms the shortcut and selects weight 8

COMP-016 froze its measurement plan before loading the model and pinned the
exact training-era Gemma snapshot. The scorer casts the normalized hidden state
and tied output weights before the full-vocabulary float32 projection rather
than casting already-quantized logits. Across train150 and development57 it
finds no exact direct-action ties. Deterministic JSON and direct top-1 agree on
the development counts exactly: 21/57 control and 17/57 augmentation. The
direct cyclic result also preserves the mechanism diagnosis: invariance
0.000→0.149 and mean probability TV 0.484→0.197.

The fixed-checkpoint loss split finds format/action mean token NLL
0.0/1.293 for control and 2.3e-8/1.495 for augmentation. The zero is a
float32-resolution result, not a claim of mathematical probability one.
Effectively all measurable loss is on the move choice, while the existing
unit-weight objective assigns eight supervised positions to already-solved
format and one to the move. Keep the format scaffold because COMP-011 showed
that deleting it fails; prospectively test one action weight of 8 so format and
action contribute equal total mass. Do not sweep weights or reinterpret this
diagnostic as evidence that pure action-only training works.

### 2026-07-23 — training seeds are sequential evidence, not an up-front tax

During fast development, a training seed is useful when it can change the next
decision. COMP-015 seed 0 already fails the behaviour gate, its strict
three-pair success claim cannot pass, and COMP-016 rejects the main measurement
alternative. Training a third treatment would therefore delay the selected
output-mode and loss-weighting tests without changing the decision to move on.
Stop that treatment before weights, score the already-complete seed-1 pair, and
retain the completed seed-2 control as unpaired provenance only. The exact
amendment is `configs/competence/comp_015_fast_iteration_stop.json`, SHA-256
`88755f50944d4a0c9b4543e5f089c81cca215e8bd157734750f654096f929774`.

Use a sequential policy prospectively: one tiny screening seed; one full
development seed only after the tiny gate; a second training seed only for a
promising or genuinely ambiguous full result. A single failed seed can reject
an idea for fast iteration, but a single positive seed cannot establish that
the method works. Add frozen independent training repeats before a robust
positive claim, final-cohort release, or checkpoint promotion. Within-policy
rollout repeats answer a different question—how variable generation and game
outcomes are—and do not replace independent training seeds.

### 2026-07-23 — seed 1 triggers, rather than merely consumes, another seed

The cheap score of the already-complete COMP-015 seed-1 pair changes the
decision. Control→augmentation moves direct development agreement
24/57→31/57, greedy agreement 23/57→31/57, cyclic invariance
0.335→0.824, TV 0.432→0.325, and cyclic source-macro agreement/NLL
0.378/2.173→0.564/1.342. Train fit falls and development NLL worsens
slightly; the treatment also misses 32/57 by one and TV≤0.20 by a wide margin.
This does not rescue COMP-015, but it is genuinely promising and opposite to
seed 0 on held-out choices. The sequential rule therefore licenses only the
treatment matched to the already-complete seed-2 control. Freeze that change
under `c0d621d9…0ac8`, close COMP-015 afterwards, and add no further seed.

### 2026-07-23 — seed 2 closes COMP-015 and retains augmentation only as a component

The ambiguity-triggered seed-2 treatment passes every prospectively frozen
carry threshold relative to its completed control: direct development agreement
improves 28/57→32/57, invariance rises 0.362→0.629, TV falls
0.473→0.298, source-macro agreement rises 0.371→0.503, and source-macro
NLL falls 2.073→1.262. This corroborates seed 1's useful development and
invariance directions. It does not rescue the original strict claim: seed 0
regressed, seed 2 train fit falls 113/150→81/150, and no treatment meets both
invariance ≥0.75 and TV ≤0.20. Close COMP-015 with no behaviour and no final
release. Retain cyclic augmentation only for a separately preregistered
combination after some other intervention independently clears its gate;
COMP-018 later rejects fixed move-choice weight 8.

### 2026-07-23 — visible JSON reasoning is worse than immediate action JSON

COMP-017 changes only the requested output schema while native thinking remains
disabled. The base model's immediate-action arm is harness-valid on 57/57 and
teacher-correct on 26/57. Visible reasoning is valid on 55/57 and correct on
19/57, hits the 512-token cap twice, averages 217 rather than 16 completion
tokens, and takes 8.37 rather than 1.04 seconds. It fails every static fight
gate. Do not run reasoning-mode fights or train rationale text; retain immediate
action-only JSON. The report/result are `69fc1c5e…5a77` /
`0934aefd…fdbd`.

### 2026-07-23 — fixed weight 8 equalizes loss mass but destroys tiny-task learning

The new MLX-only strict-teacher option gives action tokens a numeric relative
weight without deleting required-format supervision and normalizes by total
weight mass. On the exact tokenizer, tiny32 has 256 format mass and 256 action
mass; full150 has 1,200 and 1,200. Non-unit weights are rejected outside the
strict action-mask path, and provenance records raw counts, both masses, and
normalization. Unit weight deliberately omits numeric weights and retains the
historical boolean-mask computation; the exact seed-2 augmentation preparation
regenerates byte-identically (`91177b9b…ad3`). COMP-018 (`5512cc8c…4d54`)
therefore tested weight 8 once behind harder-tiny32 and conditional full150
gates.

The treatment fails at 7/32 with a near-uniform, state-independent move-digit
distribution, even though final JSON is valid/legal on 32/32. Its halfway
checkpoint is already in the same basin. A prospectively frozen post-failure
unit control reaches 31/32, NLL 0.220, and reproduces the historical COMP-011
checkpoint bytes exactly under current code. Token alignment, gradient scaling,
and compiled numeric-weight learning checks all pass. Reject weight 8, do not
run full150, add a seed, or sweep weights. The surprising lesson is that
near-zero endpoint format loss does not make early format supervision
optimization-irrelevant; the wrapper tokens are useful scaffolding in this
setup.

### 2026-07-23 — final Nob inference is clustered by world seed

The embargoed 13 final Nob windows contain only 12 distinct world seeds. Treat
world seed as the inferential unit, keep both seed-604 windows in one cluster,
freeze power before release, and extend the blind cohort first if 12 clusters
are inadequate. Collect final teacher labels once on frozen base-visited states
and never feed them back into checkpoint or intervention selection.

### 2026-07-23 — train fit is not permission to run behaviour

COMP-012 nearly memorized full150 but failed both the 32/57 development
agreement threshold and the no-worse-than-base NLL guardrail. The behavioural
gate was honored: no game episode was run. Low mixed-token training loss and
149/150 train action accuracy establish adapter fit, not competence.

### 2026-07-23 — development and final Nob cohorts are now separate

The original 57 teacher-scored states have been inspected across several
experiments and used to choose objectives/schedules. Call them development data;
do not select on them and claim the same windows as an untouched final test.
The new seed-600–659 cohort reserves 13 world-seed-grouped holdout windows. Two
isolated v2 replays pass all 31 source windows with identical public signatures.
Keep the 13 final windows free of teacher queries, action scoring, and model
rollouts until one preregistered matched behavioural comparison.

## Changelog

### 2026-07-23

- Completed COMP-018 under `5512cc8c…4d54`. Fixed action weight 8 fails the
  harder-tiny gate at 7/32 and a near-uniform action distribution, so full150
  was not run. The matched current unit control reaches 31/32 and reproduces
  historical step-320/final adapter bytes exactly; implementation audits find
  correct targets and gradients. Reject weight 8 without another seed or a
  weight sweep. Result `configs/competence/comp_018_result.json`.
- Closed COMP-015 after the single ambiguity-triggered seed-2 treatment.
  Development improves 28/57→32/57 and all frozen carry thresholds pass, but
  train fit regresses and the standalone invariance/TV gate still fails. The
  strict claim is rejected; no behaviour or final data ran. Cyclic augmentation
  is retained only as a possible later component. Result
  `configs/competence/comp_015_result.json`.
- Completed COMP-017 under frozen config `1ebb217a…f124`. Immediate action-only
  no-native-thinking beats visible JSON reasoning 26/57→19/57, while the
  reasoning arm is less valid, reaches the 512-token cap twice, and is roughly
  eight times slower. Result `0934aefd…fdbd`; no fights or final data ran.
- Scored the complete COMP-015 seed-1 pair with genuine float32 direct actions
  and every nonzero cyclic rotation. Unlike seed 0, augmentation improves
  development 24/57→31/57 and invariance 0.335→0.824, but remains below
  the action/TV gates. This triggers exactly one matched seed-2 treatment under
  amendment `c0d621d9…0ac8`; no further COMP-015 seed or behaviour is allowed.
- Added fail-closed MLX move-token weighting for strict teacher data. Weight 8
  retains format supervision and gives exact equal format/action mass on both
  tiny32 and full150. Unit weight preserves the historical boolean-mask path;
  the planned seed-2 preparation is byte-identical to the stopped preparation.
  COMP-018 is frozen as `5512cc8c…4d54`.
- Changed development-time seed allocation from a fixed three-pair grid to a
  sequential policy under amendment `88755f50…9774`. The seed-1 control is
  complete, giving a second complete pair to score. The seed-2 control also
  completed but is retained as unpaired provenance only; its treatment was
  stopped before any model weights or checkpoint were saved. Further training
  repeats are now conditional on promising or ambiguous early signal, while
  robust positive claims still require independent frozen repeats.
- Completed COMP-016 under frozen config `900bf5ee…be38`. Added genuine float32
  direct action-digit scoring, exact greedy-JSON comparison, and fixed-checkpoint
  format/action loss decomposition. The four reports are content-addressed in
  result config `7fbae99e…9df9`; no final states or behaviour were queried.
- Hardened retained local model work against mutable Hugging Face repository
  names. Diagnostics and strict search-teacher MLX training now resolve the
  exact cached revision `eec12d0899edea9b738ab1009af9159cdfd70d71`,
  content-address its files, and fail closed on mismatch.
- Stopped COMP-002 after 71 decisions when native reasoning exposed the missing
  card-type alias: Attacks were described as Enrage-triggering Skills while
  Defend was described as safe. Preserved the partial output as invalid evidence.
- Implemented versioned `combat_public_v2` card-type labels across state, actions,
  selections, structured cards, search descriptions, replay normalization, and
  coupled analysis/glossary parsers while retaining v1 compatibility.
- Rejected the untrained COMP-005 v1 dataset and moved it intact to an explicitly
  invalid archive; no model had been loaded or scored on these data.
- Completed COMP-004A: aggregated root visits passed over native winning-sequence
  selection (7/10 vs 4/10 completed wins; 3 vs 6 ambiguities). Wired this rule
  into prospective teacher collection with explicit abstention provenance.
- Regenerated `source.validated_v5.json` cohorts with signature schema 2 and the
  current v2 binary: Nob 43/43 (32/11), Lagavulin 43/43 (33/10), and Sentries
  36/36 (27/9). A second isolated pass produced identical per-window public
  signatures. Historical v1 exclusions remain preserved as build-specific facts.
- Completed the v2 aggregated-root teacher audit and dense Nob collection. Direct
  50k accepts 57/60 held-out first-per-turn states and 600/652 train states;
  5k accepts only 41/60 held-out states and remains rejected. Built the 600-row
  action-only dataset, its deterministic tiny32 prefix, and a 57-row held-out
  teacher-action set without loading a model.
- Completed COMP-008's easy-tiny32 positive control at 32/32, establishing that
  E4B, the current LoRA attachment, and the mixed mask can memorize repaired
  teacher actions. Its 29/57 held-out top-1 result had worse NLL, so the static
  teacher chain did not pass through to behaviour.
- Completed COMP-009. Three passes over all 600 every-micro-action states
  produced an all-index-0 policy and failed held-out agreement (21/57 versus
  base 22/57); no behavioural evaluation was run.
- Added `first_per_turn` source projection. It operates before confidence and
  public-hash filtering and never substitutes a later action when the earliest
  row fails. Prospective builder hardening now distinguishes missing/ambiguous
  references from true public aliases and rejects conflicting duplicate source
  coordinates; frozen COMP-010 artifacts remain byte-identical and retain their
  historical two-row accounting label.
- Completed COMP-010/011. Ten-pass action-value-only training failed 11/32.
  Under a matched 20-pass comparison, mixed format+action reached 31/32 and
  action-value-only reached 15/32, selecting the mixed mask for COMP-012.
- Preregistered and started COMP-012: final-only scoring of a 20-pass mixed-mask
  adapter over 150 aligned first-per-turn train states, with strict train,
  held-out likelihood/agreement, diversity, and invalid-row gates before any
  behavioural rollout.
- Completed COMP-012: 149/150 train but 28/57 development, below the 32/57
  threshold, with NLL worsening 1.777→2.572. Honored the gate and ran no
  behavioural episodes.
- Completed COMP-013's halfway-checkpoint score: step 1,500 fits 102/150 train
  and 27/57 development with NLL 1.183. The next 1,500 updates fit 47 more train
  rows for one development choice while sharply worsening calibration. Early
  stopping alone is not a sufficient policy solution.
- Completed COMP-014 over both checkpoints and all 221 nonzero cyclic rotations
  per checkpoint. Step-1,500/3,000 semantic top-1 invariance is 0.448/0.253 and
  probability TV is 0.400/0.696; the frozen sensitive branch selects COMP-015.
  Post-run CLI hardening now releases each checkpoint before loading the next
  and atomically publishes reports; the completed report retains its exact
  pre-hardening code hash and is byte-unchanged.
- Froze a fresh final Nob cohort from disjoint source seeds 600–659. Both
  isolated validations pass 31/31 with identical signatures; 13 holdout windows
  in 12 world-seed clusters are embargoed from teacher/model inspection.

- Audited COMP-001 provenance before rerunning it: `base_vllm_t07_k4` used the
  tier-1 serializer fixes at `8fc92b6`, so a duplicate run was initially
  avoided. The later COMP-002 card-type discovery created `combat_public_v2`;
  COMP-001 is therefore now correctly classified as a historical pre-v2
  baseline and Gate C0 again requires a matched v2 base arm.
- Specified and implemented `combat_public_v1`, including public pile multisets,
  upgrade/current-cost data, relic/potion state, per-turn history and selection
  context, with an explicit legacy control and hidden-order exclusions.
- Added byte-level competence-interface provenance, a non-mutating search teacher
  query, raw audit/dataset/report tooling, and public-state action identity.
- Implemented tokenizer-aware compact-target/mixed action masking across the
  MLX/TRL SFT, PG, and GRPO paths, plus token accounting and a tiny
  adapter/reload smoke test. Later added an explicit MLX action-value-only
  override without silently changing historical manifest semantics.
- Added Lagavulin and Sentries local curricula and an isolated-subprocess
  validator. On that historical simulator build, frozen passing cohorts were
  Nob 42/43, Lagavulin 40/43, and Sentries 36/36; all exclusions and source
  hashes are persisted. The later v5 cohorts supersede these counts.
- Preregistered and started COMP-002 and COMP-004 on the replay-validated Nob
  cohort. The held-out sample changed from 11 to 10 before either result was
  inspected because `seed_128_r0_w0` does not replay on the current build.
- Closed additional public-interface aliases: correct stance/resources, all
  active player/enemy statuses (including bitfield-only Corruption), indexed
  same-name potion targets, and draw-order-independent Secret Technique/Weapon
  selection menus. The legacy text path remains byte-compatible.
- Added complete policy-visible start signatures and validation schema v2. On
  that historical simulator build, the frozen cohorts were Nob 42/43,
  Lagavulin 40/43, and Sentries 36/36, with
  exact public state/action/encounter signatures, source JSONL hashes, and the
  actually loaded simulator binary hash enforced by every replay consumer.
- Completed the corrected COMP-004 independent-seed audit. It rejects 5k;
  31/56 rows qualify at 5k and 38/56 qualify when labelled directly at 50k under
  the same hidden-action identity rule.
- Completed COMP-005 train-state collection: 493/652 examples survive the direct
  50k hidden-order filter. Froze the deterministic tiny32 subset and 640-update
  positive-control schedule before loading/scoring a model on these data.

### 2026-07-22

- Created the competence-first plan.
- Recorded the completed whole-run RWR and Nob self-BC nulls.
- Recorded the 43-window search-agent Nob positive control.
- Established the staged gates C0–C4, experiment registry, and immediate execution order.
