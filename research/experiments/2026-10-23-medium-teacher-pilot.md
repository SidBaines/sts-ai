# Learning decisions with medium-effort reasoning

**Status:** planned 23 October 2026; on hold since 25 October. **Verdict:** not run. A review found that this design would probably end inconclusive, and [a staged replacement](../brainstorms/2026-10-25-getting-the-model-to-learn.md) awaits Sid's decisions.

## Question

Does single-decision teacher feedback improve held-out action choices while the
model develops its own reasoning, and does that improvement transfer to games?
Approach: [teacher feedback followed by full-game outcomes](../strategies.md#teacher-feedback-rl-followed-by-full-game-outcomes).

## Design

Medium effort is fixed for all pilot training and checkpoint evaluations.
Sid selected medium and requested an updated plan on 23 October. The remaining
defaults below are proposals for the pilot design.
Keep the released environment and current public-state/menu prompt fixed.
No target reasoning, reasoning SFT, reasoning-style reward or history change.

### Proposed starting configuration

| Choice | Proposed setting |
|---|---|
| Model and environment | Pinned K2-Horizon-0.9B with a fresh adapter, and env-v0.1.5.1; smoke-test weights are not the pilot starting point |
| Adapter | One shared LoRA for both phases; rank 16, alpha 32 |
| Teacher reward | Frozen champion probability of the chosen action; invalid/capped model responses receive -1 |
| Sampling | Temperature 0.6, full support: top-p 1, with top-k truncation disabled |
| Output cap | 4,096 generated tokens; record caps and do not silently raise it during training |
| Samples | Eight responses per state, four states per iteration |
| Rollouts | BF16 candidate; fast serving with tested adapter refresh/cache handling, or validated HF fallback |
| HF generation fallback | Generation microbatch eight; optional HF prompt reuse stays off |
| Updates | SDPA, gradient checkpointing, microbatch one; accumulate 32 responses per optimizer step |
| Objective | One update epoch, learning rate 0.00001, clipping 0.2, KL coefficient 0.02 |
| Token normalization | Sum generated-token contributions divided by the fixed constant 2,048, for policy and KL |

The output cap and loss denominator are different quantities. Fixed token
normalization avoids dividing by the model's realized response length; it does
not make the whole group-normalized/clipped objective unbiased. Freeze actual
settings and run configuration after preflight, before the first pilot update.

### Rollout and update preflight

Use one bounded engineering pass, with at most one GPU-hour for its checks.
Compare the same states, sampling settings, output cap and current adapter.
Measure generation, old/reference scoring, updates, adapter refresh and saving
together; standalone serving rates are not the pilot's training throughput.
Verify that generated responses use the intended adapter, and that cached
prefixes cannot survive a change of adapter weights under a stale identity.
Check sampling-versus-recomputed token probabilities and unchanged-weight
clipping effects across both phases. Set and record numerical acceptance
criteria before choosing the backend; exact greedy-text equality is not the gate.
Exercise a long-completion update near the proposed cap and verify memory,
finite gradients, checkpoint reload and resume. Adopt faster serving only if
these checks pass and its end-to-end gain justifies the integration. A validated
HF path is the fallback; failed correctness checks must be resolved before RL.
Keep the chosen backend, precision and cache policy fixed throughout this pilot.

### Data, baseline and checkpoint evaluations

Prepare data on CPU alongside independent rollout engineering. Select the most
recent validated non-LLM champion available at preparation, then freeze it for
all labels and comparisons. Verify observation/menu compatibility before labelling.
Use only public observations; retain the raw legal menu and record the
teacher's excluded actions separately. Those actions remain
legal with teacher probability zero. Teacher labels never enter model messages.
Target 4,096 training states, 512 validation states and 512 untouched test states,
with whole-world separation and combat/out-of-combat coverage across all acts.
Keep previous timing/mechanics worlds out of the untouched test panel.
Audit coverage and freeze actual counts and identities before model updates;
the 128-state timing dataset is not the pilot training corpus.

Evaluate the unchanged medium-effort model on validation states and a proposed
16-world paired game panel before training. Freeze evaluation sampling and
worlds; use greedy decoding at medium effort on the same adapter-enabled
backend for baseline and candidate. Batch independent calls where possible,
and retain every response.
Measure validity/caps, teacher probability, top-action agreement, reward-group
variance, generated-token counts, clipping/KL and end-to-end training rate.
Assess validation at baseline and fixed checkpoints around one, three, five
hours and the end of the learning allocation. Intermediate evaluations and
checkpoint overhead count inside that allocation.
Use a fixed, preselected subset of 32 validation states for qualitative reasoning
comparison; retain all validation traces, including regressions and empty output.

Choose one checkpoint by highest validation teacher probability, excluding an
observed validity decline greater than one percentage point versus the base.
Resolve ties in favor of the earlier checkpoint. Then freeze the candidate and
compare it with the base on untouched test states, using paired differences
and 95% intervals clustered by source world. A useful teacher-proxy result is
at least +0.02 mean probability with the interval above zero and no observed
validity decline greater than one percentage point; otherwise report the result
as worse, no clear difference or inconclusive as supported by the interval.
This rule is proposed before learning; it is not a game-strength promotion rule.

Compare the base and frozen candidate on the same game panel. Report wins,
defeats, model failures, simulator faults and truncations separately, with sample
sizes and paired uncertainty. Set a smaller panel during preflight if needed
for the budget, before inspecting game outcomes. Preserve unfinished attempts;
this small panel is exploratory and cannot establish a precise win rate.
Reasoning length or coherent prose is never a checkpoint-selection reward.

### Budget and stopping

Proposed maximum: ten new GPU-hours total, including one for preflight, one for
baseline evaluations, seven for training with intermediate checks, and one for
final test/game evaluation. Previously completed speed checks are separate.
Freeze an affordable schedule using measured end-to-end throughput before RL;
reserve final evaluation time rather than spending the whole cap on updates.
Start with a short initial training block. If rewards are flat across almost
all groups, losses become nonfinite, numerical mismatch causes material clipping
before updates, or time/memory exceeds the allocation, preserve outputs and
investigate before continuing. No silent reward, backend or cap changes within
the comparison. A zero-variance group can also mean an already matched teacher
action; inspect that distinction rather than treating every flat group as failure.

## Results

Not run. The completed [speed and mechanics study](2026-10-23-llm-throughput.md)
is supporting evidence, not a pilot learning result.

## Conclusion

The immediate goal is evidence of learning and a record of reasoning changing.
A teacher-proxy gain does not establish improved game strength or faithful reasoning.

## Next

Complete preflight and baseline/data preparation, then freeze the proposed pilot
configuration. After the pilot, review complete-game frequency and win/loss
diversity before designing terminal-outcome RL. That
stage must settle invalid-output handling and incomplete-group selection bias;
groups with identical terminal rewards have zero outcome-policy advantage,
so nearly all-loss groups would provide very little learning signal. Separate
phase adapters, prompt-history changes and high-effort follow-ups are later
controlled experiments.

## Log

- 23 October: Sid selected medium effort and requested the next-step plan.
- 25 October: put on hold after review. The unchanged model scores no better than always choosing the first menu option; combat teacher rewards are mostly near-ties, which group standard-deviation scaling amplifies; and the plan allows about 150 updates at a low LoRA learning rate. The proposed replacement starts with an overfit test on decisive out-of-combat states.

Code: existing src/sts_llm/ at commit 4d30a8b; pilot launch configuration pending.
Data: no pilot run or scientific training corpus has been created.
