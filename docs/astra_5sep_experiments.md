# Astra experiments — 5 September 2026

Branch: `astra-5sep`. Experiments ran 22:08–23:35 UTC on 5 September 2026.
This exploratory session preserves production interfaces and frozen artifacts.
Raw outputs live under `data/experiments/astra_5sep/`.
The user-requested 26B extension continued beyond the initial hour.

## Questions and protocol, recorded before model results

1. On new diagnostic worlds 1000–1031, how much do elementary OOC rules and
   combat search budget affect floor? Paired random/heuristic × 50/500 search
   simulations, one rollout/world/arm, all starts accounted for, per-process
   timeout. Bootstrap over worlds; report errors and wins as well as floor.
2. How often do those trajectories encounter missing public information?
   Observe prompts and public map DAGs without changing policy decisions.
3. Does the base model or an RL checkpoint make different decisions when exact
   legal JSON is enforced? Freeze 133 OOC prompts from random/search50 traces,
   stratified by screen (up to16 each, selection seed905). Compare free and
   finite-menu constrained decoding on exactly the same observations. Validity
   under constraints is guaranteed by construction, not evidence of competence.
   Base, pre-instability RL (`iter_1/adapter`), and later RL (`iter_5/adapter`)
   are selected before these probe results. Greedy generation, no thinking;
   `action_text` and compact `action_only` contracts. Additional live rollouts
   will assess selected interventions; any adaptive selection remains exploratory.
4. Can cheap OOC rollout-value labels provide an informative diagnostic?
   Replay each frozen prompt, verify the exact public frontier, try legal actions,
   and continue with fixed simple policies. Label values are continuation-policy
   dependent and privileged, not optimal values or policy-visible information.
5. How much do format-dead episodes and extreme KL estimates influence the saved
   RL updates? Reconstruct existing trainer row order and analyze coefficient
   allocation and counterfactual group baselines. This is descriptive sensitivity
   analysis, not an alternative unbiased PG estimator.

The embargoed final Nob cohort and historical final eval protocol are untouched.
New diagnostic worlds become development data through this session.

### Follow-ups selected during the session

The initial baseline and frozen-state results motivated three extensions:

- Closed-loop map-only and map-plus-campfire lookahead, first on worlds
  1000–1007, then a fixed replication plan on 1100–1131. All three replication
  arms and all 32 worlds were fixed before that replication launched.
- The same frozen prompts with public glossary/cost/map information appended,
  reversed and reindexed menus, and temperature 0.7 generation. These are
  exploratory diagnostic interventions, not additional independent test sets.
- A separate cohort of all 56 fatal decisions from GRPO iteration 6. The broad
  cohort excludes singleton menus, whereas 53/56 historical failures have only
  one legal action. Targeted replay uses the original public states and menus
  with reconstructed first-attempt prompts; it does not reproduce the original
  final retry prompt, which contains an additional 55 tokens.

## What changed my view

**A useful OOC teacher is feasible, but the current RL curve still does not
establish improved decision-making.** Short simulator continuations produced a
large, replicated improvement in map/campfire play. Their action values depend
strongly on the continuation policy and access hidden future outcomes. This is
evidence for developing a supervised competence rung, not for treating a single
playout's preferred action as public-state ground truth.

**Format failure is partly an avoidable harness burden.** Almost all historical
fatal decisions were map screens and 53/56 presented no actual choice. Sampling
also exposes deterioration that greedy static probes obscure. Legal decoding
can remove malformed outputs, while leaving menu-order sensitivity and strategic
quality as separate questions.

## 1. Fresh-world baseline factorial

Worlds 1000–1031, one game per world and arm, Ironclad at Ascension 0 through
Act 3, search-controlled combat, 50 or 500 search simulations with a 2× boss
multiplier. The heuristic takes resources, rests below 55% HP,
takes the first card, and otherwise chooses the first action; it is deliberately
elementary, not an expert policy. All 128 planned games finished as losses, with
no invalid outputs, timeouts, or `undefined_behavior_evoked` flags. This flag
check is not a comprehensive phantom-power audit.

| OOC policy | Search 50: mean final floor | Search 500: mean final floor |
|---|---:|---:|
| Random | 13.97 | 16.59 |
| Simple heuristic | 17.44 | 23.44 |

Paired world-bootstrap 95% intervals:

- Heuristic minus random: **+3.47 [0.97, 5.88]** at search 50 and
  **+6.84 [3.19, 10.66]** at search 500.
- Increasing combat search: **+2.63 [1.16, 4.09]** under random OOC play and
  **+6.00 [3.13, 9.09]** under the heuristic.
- The interaction estimate, +3.38 [-0.03, 7.00], remains inconclusive.

The combat controller materially changes the floor scale. Freeze its budget
when comparing OOC policies; stronger search does not eliminate OOC leverage.

## 2. Closed-loop OOC lookahead

At each eligible branching map or campfire state, replay the exact prefix, try
every legal action, and finish the game with the simple heuristic. Choose the
action attaining the highest final floor, with deterministic first-action ties.
All other decisions use the heuristic. Combat search remains 50.

This is **privileged fixed-world lookahead**: it can see consequences through
simulation that a player cannot know. Labels are saved separately from public
observations. It is a feasibility test for a teacher and an estimate of available
OOC improvement under this controller, not a fair public-policy benchmark.

The eight-world pilot improved mean floor by +5.50 for map-only search
([1.00, 10.63]) and +4.25 for map plus campfire search ([0.50, 9.00]). The
subsequent 32-world replication gave:

| Policy | Terminal / planned | Mean floor among terminal games | Paired delta vs heuristic |
|---|---:|---:|---:|
| Heuristic | 29/32 | 17.90 | — |
| Map lookahead | 30/32 | 27.53 | +10.32 [7.46, 13.36], 28 pairs |
| Map + campfire lookahead | 31/32 | 30.10 | +12.69 [9.21, 16.45], 29 pairs |

All 96 starts are accounted for: 90 terminal losses and six simulator timeouts.
The arm means use different completed cohorts; the paired contrasts are the
appropriate comparison. No hard replay errors or `undefined_behavior_evoked`
flags were reported among completed runs; phantom-power absence was not audited.

For an all-32-world sensitivity analysis, define the bounded outcome
`min(final_floor, 50)` and allow missing outcomes anywhere in [0, 50]. The
identified mean-difference ranges are **[+4.91, +12.72]** for map-only and
**[+8.25, +14.50]** for map plus campfire. These are missing-outcome bounds,
not confidence intervals. They show the cohort's mean advantage survives the
most adverse assignment of those missing outcomes.

At map-only visited states, comparing search choices with what the heuristic
would choose at those same states gives 13 more events, six more shops, 15 fewer
monsters, four fewer rests, and no net change in elites. Twenty-five changed
choices select a different monster node, so downstream routes matter. Campfire
search changed smith to rest 43 times and rest to smith three times. These are
descriptive policy decisions, not universal Slay the Spire advice or a direct
measure of a learned risk trait.

## 3. Fixed-state teacher and model probes

The cohort contains 133 non-singleton decisions: five boss-relic states and 16
each from eight other OOC screen types, drawn from the random/search-50 traces.
It was frozen before model results. For 657 legal actions, attempt one heuristic
and three independently seeded random continuations: 2,628 labels in total.
2,612 reached terminal outcomes. All 16 failures belong to one four-action
state. That failed state is excluded consistently across comparisons, leaving
**132 complete states**.

Heuristic continuation distinguishes actions in 69/132 states; averaged random
continuation distinguishes 96. Of the 63 informative under both, the best-action
sets are disjoint in **25/63 (39.7%)**. A teacher should therefore average or
check multiple continuation policies and abstain on unstable targets.

The models are the pinned local Gemma E4B base, `iter_1/adapter` (the checkpoint
used for the next iteration's early peak), and `iter_5/adapter` (used for the
bad iteration-6 rollouts). No new weights were trained. An exhaustive tensor
comparison found all 665 shared active language-model tensors identical between
the two locally cached base snapshots; removed shared-KV duplicates and metadata
changes do not constitute an active-weight mismatch.

Greedy generation uses no thinking. Free generation has a 256-token cap; the
constrained cap is the longest candidate token length plus one. The constrained decoder
permits only exact legal JSON strings through a token trie. It makes tokenwise
greedy choices; it does not compute the most likely complete candidate string.
Static probes make one attempt, without retries. Regret below is measured in
final-floor units relative to the best privileged continuation for that state.

| Greedy semantic-text arm | Valid / 133 | Heuristic-continuation regret | Random-continuation regret |
|---|---:|---:|---:|
| Base, free | 130 | 2.147 | 1.013 |
| Base, constrained | 133 | 1.674 | 1.058 |
| Early RL, free | 133 | 2.333 | 1.217 |
| Early RL, constrained | 133 | 1.773 | 1.146 |
| Later RL, free | 133 | 2.576 | 1.311 |
| Later RL, constrained | 133 | 2.447 | 1.369 |

Regret conditions on valid responses and complete teacher labels: base-free has
129 evaluable choices; other listed arms have 132. Read validity alongside it.
Paired world-mean heuristic regret differences relative to constrained base are
+0.14 [-0.24, 0.62] for early RL and +0.43 [-0.24, 1.21] for later RL
(positive is worse). Neither demonstrates a policy-quality gain or a reliable
degradation. Compact index-output controls likewise show no uniform advantage.

At temperature 0.7, single-attempt validity is **128/133 base, 131/133 early RL,
and 114/133 later RL**. Later RL fails on eight of 16 map decisions and on several
other screen types. These are one sampled completion per state, not estimates
from repeated independent draws of each prompt.

Reversing and reindexing the same menus, then mapping predictions back to the
original semantic actions, preserves only **63/133 base choices (47.4%)** and
**64/133 early-RL choices (48.1%)**. On maps, preservation is 5/16 and 6/16.
Some switches are value ties: for base, 39 of 69 changed choices on complete
teacher states tie under heuristic continuation, and 23 tie under random
continuations. Even among informative states, semantic action consistency is
only 39.1%/42.7% under the two teacher definitions. Semantic text has not solved
menu-order fragility.

## 4. Public-information coverage and enrichment

Across the 128 baseline trajectories, inspect 8,438 decisions and their public
state/menu surfaces:

- All 351 ordinary event decisions are anonymous, encountered in 112/128 games.
- 893/893 offered relic instances and 1,925/1,939 potion offers lack supplied
  tooltips; 10,901/11,991 card offers lack a printed cost mention.
- None of 6,194 non-map decisions includes map information.
- 17/753 branching map states contain options indistinguishable by immediate
  room and aggregate counts despite different feasible downstream sequences.
- 1,083/3,158 map choices have aggregate elite counts exceeding the maximum on
  any one realizable route: the current number is a branching total, not a path
  count. It should not be interpreted as elites the player can encounter.

These are repeated offer/visitation counts, not independent model trials.
The [repository review](repository_review_2026-09-05.md) relates these omissions
to public game information and references the external rules/UI sources.

An additive experimental observation supplies public item definitions, printed
card costs, and the full public map DAG on map screens. It changes 65/133 prompts
while preserving all actions and indices. It does not fix anonymous events or
provide maps away from map screens. Base semantic choices change on 14/133 states,
but value estimates do not consistently improve under either continuation
policy. This small static probe supports **neither** declaring the missing
information harmless **nor** assuming that adding more prompt text alone will
produce competent play. A versioned complete interface needs a training and
behavioural evaluation alongside it.

## 5. Existing RL credit and failure audit

All 6,995 saved update rows were matched to their tokenizer lengths and
advantages. In iteration 6, the dataset omits 56 fatal responses and 485 retry
responses, while retaining 543 earlier decisions from format-dead episodes,
186 with positive advantages. This means the failed output is not necessarily
the action receiving the penalty, and relative baselines can reward earlier
choices on trajectories that later fail.

A diagnostic recalculation excluding format-dead episodes from group baselines
changes the sign of 105/455 retained ordinary-death rows and sends another 23
to zero. Their summed advantage changes from +326.14 to -5.99. This isolates
sensitivity to baseline composition; it is not an unbiased alternative policy
gradient estimator or proof of the instability's cause.

Format tokens receive 36–42% of absolute coefficient allocation under the
existing action mask. This is token/coefficient accounting, not a gradient-norm
measurement. One short response contributes 79.4% of the reported iteration-0
KL estimate and one contributes 82.4% in iteration 6. Those examples concern
short reward choices, not long reasoning. The mean KL should therefore be
accompanied by quantiles, maxima, and influential-row diagnostics.

The complete fatal-state cohort contains **53 single-action menus, 55 map
screens, and one reward screen**. Median fatal completion length is eight tokens;
truncation is not the general explanation. Broad competence probes that exclude
forced decisions cannot diagnose this failure class.

**55/56 fatal responses are valid JSON with the expected string-valued action
field.** They fail semantic matching, not JSON syntax; none has an ambiguous
legal prefix. Sixteen contain map-summary prose (15 copied from observation
phrases), 15 contain numeric strings, and seven give only a coordinate. Other
categories include wrong destinations, capitalization, and verb errors. Nineteen
already underwent numeric-prefix stripping. This points to confusion between
the public map description and the executable action contract, rather than a
general inability to serialize JSON.

## 6. Live and targeted-failure follow-ups

Matched worlds 1000–1015, search 50, greedy semantic-text output, no thinking,
256-token free cap (candidate-length constrained cap), 250-decision cap, one
retry. All three E4B arms have all 16 games accounted for:

| Policy | Mean final floor | Fatal action errors |
|---|---:|---:|
| Random baseline | 14.625 | 0/16 |
| Simple heuristic baseline | 19.000 | 0/16 |
| Base E4B, free | 13.500 | 2/16 |
| Base E4B, constrained | 14.688 | 0/16 |
| Early RL E4B, constrained | 16.000 | 0/16 |

Constrained minus free is **+1.19 [0.31, 2.25]** paired floors. Constrained
base minus random is +0.06 [-3.13, 3.06], and minus heuristic is
-4.31 [-7.25, -1.63]. Free generation required retries on 27 decisions;
constrained generation required none. All terminal outcomes are losses; no
simulator errors were observed. Invalid episodes are included at their reached
floor, not omitted from the outcome comparison.

The decoder changes choices as well as validity, so the outcome gain is the
effect of the whole decoding intervention. Both fatal free-generation cases
are branching menus, meaning a singleton bypass would not directly prevent
either one. The historical failure-state concentration therefore must not be
generalized to every evaluation policy or cohort.

Early RL minus constrained base is **+1.31 [-1.13, 3.63]** paired floors,
with all 16 RL games ending in terminal loss and no invalid decisions or retries.
This does not establish an improvement in play. Comparing RL with free base
would combine an adapter change and a decoding change; it is not the clean RL
contrast.

The complete historical-failure replay gives:

| E4B checkpoint | Greedy executable / 56 | Sampled (temperature 0.7) executable / 56 |
|---|---:|---:|
| Base | 35 | 40 |
| Early RL | 56 | 56 |
| Later RL | 56 | 30 |

All 26 later-RL sampled failures are singleton menus. Relative to sampled base,
later RL repairs six failures but introduces 16, a net decline of ten states;
relative to early RL, it introduces 26 failures. Greedy evaluation misses this
problem entirely. Of those 26 responses, nine copy observation text, seven emit
numeric strings, six give nonlegal map descriptions, three give bare coordinates,
and one has no parseable JSON.

This cohort was selected for historical final-retry failure, clusters within
16 training worlds, and uses fresh first-attempt prompts with one completion per
condition. It is a diagnostic of that failure class, not an unbiased estimate of
whole-game reliability. Early RL's successful outputs commonly include a menu
index prefix that the live parser strips; this is benign tolerated formatting,
not a semantic failure. Later RL's greedy outputs more often depend on unique-
prefix matching. Exact-contract compliance and executable validity are both
retained in the failure-replay results.

### User-requested 26B-A4B extension

During the session the user requested Gemma 4 26B-A4B and mentioned a local
Ollama copy. The machine has 48 GiB unified memory. A pinned download of
[`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)
was started (revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`, 15.37 GB).
The existing `gemma4:26b-mlx` model is actually mixed NVFP4/MXFP8 safetensors,
not GGUF. It was successfully reused through the experimental
`astra_stage_ollama_mlx.py` importer: language tensors are symlinked, packed
weight/scales names are remapped, and fused expert gate/up arrays are split
without numerical requantization. All source weight hashes were verified; all
normalized shapes and 36 packed-value/scale samples passed checks. Strict lazy
loading accepted 983 parameter arrays, and a nine-screen generation smoke ran
successfully (8/9 executable responses). The redundant full HF weight download
was stopped; downloaded metadata and resumable partial files were retained.

The staged model uses the cached tokenizer with the explicitly pinned HF26B
chat template, whose source and hash are recorded. Its no-thinking prompt closes
an empty thought channel; the E4B template differs. The single invalid smoke
response explicitly declined to choose among anonymous event-option placeholders.
That is a contract failure and evidence of an observation problem, not ordinary
malformed JSON.
The optional, unlaunched later-E4B-checkpoint live arm was removed to allocate
time to 26B, with the original plan and amendment retained. Model comparisons
necessarily change architecture, weights, quantization and chat template as a
bundle; they cannot isolate the effect of parameter count.

All five 133-state 26B arms and both 56-state failure-replay arms completed:

| 26B arm | Executable / 133 | Heuristic-continuation regret | Random-continuation regret |
|---|---:|---:|---:|
| Greedy free | 122 | 1.405 | 1.074 |
| Greedy constrained | 133 | 1.311 | 0.927 |
| Enriched constrained | 133 | 1.402 | 1.003 |
| Reversed-menu constrained | 133 | 1.811 | 1.141 |
| Sampled free (0.7) | 123 | 1.705 | 1.052 |

As above, regret conditions on valid responses and complete teacher labels;
free arms exclude different states. The clean constrained 26B–E4B comparison
uses all 132 teacher-complete states: world-mean regret difference
**-0.35 [-0.71, -0.02]** under heuristic continuation, but
**-0.02 [-0.30, 0.32]** under averaged random continuation. This is a modest,
teacher-dependent static advantage, not proof of superior live play.

All 11 greedy invalid responses were manually reviewed: each explicitly objects
to missing event-option text, all on the 16 anonymous event screens. The other
117 decisions are executable. Ten abstentions end normally; one reaches the
256-token cap while explaining the missing information. At temperature 0.7,
eight event decisions and two other decisions are invalid.

26B preserves **93/133 (69.9%)** semantic choices under menu reversal, versus
63/133 for E4B. Map preservation is **13/16**, versus 5/16 for E4B. Nevertheless,
only **1/16 Neow choices** is preserved; those action descriptions themselves
contain fixed `event option N` identifiers alongside the displayed menu indices.
The experiment establishes presentation sensitivity, not its sole mechanism.

Additional public glossary/cost/map text again does not reliably improve values.
On the 64 changed, teacher-complete states, paired world-mean regret changes are
+0.12 [-0.06, 0.36] under heuristic continuation and +0.08 [-0.08, 0.30]
under random continuation. Anonymous event text was not repaired by this
enrichment, so its absence remains a live limitation.

On the historical failure cohort, **26B is executable on 56/56 states at both
temperatures**. After JSON extraction, 55 selected descriptions match exactly
and one uses a valid unique prefix. Many raw replies contain Markdown fences,
so executable validity should not be confused with strict raw-JSON compliance.
Its behavior is therefore more robust on this selected communication
failure class than sampled later-RL E4B; the cohort contains only three actual
branching choices and cannot establish a strategic advantage.

One unchanged event prompt changes action between the original and enriched
greedy batches. Other prompts in its batch changed; the mechanism behind this
switch was not isolated. The changed-state subset is reported separately, and
individual switches should not all be attributed to their own added information.

The 16-world constrained 26B live arm reached **15 terminal losses and one
native-simulator timeout**, with no invalid decisions or retries. The observed
15-game mean floor is **17.0**. Its missing world is 1007: after the model
generated a map choice, the simulator stopped progressing. The last saved
state was floor 29. A one-second process sample placed all 844 main-thread
samples inside the native simulator module; exact function attribution was
unavailable in the stripped binary. The predeclared 900-second suite timeout
ended the process. No outcome was fabricated and no replacement run was made.

| 26B minus comparator | Mean paired floor difference on 15 observed worlds | All-16-world capped missing-outcome bounds |
|---|---:|---:|
| Constrained base E4B | +2.87 [-0.27, 5.93] | [+1.25, +4.38] |
| Constrained early-RL E4B | +1.00 [-2.47, 4.20] | [-0.06, +3.06] |
| Simple heuristic | -1.60 [-5.07, 1.73] | [-3.06, +0.06] |
| Random | +2.73 [0.40, 5.47] | [+1.31, +4.44] |

The middle column gives descriptive 95% world-bootstrap intervals conditional
on observed outcomes. The last column uses `min(final_floor,50)`, assigning
the missing 26B outcome anywhere in [0,50]; those bounds describe this cohort
and are **not confidence intervals**. They preserve the finite-cohort direction
against constrained base even under an adverse missing outcome, but do not
establish a population-level advantage. The data do not show superiority over
the simple heuristic or the early RL checkpoint.

26B is therefore a promising next baseline for communication and presentation
robustness, with modest live evidence. Increasing model size alone has not
established the competence improvement needed for a clean framing claim.
The stalled world also demonstrates why seed screening under another policy
is insufficient: all baseline policies completed that world, while this policy
reached a native state that stalled. The experiment does not identify the
underlying native bug's exact cause.

## What I would do next

1. Establish an explicit forced-action path and exact legal decoding as separate,
   versioned harness interventions. Log automatic actions as environment actions,
   exclude them from policy-gradient examples, and verify that the resulting
   credit horizon and frozen-trajectory treatment are deliberate.
2. Build a narrow **map/campfire competence task** with complete public state,
   a fixed combat controller, several continuation policies, and uncertainty-
   aware teacher targets. Average hidden futures consistent with the public
   state where possible; the fixed-world oracle's full gain may not be
   recoverable by a public-information policy. Split by whole world/run, not neighboring decisions.
   Evaluate learned policies on fresh complete games with validity and simulator
   failures beside floor. The privileged teacher is supervision, never an input.
3. Before resuming GRPO, test the objective on a small frozen diagnostic batch:
   retain failure causality in the logging, audit group composition and KL tails,
   and establish that sampled policy quality improves under a fixed valid-output
   contract. Greedy legality alone is too weak a gate.
4. Once that competence rung works, return to the original matched-data framing
   experiment. Freeze trajectories, targets, reward, teacher budget, and endpoint;
   vary framing only. Retain menu permutations and transfer-to-neighboring-traits
   evaluations so a stylistic or position preference is not mistaken for a trait.
   Measure risk-related choices conditional on the opportunities available,
   separately from floor: the lookahead improvement did not require a net change
   in elite selections at its visited map states.

## Reproducibility, scope, and artifacts

Only experimental scripts, regression/analysis tests, and documentation were
added. Production observation strings, JSONL schemas, frozen cohorts, and model
weights were not edited. All new worlds and saved probes are development data;
the embargoed final Nob cohort remains untouched. No paid remote compute was
used. Local MLX inference and native CPU simulation were sufficient.

Raw artifacts are under `data/experiments/astra_5sep/` (gitignored):

| Directory | Contents |
|---|---|
| `baselines/` | Frozen factorial plan, every attempt, traces, paired summaries |
| `information/` | Coverage denominators, examples, per-decision audit |
| `teacher/` | 133-state action/continuation labels and replay diagnostics |
| `lookahead/`, `lookahead_replication/` | Plans, attempts, traces, privileged choices, sensitivity bounds |
| `policy/` | Frozen original/enriched/reversed cohorts, model configs, predictions, validated paired analysis |
| `live/`, `failure_replay/` | Complete-game and historical-failure follow-ups |
| `credit/` | RL row/KL audit and exhaustive base-snapshot tensor comparison |
| `isolation_fullgame/`, `validation/` | Independent process parity and CLI smoke evidence |

Model configurations record snapshot/adapter identities, source-cohort hashes,
script hash, decoding settings, and start time. Suite plans and attempt ledgers
retain commands and wall times. The primary analyzer fails closed on incomplete
or mismatched cohorts, altered source hashes, inconsistent menus, or undeclared
generation changes. Regret intervals resample world-level means; they are
exploratory and do not correct for the many comparisons made this session.

The experimental native-process proxy reproduced **32/32 full baseline traces
byte-for-byte** and handles a deliberately hung worker without freezing the
orchestrator. Live model games use the original in-process environment uniformly;
the proxy was not substituted after an arm had started. Getter-sensitive replay
was also reproduced and documented in `src/sts_ai/interactive/CLAUDE.md`.
Closed-loop lookahead runs `prepare_decision` at every replayed prefix step.
The fixed-state teacher used lighter replay followed by an exact target-frontier
check; it rejected the one mismatching state rather than silently labeling it.

The final code suite passed **772 tests, with 11 skipped**. The required CLI
random smoke (seed 3, 78 decisions) reached terminal loss at floor 24, with no
undefined-behavior flag, and matched the earlier session trace byte-for-byte.
Pure regression checks cover the decoder trie,
cohort/provenance validation, failure accounting, the suite log-name collision,
and packed-weight import transformations.

[Study accounting](astra_5sep_accounting.json) records **304 planned research
whole-game slots on 64 simulator worlds: 297 observed outcomes and seven
missing outcomes** (six replication timeouts and one 26B timeout). The 64 live
model slots contain 63 observed outcomes. The main fixed-state studies produced
**3,241 responses**: 21 arms ×133 states and eight arms ×56 historical failure
states. These are repeated matched observations, not independent games.
Teacher continuation attempts, development smoke tests, replay checks, the
unlaunched cancelled arm, and the resumed pre-inference log-name error are
accounted for separately.

Compact tracked evidence accompanies this report:
[simulator results](astra_5sep_simulator_results.json),
[policy probes](astra_5sep_policy_results.json),
[live comparisons](astra_5sep_live_results.json),
[failure replay](astra_5sep_failure_replay_results.json), and
[credit audit](astra_5sep_credit_results.json).
