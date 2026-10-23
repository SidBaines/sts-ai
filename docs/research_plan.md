> Historical pre-CLI research plan. Current work is described in [research/README.md](../research/README.md).

# SlayTheSpireAI Research Plan

## Research question

This repository tests how training-time framing changes what a language model
learns from the same reward signal. The motivating question is:

> When a model is reinforced for behaviour along a graded axis such as
> risk-taking, does the framing of the training context determine which broader
> latent concept absorbs the update?

For example, equivalent successful trajectories could be presented as
"risk-reward tradeoffs" or as "adventurous" behaviour. The eventual evaluation
asks whether those otherwise matched training conditions generalize differently
to nearby traits such as risk-seeking, adventure-seeking, confidence, or
impulsivity.

Slay the Spire is the first environment. Its pathing, elite fights, low-HP
campfires, card rewards, shops, potions, and boss preparation provide graded,
interacting tradeoffs rather than a single explicit risk control.

## Active priority: competence first

Competence work precedes the framing experiment. A framing comparison is not
interpretable until the student can learn a useful policy through the current
interface and training pipeline. Resume framing work from a frozen competence
checkpoint, rather than varying framing while basic policy learning remains
unsettled. Past competence experiments and remaining limitations are summarized
in [`experiment_history.md`](experiment_history.md).

## Design commitments

- **Prompt neutrality.** Base observations and instructions describe facts and
  legal actions without strategic advice. Framing is an explicit experimental
  variable, not an accidental property of the serializer or reward.
- **Matched data for the primary arm.** Generate neutral trajectories once and
  train framing variants on the same states, actions, and rewards. This
  fixed-rollout arm is the cleanest test of interpretation effects.
- **On-policy follow-up.** A later arm may generate trajectories separately under
  each framing to measure the combined effect of interpretation and visited-state
  distribution.
- **Trait-neutral reward.** Reward game progress and outcomes, not words or
  proxies that encode the target framing. Keep reward logic identical across
  framing conditions.
- **Markovian public observations.** Each decision receives a fresh,
  self-contained, human-visible state and legal-action list. Private simulator
  state and search rollouts must not enter the policy prompt.
- **Explicit seed identity.** World seed, rollout index, and policy seed are
  distinct. Comparisons use frozen splits and paired identities where applicable.
- **Reproducible, schema-stable artifacts.** Frozen traces are immutable inputs;
  derived corrections use sidecars. Interface, schema, simulator, model,
  tokenizer, and adapter identities are recorded, and incompatible artifacts fail
  closed rather than being silently reinterpreted.
- **Outcome and validity together.** Policy quality is always read beside invalid
  output, timeout, and simulator-error rates. Partial asynchronous batches are not
  compared as if they were complete cohorts.

## Harness architecture

The environment is `gamerpuppy/sts_lightspeed`, built locally with a versioned
Python-binding patch. `LightspeedHybridEnv` exposes legal actions, public state,
and deterministic trace recording.

The default hybrid mode gives Python control of Neow, pathing, rewards, shops,
events, card selection, treasure rooms, and campfires while the built-in
Lightspeed search agent resolves combat. Full-control mode instead exposes each
combat micro-action to the policy. Both modes use the same agent protocol and
JSONL decision schema.

Serial rollouts support smoke tests and diagnosis. MLX lockstep batching supports
local generation, while the vLLM streaming path keeps multiple rollouts in flight
for higher-throughput evaluation and future on-policy training. Training code
supports filtered behavioural cloning, search-teacher SFT, offline policy
gradient, and an on-policy GRPO loop; optional ML dependencies remain separated
from the dependency-free core.

Public combat observations are versioned. The frozen `combat_public_v3` surface
extends card-type-complete v2 observations with simulator-computed attack damage
and derived turn arithmetic. The opt-in `combat_public_v4` fight-lab adapter
repairs audited glossary/selection gaps and replaces overly categorical turn
claims with explicitly scoped arithmetic. Search-teacher queries operate on cloned simulator
state and remain privileged labels, not policy observations.

## RL design notes (for the framing experiment)

- Run the offline, matched-data arm first (filtered BC → reward-weighted
  regression on shared neutral rollouts); it is the cleanest isolation of the
  framing effect. On-policy follows.
- For on-policy, prefer group-relative methods (GRPO/RLOO) over PPO — no critic
  network fits local hardware. Groups need `temperature > 0`: at temperature 0
  all rollouts of a seed are identical and group advantages are zero.
- Hybrid combat control masks the consequences of out-of-combat choices (the
  search agent absorbs mistakes), so start RL on the out-of-combat action
  space — those are the risk-relevant decisions — before full-combat RL.
- An unrecoverable invalid decision ends the episode, so outcome metrics mix
  format compliance with policy quality: always report `agent_invalid` rate
  beside every outcome comparison, and give errored/truncated episodes a
  defined reward rather than scoring them as ordinary losses.
- Whether the gradient flows through reasoning tokens is an experimental
  choice, not a default: framing may act through reasoning style, so keep it a
  logged, switchable flag.

## Frozen seed splits

[`configs/frozen_seeds.json`](../configs/frozen_seeds.json) records the world
seeds used for evaluation splits, with erroring seeds excluded (its `excluded.*`
lists carry the reasons). Seed 2 is excluded from LLM splits (crash-class
failure) and seed 1 is diagnostic-only; the clean intersection is 142 seeds and
the LLM-safe set is 141. The training split is not yet frozen. Seed behaviour
can differ across machines/builds because of the simulator's known
uninitialized-memory bug, and the splits derive from Act-1-era runs.
Regenerate with `scripts/run_batch.py` plus the exclusion lists in the config.

## Current state

On 6 September, local single-fight setup moved to a pinned, fresh HF MLX 4-bit
Gemma 4 26B-A4B instruction checkpoint. The new
[single-fight lab](single_fight_lab.md) collects bounded heuristic trajectories,
filters arbitrary public combat positions, verifies exact replay before isolated
fight evaluation, and reuses the Studio board in a local collection/trajectory
browser. The browser now launches Gemma from the exact displayed source combat
state, reports background progress/errors, supports cancellation, and opens the
saved trajectory on completion. Each launch freezes a single-state selection
and preserves the existing evaluator's replay/seed contract. Its 12-world pilot passed independent repeat/replay checks and contains
1,898 states / 89 battle starts. Three opening eval fights completed under Gemma
with no invalid actions; this is a mechanics cohort, not a competence estimate.
A short QLoRA update and adapter reload also passed on the 48 GB Mac. Full
combat-prompt training, broader baseline panels, and held-out-enemy learning
experiments remain next steps. The historical interfaces and frozen artifacts
are unchanged. See the runbook for quantization identity, output-parser/cap
caveats, and the limits of the simulator checks.

A subsequent 600-second native-search collection (including independent repeat
generation and replay checks) produced 49,462 positions and 2,132 completed fights
from 227 validated Act 1 worlds, stored under
`data/experiments/single_fight_v1/search_10min_2026-09-06` (1.22 GB including
validation copies). There are 1,598 train / 534 eval combat starts across 25 enemy
types. Twelve further attempts are retained but excluded: nine phantom-power
flags, two simulator undefined-behavior flags, and one collection-deadline cutoff.
This collector replans native search at every displayed action, rather than
using the pilot's first-legal combat policy. It is a new data cohort, not a
replacement for frozen trajectories or a student-model competence result.

Full-bank stochastic Gemma sampling was launched as
`runs/gemma4_t1_p095_k64_v1` under that bank: 2,132 scheduled attempts with
temperature 1.0, top-p 0.95, top-k 64, thinking off, and per-state sampling seeds.
The MLX factory now forwards both sampling filters (previously ignored).
The resumable supervisor saves every decision and atomic fight results for
preliminary analysis, retains incomplete attempts, and preserves interrupted
traces before same-seed restart. A 26-decision fresh-process repeat matched
exactly excluding latency. Aggregate outcomes and generalization analysis remain
pending completion or a deliberate early stop; live progress is in the arm's
`progress.json` and `report.json`. It was paused at the user's request on
6 September at 20:13 UTC: 201 finished attempts (178 wins, 11 deaths, 11 invalid
action stops, one timeout), plus an interrupted trajectory with 23 decisions
preserved. There are 1,931 pending attempts, including that interrupted fight.
The worker exited and the run lock was released; resume remains explicit.

Rerunnable interim analysis now lives in `scripts/analyze_fight_bank.py`, with
a dated 53-figure gallery and frozen input snapshots under
`outputs/gemma_fight_analysis_2026-09-06/`. The latest paused snapshot contains
201 attempts and 189 resolved pairs. Gemma loses 9.48 more net HP on average
than search from the same start (13.99 versus 4.52 HP). Win fractions are 88.6%
of all finished attempts and 94.2% of resolved fights. All 53 captioned figures
were regenerated; earlier snapshots are retained. These interim figures keep
incomplete attempts explicit and do not establish held-out-enemy competence.

The [6 September Gemma template audit](gemma_template_audit_2026-09-06.md)
examined 3,686 saved decisions from those 201 attempts and the partial trace.
It found ten concrete information/presentation issues, including dropped intent
descriptions (763 records), missing active relic/power rules, an incorrect
Flame Barrier+ effect, and overly categorical TURN MATH claims. Representative
prompts and transitions are preserved under `outputs/gemma_template_audit_2026-09-06/`.
The audit itself changed no runtime interface. Subsequent
[versioned repairs](gemma_template_fixes_2026-09-06.md) now implement all ten
findings in `combat_public_v4`, keeping the original renderer and native binary
unchanged. All 202 attempted source starts pass exact v3-prefix replay followed
by v4 public-state/action-identity checks. The paired rerun keeps the model,
sampling parameters, thinking-off setting, and per-fight seeds fixed; its new
arm is `runs/gemma4_t1_p095_k64_public_v4_v1`. Performance effects remain
unmeasured until new outcomes are available; the original arm remains paused.

A matched thinking-on follow-up is queued as
`runs/gemma4_t1_p095_k64_public_v4_thinking_v1`. It freezes the full 202-start
schedule from the v4 thinking-off arm, including starts still pending there,
and enables thinking with the same per-fight seeds and sampling filters.
At the user's subsequent request, its initial total completion budget is now
16,384 tokens, with an automatic first-20-generation check: at least two cap
hits and a rate of at least 10% (minimum five observations) raises the limit
through 32,768 / 65,536 / 131,072. On each increase, early attempt artifacts are
archived and all 202 starts restart with their original seeds, keeping the main
cohort at one budget. Watchdog limits scale with the cap; the maximum-tier
condition is recorded explicitly if truncation persists. The separate launcher waits for every preceding attempt to finish
and for model-worker locks to release; no active sampler code was modified.
The queue and its cancellation/launch gates passed regression tests; the full
suite passed 852 tests with 11 skips plus the bounded smoke rollout. Native
thinking-template activation was checked with the local tokenizer. Budget control
adds its own regressions and a dedicated runtime shim without changing the
active thinking-off implementation. Model
outcomes for this queued arm are not yet available.

Overnight monitoring now runs separately through `scripts/monitor_fight_runs.py`,
checking live process handles, progress, active-trace writes, and queue/budget
status every 30 seconds. `scripts/compare_fight_formats.py` produces the
three-arm comparison with 45 captioned PNG/SVG plots and a written report under
`outputs/gemma_three_arm_comparison_2026-09-06/`; it refreshes at milestones and
after both new arms finish. The report is currently interim. Primary comparisons
use identical finished starts, HP comparisons require resolution, and unfinished
data remains explicit. The old arm's retained partial means at most 201 starts
have finished observations in all three arms before infrastructure exclusions. State matching permits only the
declared v3/v4 presentation change, and cap counts use the recorded token limits.
Final performance conclusions remain pending the thinking run and verification
of the final report/figures.
The comparison's derived token-limit diagnostic now applies the same near-limit
rule to every arm: counts at/above the limit, plus incomplete responses within
eight tokens below it. This gives three indicators in the old arm versus its
frozen sampler's one strict-threshold count; raw counts and near-limit counts
are reported separately because saved text is re-tokenized. An interim residual
failure review also preserves remaining literal-menu copying errors without
changing the active experiment.
The Mac entered low-power sleep at 1% battery on 6 September at 23:02 BST and
woke on 7 September at 08:48 BST, interrupting overnight computation despite
the idle-sleep assertion. The resulting v4 no-thinking watchdog timeout for
`seed_4194_r0_d24` was archived as an infrastructure interruption and rerun
with the same seed. Its four saved prefix decisions matched the recovered
trajectory exactly excluding latency. The recovered fight won; the queue
then waited for all 202 no-thinking results before starting thinking.
The v4 no-thinking arm finished on 7 September at 09:01 BST with 180 wins,
16 combat deaths, and six invalid-action stops. Thinking launched automatically;
its first saved decision contains 4,700 extracted thinking tokens and a valid
final action (4,727 total tokens, 73.4 seconds), without truncation. This verifies
native thinking activation in the actual run, but its outcome comparison and
representative throughput estimate remain pending further generations.
The first-20-generation budget check subsequently passed: all 20 responses
were valid, their thinking blocks closed, and none was truncated. The maximum
saved completion count was 7,107 tokens, so the automatic controller accepted
the initial 16,384-token limit. The verification and prefix digest are saved in
`outputs/gemma_overnight_monitor_2026-09-06/thinking_budget_verification.json`.
Subsequent power-log review identified the old arm's sole timeout
(`seed_4229_r0_d138`) as another low-power-sleep interruption, from 20:07 to
20:24 BST on 6 September. The frozen old attempt is retained with a hashed
annotation and excluded from matched model-performance estimates. Therefore
the maximum primary three-arm cohort is 200, with historical 202-start coverage
still shown separately. On those 200 comparable no-thinking attempts, both old
and new formats win 178; the new format loses 1.87 more net HP on the 188
jointly resolved starts (world-bootstrap 95% interval 0.79 to 2.93 HP). This
conditional HP result is not evidence of held-out generalization.
The thinking arm's first fight subsequently won with 67 HP remaining, compared
with 46 (old format) and 44 (v4 no thinking), taking 1,089 seconds and 22 decisions.
This is one matched case, not an aggregate performance estimate. The Mac then
entered clamshell sleep at 09:20 BST and logged **Thermal Emergency Sleep** at
09:21 BST on 7 September. Subsequent maintenance wakes caused another watchdog
timeout before its first decision. Inference was deliberately paused pending
confirmation of a cool, ventilated setup on power. That timeout and the next
partial attempt are archived under the thinking arm's
`infrastructure_interruptions/thermal_sleep_20260907T082127Z`; they will replay
with their original seeds after resumption. The one completed thinking win and
accepted 16,384-token budget remain intact. The three-arm goal is unfinished.
The user subsequently confirmed that inference should remain paused during inspection. A
read-only, side-by-side viewer now opens the three completed trajectories for
`seed_4050_r0_d95` (three Sentries, floor 10). It supports shared turn jumps,
independent decision stepping, explicit final outcomes, full observations,
thinking, and raw responses. Its self-contained artifact is
`outputs/fight_comparison_seed_4050_r0_d95/index.html`, served locally on port
8013; rebuilding uses `scripts/view_fight_comparison.py`. It loads saved data
only and cannot launch inference.

At the user's explicit request, thinking sampling resumed on 7 September at
15:33 BST through the validated queue launcher. The accepted 16,384-token
budget and sampling settings remain fixed, with one completed fight preserved
and 201 remaining. Archived interrupted attempts replay with their original
seeds. Incremental trajectory writes and the separate progress monitor remain
enabled; the resume audit lives under the thinking run's
`resume_history/20260907T143312Z/launch.json`.

The user requested another pause at 18:06 local time on 7 September.
The queue and model worker stopped; 10 attempts are finished
(8 wins, 2 invalid-action stops), and the active
partial trajectory retains 0 saved decisions. Further inference awaits an
explicit resume. The run's `user_pause.json` records the verified stop.

The 8 September paused-data analysis refreshes all 45 comparison plots and
adds a detailed audit under `outputs/gemma_analysis_2026-09-08/REPORT.md`.
On ten common starts the thinking arm records eight wins versus ten in each
no-thinking arm; the eight jointly resolved pairs save 4.625 HP on average
versus v4/no-thinking, with strong selection and small-sample limitations.
The two thinking stops are distinct: one near-budget unclosed thought and one
valid final-channel action rejected because an unmatched brace in the reasoning
confuses the whole-response JSON scanner. No outcome was retrospectively changed.
The existing near-cap plot proxy misses the truncated reply (nine tokens below
the limit); the audit records the explicit truncation flag separately.
Remaining verified presentation gaps include slime split thresholds/offspring
HP, pre-activation enemy buff quantities, Rampage bonus notation, and the energy
denominator. The v4 name-coverage scan covers 4,130 decisions but does not prove
mechanical completeness. Parser/presentation repairs remain pending; inference
stays paused and frozen runtime files are unchanged.

The subsequent requested repairs implement `combat_public_v5` and
`final_channel_v2` parsing; see
[the 8 September fix report](gemma_template_fixes_2026-09-08.md).
V5 is the default for new fight-lab evaluations and adds exact instance card
rules, scoped Block calculations, slime/buff mechanics, round/energy notation,
and consistent multi-card selection labels. Final-channel parsing is shared
with training action-span extraction. The historical corpus also exposed two
accepted JSON objects inside unclosed thinking, so the corrected diagnostics
report three unclosed thinking replies among 127. Original outcomes stay fixed.
Validation passes 889 tests (11 skips), a bounded smoke rollout, all 202 source
starts, and 4,130 decisions across the 212 saved v4 attempts. Inference remains
paused; new model outcomes must use a new arm rather than modifying old runtime
fingerprints. Pre-repair source is archived for historical reproduction.

The [2026-09-05 repository review](repository_review_2026-09-05.md) audits the
implementation and local artifacts. It records qualifications to the older
status prose below: incomplete OOC observations, RL update/metric limitations,
and training/evaluation overlap and quarantine findings. These remain unresolved;
the review did not change frozen interfaces or rerun training experiments.

The harness, replay validation, search-teacher collection, semantic teacher
targets, quarantine-aware metrics, local/pod inference, and training paths are
implemented. Frozen competence artifacts exist (replay-validated cohorts,
teacher label files, and the embargoed final Nob cohort), so the
schema-stability and reproducibility rules in `CLAUDE.md` are active — treat
`state_text`/action-text changes as versioned interface changes, not free
edits. The strongest static result so far comes from semantic action-text
targets on a small Gremlin Nob dataset. A first behavioural comparison on
held-out saved fights favoured that adapter (more wins, fewer combat deaths),
while also showing that mid-turn and card-select states — absent from the
first-per-turn training set — dominate its live-play failures. Coverage is
limited to one encounter and the simulator still has an unresolved
phantom-power bug. See
[`experiment_history.md`](experiment_history.md) for the concise evidence record
and open items.

Competence work has since reached whole games and a first RL rung. A shared
three-encounter combat adapter generalizes across held-out fights, and two
whole-game base arms are recorded on the frozen eval-30 seeds: with the search
agent resolving combat the base model reaches mean floor 18.9 with no invalid
decisions, while under full combat control it reaches 10.6 and loses a third of
its episodes to format failures. The first out-of-combat GRPO run
(`ooc_grpo_v1`) is paused incomplete at iteration 6 of 12; it peaked at mean
floor 18.6 before a format-drift instability took over the curve, so it is not
yet evidence about out-of-combat play quality. Its status, resume instructions
and the outstanding paired evaluation are in
[`ooc_rl_v1_status.md`](ooc_rl_v1_status.md).

Two constraints from that work shape what comes next. Format robustness, not
policy quality alone, constrains interpretation of out-of-combat RL under the
`action_text` contract. The built-in search agent only searches in combat and
its scripted map-screen pathing is random. However, the
[5 September diagnostic experiments](astra_5sep_experiments.md) now demonstrate
an experimental OOC supervision source: replaying public frontiers and evaluating
legal actions with full-game continuations. Closed-loop map/campfire lookahead
improved paired final floor by +12.69 [9.21, 16.45] on 29 complete pairs from a
fixed 32-world replication; all missing outcomes were separately accounted for.
This is privileged fixed-world supervision, and preferred actions differ across
continuation policies. It is not yet a validated public-state teacher dataset
or a learned competence improvement.

That session also found substantial semantic menu-order sensitivity, sampled
format deterioration in the later RL checkpoint, and 53 single-action menus
among 56 historical fatal decisions. Exact legal decoding and automatic forced
actions are promising experimental controls; neither has been adopted into the
production interface or frozen training data. Further competence studies should
separate observation completeness, valid-output policy, and teacher quality
before attributing floor changes to a learned risk trait. The primary framing
design remains matched-data training from a frozen competent checkpoint.

The same session evaluated the user's cached Gemma 4 26B-A4B NVFP4/MXFP8
model locally. It improved communication and menu-permutation robustness, but
explicitly declined anonymous event prompts with missing option text. Its
constrained live arm completed 15/16 worlds before a native-simulator stall;
the paired floor difference against constrained E4B was +2.87 [-0.27, 5.93]
on observed worlds. It is a promising next baseline, not established evidence
that model scaling has solved OOC competence. Simulator isolation and complete
event descriptions remain necessary work for the next study.
