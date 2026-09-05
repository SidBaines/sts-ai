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

Public combat observations are versioned. The current `combat_public_v3` surface
extends card-type-complete v2 observations with simulator-computed attack damage
and derived turn arithmetic. Search-teacher queries operate on cloned simulator
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
policy quality, is the binding constraint on out-of-combat RL under the
`action_text` contract. And there is no out-of-combat teacher: the built-in
search agent only searches in combat, and its scripted map-screen pathing is
random, so the most floor-relevant out-of-combat decision class has no
supervision source.
