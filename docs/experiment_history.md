# Experiment History

Git history retains the detailed plans, preregistrations, results, progress reports, and analysis files removed during the repository context cleanup.

## Environment and harness

The environment is Slay the Spire through `gamerpuppy/sts_lightspeed`. The harness can give Python control of out-of-combat choices while the built-in search agent resolves combat, or expose combat micro-actions to a model. Competence work used `mlx-community/gemma-4-e4b-it-bf16` (Gemma E4B) as the student and a privileged Lightspeed search policy as the teacher.

## Full-run training era

Prompt-only sweeps from roughly 0.6B to 12B parameters all recorded zero wins and mean final floors around 10–13. A 100-rollout paired comparison of whole-run reward-weighted regression plus sparse tactical hints against base measured a floor delta of -0.46, 95% CI [-1.91, +0.99]. Nob-only self-behavioural-cloning on the model's winning fight windows measured a reward delta of +0.003, 95% CI [-0.16, +0.18]. Both training results were treated as nulls.

## Interface era

The first public combat interface omitted card type. In a single stopped live transcript, the model treated Attacks as Enrage-triggering Skills and Defend as safe, inverting the relevant Gremlin Nob logic. `combat_public_v2` added Attack, Skill, Power, Status, and Curse types throughout the public state and legal actions; persistent enemy-power serialization was added to the public state in the same period. `combat_public_v3` later added simulator-computed damage annotations for all attack types through an oracle-tested path, plus derived `TURN MATH` lines. A known simulator uninitialized-memory bug can produce impossible "phantom" powers. The surviving quarantine sidecar, [`configs/competence/state_sanity_quarantine_v1.json`](../configs/competence/state_sanity_quarantine_v1.json), flags 46 state hashes in five windows across the frozen label files; the development split was affected more than training (17/60 states across 3/11 windows versus 8/150 across 2/32 windows).

## Teacher era

The built-in search agent solved 43/43 saved Gremlin Nob starts at both 5,000 and 50,000 simulations. Hidden-order-consensus labels using 50,000 simulations were adopted over 5,000-simulation labels. Displayed-action-aggregated root visits were adopted over the searcher's raw winning-sequence line. Teacher per-query vote agreement with the hidden-order consensus was 81.8%; near-tie states also showed first-action ordering noise even when the broader turn set was stable.

## Index-target SFT era (experiments 008–018)

Tiny-set memorization controls reached 31–32 correct choices out of 32. In a matched 20-pass comparison on the harder tiny set, the mixed format-plus-action mask reached 31/32 (mean teacher probability 0.821) while an action-value-only mask reached 15/32 (0.419); the mixed mask was retained as the default. Training on every eligible micro-action collapsed to one displayed index. On 150 aligned first-per-turn states, the final adapter fit 149/150 training states but scored 28/57 on development, versus 22/57 for base. Under cyclic reordering of the same legal-action menu, semantic top-1 invariance measured 0.448 and 0.253 at the two evaluated checkpoints. Permutation augmentation, action-token loss reweighting, and visible-reasoning output were each evaluated and dropped: at action-token weight 8 the tiny set was not learned (7/32), while the matched unit-weight control reached 31/32. These results motivated replacing the displayed-index target rather than attributing the gap to a single established cause.

## Metric reset

Development choices were rescored using visit-share regret, tie-aware top-set agreement, and turn-set membership, with a quarantine-clean split reported separately. On the full split, base scored 0.421 strict top-1 with 0.237 mean regret; the best index-target adapter scored 0.509 with 0.183 regret. A 25-line hand heuristic scored 32/57 on the same strict choice metric.

## Semantic-target result (experiment 020)

The semantic action-text comparison held the 150 states, model, and training recipe fixed while changing completions from menu indices to exact legal-action text. Across three training seeds, the best checkpoints reached 0.614 top-1 and 0.128–0.143 mean regret. At the fixed step-1,500 endpoint, the three seeds scored 0.579/0.614/0.614 top-1, versus 0.509 for the best index-target adapter. Free generation was 100% valid JSON, 94.7% legal, and 0.596 top-1. Best-checkpoint position varied by seed: one seed was degenerate at step 750 and recovered by step 1,500. Future comparisons therefore require fixed-step endpoints or a real validation split rather than development-based checkpoint selection. A turn-plan variant trained on 130/150 states scored lower on first-action agreement, with a best top-1 of 0.526, while producing 100% schema-valid plans.

## Behavioural evaluation of the semantic adapter (2026-08-12)

Live play under the `action_text` contract was added to the agent parser (exact match, then unique-prefix, then integer fallback), and the seed-1 step-1,500 adapter was compared with base on the 43 saved Nob starts (4 samples per window at temperature 0.7, `combat_public_v2`, quarantined windows reported separately). On the 8 clean holdout windows (32 episodes per arm): wins 20 vs 14, combat deaths 4 vs 8, invalid-format stops 8 vs 10, favouring the adapter. On the 30 clean training windows (120 episodes per arm) the adapter's invalid-format stops rose to 81 vs 46, capping its wins at 34 vs 49; its stops concentrated at median turn 1 on mid-turn states (a first card already played) and card-select screens — state families absent from the first-per-turn training set. Paired HP-loss deltas (−14.5 holdout, −17.3 train) are confounded by early invalid-stop truncation and were not treated as play-quality evidence. On quarantined windows all 12 adapter holdout episodes ended invalid on corrupted screens. Won fights cost the adapter less HP than base wins (22.6–25.0 vs 34.4–37.2 mean, selection-biased). The search teacher's reference on the same starts remains 43/43.

## Dense-state retraining (experiment 021, 2026-08-12)

Holding the model, recipe, and label source fixed, the training set was changed from 150 first-per-turn states to all 600 consensus-passing dense states (every visited micro-action, including card-select screens). On the clean holdout behaviour protocol, the step-750 checkpoint won 21/32 with one invalid-format stop (versus 20/32 with eight for the first-per-turn adapter and 14/32 with ten for base); its paired reward delta over base was +0.24 with a 95% bootstrap CI of [+0.04, +0.45]. The step-6000 checkpoint won 22/32 with six invalid stops. Intermediate checkpoints were much worse (step 2250: 9/32 with 21 combat deaths), and static dev57 scores oscillated non-monotonically across checkpoints (strict top-1 between 0.246 and 0.632) with poor rank agreement between static scores and behavioural outcomes. All results are from a single training seed.

## Dense-recipe replication (2026-08-13)

Training seeds 0 and 2 replicated the dense recipe end-to-end. At the step-750 endpoint the three seeds won 21/19/23 of 32 clean-holdout episodes with at most one invalid-format stop each; at step 6000 they won 22/23/26 with 6/0/0 invalid stops (pooled 71/96, 74%, versus base 44%). Paired reward deltas over base at step 6000 were +0.51 (95% CI [+0.26, +0.77]) and +0.63 ([+0.32, +0.94]) for seeds 0 and 2. The single-seed inference that early checkpoints play better did not replicate: step 6000 was at least as good as step 750 in every seed. Static dev57 instability replicated (each seed shows a mid-training top-1 crater, 0.14–0.23), and static rank continued not to predict behavioural rank.

## Encounter generalization (experiment 022, 2026-08-13)

The Lagavulin and Sentries cohorts (31 and 27 train windows) were teacher-labeled with the exact Nob protocol (950 and 910 dense states; ~2.7 s/state), after fixing a replay failure on serializer-drifted action descriptions (`resolve_action_index` now falls back to a unique engine-bits match). One shared adapter was trained on all three encounters (2,252 examples, seed 1, 6,000 iterations) and behaviour-evaluated at step 6,000 on each holdout. Results: Lagavulin 25/36 wins versus base 10/36 (invalid stops 1 versus 20), paired reward +0.47 [+0.14, +0.80]; Sentries 22/36 versus 2/36 (invalid stops 2 versus 34), +0.86 [+0.44, +1.28]; Nob clean holdout 25/32 with zero invalid stops versus base 14/32, +0.45 [+0.19, +0.72]. Against the Nob-only dense adapter the shared adapter showed no regression (+0.07 [−0.21, +0.46]) and the numerically best Nob arm so far. The earlier skills-per-turn hypothesis was not supported: the winning shared adapter plays more skills per turn on Nob than arms it outperforms, and raw skill counts are confounded by deck composition and fight length. Its dev57 static (0.526 top-1) remained unpredictive of its behavioural rank.

## Whole-game arms (experiment 023, 2026-08-14)

The shared three-encounter adapter was taken out of isolated fight windows into
whole games on the first 30 frozen eval seeds
(`configs/competence/comp023_eval30_seeds.json`), 2 rollouts per seed at
temperature 0.7 under `combat_public_v2` and `action_text`. Before launch,
fusion faithfulness was verified (MLX-plus-adapter versus MLX-plus-fused,
greedy on 57 stored dev prompts, 57/57 exact text matches) and engine
equivalence was verified (MLX versus vLLM-metal on the same fused weights,
greedy fight episode, 16/16 identical decisions and the same outcome).

The first launch failed informatively: every adapter run and every base run
stopped with `agent_invalid` at the first out-of-combat screen. Exact-text
`action_text` is unusable out of combat for both models, because event and
reward strings are long. Three fixes followed. Output contracts became
per-phase, threaded through all three orchestrators. The adapter turned out to
have forgotten the index contract out of combat (emitting
`{"action_index": "play_card"}`), which motivated `CompositeAgent` — the
adapter fights, the base model navigates — with the side benefit that arms
share an identical out-of-combat policy, isolating combat skill. The base
model's dominant combat format failure was copying the menu line verbatim
(`"0: play Strike…"`), so the parser now strips the `N: ` prefix before
matching. A composite smoke test covering 2 full games and 251 mixed-phase
decisions produced zero invalid decisions.

Two base arms completed, 60 episodes each. With the built-in search agent
resolving combat, `base_hybrid` reached mean final floor 18.9 (median 16,
max 50) with all 60 episodes terminal and zero invalid decisions in 4,198.
With the model in full combat control, `base_fc` reached mean floor 10.6
(median 10, max 24), and 20 of 60 episodes ended in `agent_invalid` despite a
per-decision invalid rate of only 0.2% (20 of 8,778) — over a long episode a
low per-decision failure rate compounds into a third of runs lost to format.
The `adapter_fc` composite arm was postponed to free the GPU for RL and has not
been run.

A parallel label-integrity audit re-collected the Lagavulin and Sentries labels
under the stem-gated resolver; both were byte-identical, confirming the teacher
data and the shared adapter trained on it are clean.

An investigation into an out-of-combat teacher found that **none exists**.
`ScumSearchAgent2` only searches in combat; out of combat it runs hand-scripted
policies of very uneven quality — expert-curated card-reward weights and
sensible rest, boss-relic and treasure rules, but many events fall through to
random, and map-screen pathing is literally `stepRandom`. Exposing it as a
teacher would teach random pathing, the most floor-relevant out-of-combat
decision class. Three routes were costed: (A) expose the scripted policies as
oracles, worth it only for card-reward and card-select screens; (B) a
playout-based out-of-combat search teacher scoring each option by N full
playouts from a cloned `GameContext`, the principled option and roughly one to
two days, with the caveat that a floor-maximising playout teacher bakes in a
risk posture that the framing experiment would need to account for; and (C)
self-imitation format unification, converting outcome-filtered out-of-combat
decisions from our own runs into `action_text` SFT rows so one adapter speaks
the same contract everywhere. None have been implemented.

## Out-of-combat GRPO (`ooc_grpo_v1`, 2026-08-17 to 08-19)

The first RL rung: hybrid out-of-combat GRPO over final floor, with the search
agent resolving all combats so within-group floor variance is attributable to
out-of-combat decisions. The run reached iteration 6 of 12 and is paused
incomplete. Mean floor rose from 16.8 at the base policy to a peak of 18.6 at
iteration 2, then fell to 9.8 by iteration 6.

The result is a **format-drift instability rather than a play-quality finding**.
The lenient `action_text` parser combined with on-policy supervision of emitted
tokens forms an amplifying loop: leniently-resolved forms get reinforced, drift
to the parser's edge, then a wave of invalid deaths slams format back. The
invalid-death series across iterations 0–6 was 22, 1, 0, 0, 2, 0, 56 — a
growing oscillation, not a damping one — so by iteration 6 the reward curve
mostly measured format dynamics. A v2 recipe (supervise the canonical action
description rather than the emitted variant, add a retry penalty, raise
`kl_beta`) was identified but deliberately left unapplied to keep the v1 curve
clean.

The run also exercised the harness against the simulator's uninitialized-memory
hang, which proved to be **state-dependent and therefore not screenable per
seed across policies**. This produced the policy-seed salting mechanism, a
three-layer fix for restart-induced trajectory concatenation, and the
supervisor tooling in `scripts/rl_ops/`. Full status, resume instructions, the
per-iteration table and the incident log are in
[`ooc_rl_v1_status.md`](ooc_rl_v1_status.md).

## Open items

- Checkpoint selection still relies on behavioural evaluation; no frozen validation split exists, and dev57 statics remain unpredictive.
- The gap to the search teacher (100% on all cohorts) is 22–39 points of win rate depending on encounter.
- The adapter has still not been evaluated inside full-game runs; the `adapter_fc` composite arm was postponed and only the two base arms exist.
- `ooc_grpo_v1` is paused at iteration 6 of 12, and its outstanding deliverable — a paired evaluation of the best pre-instability adapter against the `base_hybrid` control arm — has not been run. See [`ooc_rl_v1_status.md`](ooc_rl_v1_status.md).
- Format robustness, not play quality, is currently the binding constraint on out-of-combat RL; the v2 recipe addressing it is unimplemented.
- No out-of-combat teacher exists, and none of the three costed routes has been implemented. Map-screen pathing — the most floor-relevant out-of-combat decision — has no supervision source at all.
- Lagavulin/Sentries labels have not been state-sanity audited; the phantom-power quarantine only covers the Nob files.
- All shared-adapter results are single-seed (seed 1) at a single endpoint (step 6,000).
- `combat_public_v3` prompts have not yet been used for training.
- The simulator's uninitialized-memory bugs are unfixed at source: the phantom-power variant, and the hang variant that is state-dependent and therefore not screenable per seed across policies.
- The embargoed 13-window final Nob cohort remains untouched; its operational record is [`configs/competence/nob_fresh_final_cohort_v1.json`](../configs/competence/nob_fresh_final_cohort_v1.json).
