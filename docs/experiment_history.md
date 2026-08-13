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

## Open items

- Checkpoint selection still relies on behavioural evaluation; no frozen validation split exists, and dev57 statics remain unpredictive.
- The gap to the search teacher (100% on all cohorts) is 22–39 points of win rate depending on encounter.
- The adapter has not been evaluated inside full-game runs (hybrid or full-control), only on isolated fight windows.
- Lagavulin/Sentries labels have not been state-sanity audited; the phantom-power quarantine only covers the Nob files. The simulator UB is unfixed at source.
- All shared-adapter results are single-seed (seed 1) at a single endpoint (step 6,000).
- `combat_public_v3` prompts have not yet been used for training.
- The simulator's phantom-power bug is not fixed at source.
- The embargoed 13-window final Nob cohort remains untouched; its operational record is [`configs/competence/nob_fresh_final_cohort_v1.json`](../configs/competence/nob_fresh_final_cohort_v1.json).
