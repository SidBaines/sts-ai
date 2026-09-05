# Repository review — 2026-09-05

Review of HEAD `a084dbf` and the locally available experiment artifacts. This is
an assessment, not a new experiment or an implementation change. The findings
below qualify older status prose where noted; frozen artifacts were not edited.

## Overall assessment

The infrastructure is substantially more mature than the scientific evidence.
Keep it: the environment/agent separation, trace contract, replay validation,
teacher isolation, provenance checks, and local task machinery are useful research
assets. The strongest demonstrated result is improved end-to-end performance on
saved elite fights after semantic-target, dense-state SFT. Stable whole-game RL
improvement and the original framing effect have not been demonstrated.

Format failure is a measured bottleneck, but not the only impediment to OOC
learning. The public interface omits essential event and planning information,
and the RL update has important limitations independent of the parser.

## Code structure and verification

The dependency-free core, optional ML backends, shared rollout record builders,
explicit seeds, versioned observations, and strict local-task comparisons are
strong choices. Tests cover many historical failures. Complexity concentrates in
`agents.py` (1,288 lines), `train/train_mlx.py` (1,876), and large teacher/evaluation
scripts. Observation text is also an internal API parsed by the glossary, viewer,
affordances, and risk metrics. Future improvements should move shared semantics
toward structured public state and keep text as a rendering boundary, while
preserving frozen interfaces.

Onboarding and reproducible setup lag the implementation. README still calls full
combat binding future work, and its manual build omits the maintained binding
patch and pybind11 update. `scripts/build_lightspeed.sh` clones an unpinned upstream
HEAD and decides whether to apply the patch from one marker string. Existing
binary/source hashes help detect identity differences, but do not make a fresh
checkout reproduce the original build. No checked-in CI workflow was found.
Generated evidence is gitignored; an artifact index with durable locations and
hashes would make this research easier to hand over.

Checks run during this review:

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`:
  714 tests, OK, 11 skipped. This does not certify skipped heavyweight model tests.
- Random hybrid smoke: seed 3, max 80 decisions, normal game loss at floor 24
  after 78 decisions.
- Two fresh processes repeated that rollout with byte-identical JSONL traces
  (SHA-256 `4562bc503ef3862b4473829477f1fc3ec49af4feb8b2950fab5fbf78f8441fbf`).
  This checks one seed on this build, not cross-build determinism.
- The versioned patch reverse-applies cleanly to the local simulator source;
  upstream checkout is `7476a81954020087da31d41d16fddf475746ec2d`.

## Gameplay information audit

Public combat v2/v3 are substantial interfaces: card types and costs, enemy
intents and powers, unordered piles, relic counters, potion slots, and current-turn
history. Search labels are generated on clones and kept privileged. This is
appropriate because [upstream Lightspeed](https://github.com/gamerpuppy/sts_lightspeed)
uses knowledge of RNG state. Hybrid results describe OOC decisions coupled to
that particular combat solver, not ordinary human-information full-game play.

The most consequential gaps are:

1. **Anonymous ordinary events.** Binding patch lines 534–545 enrich only Neow.
   Other events have `event option N` actions and no event identity or consequence
   text in state. Reproduced in the review smoke at decision 42, floor 12: three
   anonymous options. Human event choices expose their consequences; for example,
   [Falling](https://slaythespire.wiki.gg/wiki/Falling) identifies the cards to lose.
   These prompts cannot support informed event-specific tradeoffs.
2. **Lossy and unavailable route information.** `glossary._render_map` reports
   the union of reachable room counts, losing sequence and exclusive branches.
   A campfire before an elite is strategically different from one afterward.
   The actual [map](https://slaythespire.wiki.gg/wiki/Map_Generation) preserves
   connecting paths. `LightspeedHybridEnv.map_graph` supplies nothing on shops,
   rewards, campfires, or Neow, although those fresh standalone decisions can
   depend on the future route.
3. **Incomplete offered-object descriptions.** OOC glossary enrichment defines
   offered cards but only already-held relics and potions. New relic/potion offers
   can lack their effects. Offered cards generally lack energy costs. This makes
   the interface unevenly dependent on memorized game knowledge.
4. **Incorrect or overstated reference text.** `glossary.py` describes Flight as
   halving enemy outgoing attack damage; it actually halves attack damage the
   enemy receives. Time Warp's counter persists across turns, unlike its
   description of twelve cards “this turn.” These agree with the local simulator
   and the [buff reference](https://slaythespire.wiki.gg/wiki/Buff) /
   [Time Warp reference](https://www.slaythespire.gg/buffs/Time_Warp).
   Whirlwind's glossary omits its upgraded damage; see the
   [card reference](https://slaythespire.wiki.gg/wiki/Whirlwind).
5. **v3 arithmetic is not a tactical proof.** `turn_math.py` computes a static
   knapsack over currently displayed damage but emits categorical lethal claims.
   Bash 8 + Strike 6, three energy, versus 16 HP produces “not lethal this turn”;
   Bash's Vulnerable makes the subsequent Strike deal 9, so that sequence deals
   17. The end-turn projection also omits effects beyond incoming attacks, block,
   and Metallicize. Preserve the distinction between an arithmetic estimate and
   a rules-complete conclusion. The recent main studies used v2, so this v3 issue
   is not an explanation for their results.

Any fixes must be new versioned observations, not edits to frozen trace text.

## RL implementation and interpretation

Group-relative floor rewards, distinct policy/world seeds, and critic-free LoRA
updates are sensible pilot choices. Nevertheless:

- Both trainers set `logp_old` from the current forward pass with gradients
  stopped. Ratio is always one and clipping cannot constrain policy movement.
  The OOC run makes approximately 125 optimizer updates after generating a
  cohort; later updates therefore use stale-policy data. One pass over examples
  does not guarantee on-policy updates throughout the pass. This is signed PG
  plus KL, without effective clipping against the generating policy. Compare
  the [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer).
- `pg_dataset.py` inherits SFT exclusions for unexecuted, invalid, and retried
  responses. Iteration 6 excludes 56 fatal unexecuted decisions and 485 retried
  decisions. Negative episode advantages affect preceding retained actions;
  the fatal malformed completion gets no direct gradient. Retrying needs its
  actual conditioning prompt recorded before it can be used faithfully for PG.
- Saved MLX trainer logs show large sampled KL-estimator outliers: iteration 0
  mean 4.08, maximum 3,237.87; iteration 6 mean 4.73, maximum 3,888.27. These are
  sampled estimates, not exact distribution KL. MLX has no gradient-norm clip.
  Investigate these rows and gradient norms before assuming increasing KL weight
  alone will stabilize training. The claimed parser feedback mechanism is a
  hypothesis, not an identified cause.
- `iter_N/rollouts` evaluates the incoming policy, before `iter_N/adapter` is
  trained. Iteration 2's peak measured `iter_1/adapter`. The older status note's
  suggestion that adapters 2 or 3 are best is not established by that curve.
- `run_grpo` hard-codes action masking. Generic dataset CLIs lack an explicit
  matched-data reframe intervention, despite core override support, and do not
  fully expose modern output contracts. The original framing protocol is not
  ready as an end-to-end CLI workflow.

Floor is a competence reward, not a direct risk-preference measure. Elite visits,
self-damage card drafting, and resting depend on deck strength and opportunities.
In addition, `risk_events` currently classifies terminal records with
`action_executed=False` as behavior: an unexecuted low-HP `smith` record was
reproduced as a risk-seeking event. Correct this before using those proxies for
the framing study, and condition preference comparisons on comparable choices.

## What the evidence supports

Verified from local metadata/comparison artifacts:

| Task | Base wins | Shared adapter wins | Invalid stops, base → adapter |
|---|---:|---:|---:|
| Nob, clean saved holdout | 14/32 | 25/32 | 10 → 0 |
| Lagavulin saved holdout | 10/36 | 25/36 | 20 → 1 |
| Sentries saved holdout | 2/36 | 22/36 | 34 → 2 |

These are meaningful end-to-end gains. Dense Nob-only training replicated over
three training seeds; the shared three-encounter result has one training seed.
Local comparisons appropriately bootstrap saved-window means: the independent
situations number eight or nine, not 32 or 36. In particular, Sentries' dramatic
gain cannot isolate tactical learning from output compliance. These are held-out
starts within trained encounter types, not unseen-encounter transfer. Reused
development cohorts remain exploratory; static teacher-choice accuracy has not
reliably ranked behavioral performance.

Whole-game base arms each have 60 episodes: search combat gives mean floor
18.883, no invalid stops; model combat gives 10.617, 20 invalid stops. Both have
zero wins. The shared-adapter composite whole-game arm has not been run.

OOC RL iterations 0–6 each have 96 completed episodes. Their mean floors are
16.781, 16.823, 18.594, 17.760, 13.438, 15.083, 9.833; invalid deaths are
22, 1, 0, 0, 2, 0, 56. This establishes instability, not held-out policy gain.
Iteration 7 contains 29 traces but only 17 completed metadata sidecars; the other
12 are partial, contrary to the status note calling all 29 finished.

Further evidence qualifications:

- `data/competence/comp_022/datasets/train_dense3_action_text.jsonl` contains
  713/2,252 examples from 23/30 COMP023 eval world seeds. A future composite arm
  on eval30 can measure whole-game deployment performance, but not unseen-world
  generalization. OOC RL seeds 202–217 are disjoint from eval30.
- That training file retains 78 rows from two quarantined Nob windows, including
  28 explicitly flagged state hashes. Resolver-integrity checks do not establish
  state-sanity cleanliness; quarantine-aware reporting is not training exclusion.
- The old historical hybrid control uses OOC `reasoning_action`, one retry,
  streaming vLLM; RL uses `action_text`, two retries, MLX. It cannot simply be
  reused as a weights-only treatment control without matching settings or
  declaring the additional interventions.
- `compare_paired.py` averages by world seed and takes the intersection. Unlike
  `local_task_compare.py`, it does not enforce complete cohorts, exact episode
  identities, or matched interfaces/settings. Partial inputs can produce a report.
- The history's shared-Nob clean counts (25/32 vs 14/32) accompany a reward CI
  from all eleven windows, including quarantine. The full-cohort delta is
  +0.447 [0.193, 0.722]; recomputing the eight clean windows with the existing
  bootstrap gives +0.519 [0.190, 0.865]. Both favor the adapter; label cohorts
  consistently. Source comparisons are under
  `data/local_curricula/{gremlin_nob,lagavulin,sentries}/eval/comp022_behavior/`.

Earlier whole-run reward-weighted regression and Nob self-BC were null results,
as recorded in `experiment_history.md`. No result yet tests the framing question.

## Implications for the next studies

Full-game mastery need not be the gate for the original question. A bounded,
well-observed decision distribution with reliable legal outputs, measurable
learning, and untouched test situations can support a cleaner framing study.
Preserve the local-task and trace infrastructure; prioritize information
completeness, state sanity, and a reliable action interface. Choose a fixed
competence checkpoint using separate validation, and create new disjoint tests.
Then implement the matched-data framing intervention explicitly, holding actions,
rewards, token-loss policy, and training schedule fixed, and evaluate transfer
under neutral conditions. Risk proxies require context and opportunity controls;
game success alone does not show which latent trait changed.
