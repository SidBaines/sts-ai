# Non-LLM critique and implementation: stopped handoff

Written 13 October 2026, after the user's explicit request to write up the work
and stop. Shutdown began at approximately **12:00 UTC / 13:00 London**.
This document is the latest status; dated active/pending statements in older
documents are historical. **Do not resume training or investigations without a
new user instruction and allocation.**

## Outcome and operational state

The action/observation interface, training controls, reproducibility checks and
reporting are substantially improved on an experimental branch. **Reliable
full-game learning is not established. No learner or generated teacher target
was promoted.** The original-policy transfer remains the current reference.
Scope is Ironclad, ascension 0, ordinary Acts 1–3, without the Heart.

The primary allocation was through approximately 13 October 07:30 UTC. The last
learner finished at **06:25:46 UTC**. Bounded diagnostic/code follow-ups continued
past the allocation; that was an operational overrun, not additional successful
training. I should have made the handoff and stopped earlier. The explicit stop
now supersedes all earlier continuation plans.

At shutdown, process inspection found no investigation learner, collection or
counterfactual jobs. The investigation's read-only dashboard server, PID 14580
on localhost:8388, was stopped by exact PID. Its static files remain. Unrelated
processes, original worktrees and paused campaigns were left alone. There is no
active tool-managed goal. Nothing was pushed, released or newly submitted as a
PR by this investigation.

- Worktree: `/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI`
- Branch: `codex/nonllm-critique`
- Last empirical commit before this handoff: `698ef29` (boss-relic interaction).
- Original checkout: `/Users/sidbaines/Documents/SlayTheSpireAI`, branch
  `astra-6sep`; preserve its unrelated changes.
- Checkpoints, data and native builds are **gitignored local files**. Copy them
  before archiving/deleting the worktree. A Git commit alone does not preserve
  the experiment.

## What was reviewed and the main critique

I reviewed the first non-LLM curriculum campaign, the long-curriculum v1/v2
records, selected checkpoints, evaluation and lineage records, action/encoding
code, learning objectives, worker/control code and reporting. The detailed
initial review and executable evidence are in
[the critique](nonllm_training_critique_2026-10-12.md) and
`docs/assets/nonllm_critique_2026_10_12/`. Three earlier interface, learning and
operations reviews informed that record; they are finished.

**The original success was mostly Act 1 competence.** The first frozen held-out
full-control campaign cleared Act 1 in 20/21 games, but had **0/21 full-game
victories**, 20 defeats and one simulator fault. Its frozen final data were not
reopened for this work. Later v1 often discarded accumulated learning through
rollback. V2 retained about 4.03M transitions and 13,514 optimizer updates, yet
regressed from the original 2/192 development wins to zero.

A matched head crossover on 190 common-valid worlds localized broad strategic
regression while showing that combat changes also lost the rare original wins:

| Combat / OOC heads | Act 1 clears | Act 2 clears | Full wins |
| --- | ---: | ---: | ---: |
| Original / original | 151 | 13 | 2 |
| Updated combat / original OOC | 151 | 22 | 0 |
| Original combat / updated OOC | 130 | 5 | 0 |
| Both updated | 130 | 12 | 0 |

These are paired development diagnostics, not independent final estimates.
Increasing Act 2 clears alone can select away full-game victories.

The principal setup issues were:

1. **Non-equivalent actions were collapsed by display description.** A reached
   free/ordinary Strike pair had different resource outcomes (energy 3 versus
   2), despite deduplication treating it as the same action. Legal action
   identity must preserve the actual instance and mechanics.
2. **The representation lost decision-relevant public information.** Exact
   aliases included enemy block, slime max-HP thresholds, Bomb timers, upgrades,
   inventory/offer amounts and feasible routes. Some initial alias examples
   were constructed public-state pairs; later reached-state probes confirmed
   further real losses, including deck/bottled upgrade joins and live combat
   gold. Do not present all synthetic probes as reached-game examples.
3. **Strategic credit and exposure were weak.** Terminal consequences were a
   median 177 microsteps after strategic decisions. At lambda .95 the direct
   terminal contribution is about .000114 before other factors. Bootstrapping
   can transmit credit, so this is not proof that all strategic gradients
   vanish. Nominal episode mixture also differed from actual actor exposure;
   several apparent training wins contained no OOC actor decisions.
4. **The critic omitted menu context.** Adding menu/entity context was justified.
   Weighted critic-gradient probes were roughly 1–5% of actor gradients, so
   there was no evidence that critic interference alone explained the failure.
5. **Native search labels were not a calibrated public expert.** They varied
   with search seeds and hidden draw orders; adaptive root means were biased
   diagnostics. Winning-sequence first actions and maximum root means disagreed
   in roughly 41% of sampled cases. Disagreement can reflect equally good
   actions, not necessarily wrong labels. Seeds 0 and 1 alias on this native
   RNG and must not count as independent votes.
6. **Continuation wins, fresh wins and actual learning were obscured.** Forced
   actions, repeated worlds, checkpoint rollback, stale selected models and
   incomplete exposure/update metrics made progress hard to interpret. One
   confirmed checkpoint record, CP114, had been deleted. Evaluated artifacts
   now need permanent retention and learner/champion separation.
7. **Throughput had avoidable overhead.** The measured CPU-inference/MPS-learning
   path reached about 1,038 steps/s, with parsing around 33% and updating around
   27% of wall time. A static-cache fixture showed a 2.86x ceiling; that is not
   an achieved live-training speedup. Optimizing GPU use alone would not solve
   missing learning signal.
8. **The simulator is a research approximation.** Native failures, incomplete
   public mechanics and upstream departures from vanilla game rules remain
   relevant. Sampled clean runs are not complete simulator certification.

## What changed

### Public interface and representation

The branch first implemented an unreleased 0.2/public-v2/native-API-2 interface,
then moved to the currently accepted **0.3.0 / `sts_env_public_v3` / native API 3**
after further reached-state audits. Earlier source/binary/data profiles remain
separate and explicit; old banks cannot silently become current-profile data.

- Distinct combat instances and exact native teacher routing are preserved.
  Dynamic free/X costs, source/target fields and supported Ironclad Discovery
  generation are exposed correctly.
- Native API 3 joins deck/bottled upgrade instances, exposes reward-group
  alternatives such as Prayer Wheel, and supplies live player HP/max-HP/gold
  alongside combat-entry snapshots. A reached Hand of Greed state had gold
  100→120 while the old encoder still read 100.
- `public_entities_v4` encodes cards, targets, enemy HP/max-HP/block/intents,
  powers and counters, inventory/upgrades, numeric offers and feasible paths.
  Unknown/unsupported mechanics remain explicit. Legal menus are not capped.
  Neural inputs exclude seeds, RNG state, native UIDs/bits and hidden draw order.
- Source-controlled reached regression:
  `tests/fixtures/public_semantics_native_v3.json`, including a 404-action prefix.
- Checkpoint and bank identity checks reject incompatible interface, vocabulary
  or native profiles rather than allowing plausible but invalid comparisons.

See [entities-v4](public_entities_v4.md),
[public-v2 migration](environment_public_v2.md) and
[environment contract](environment_base.md). Native changes are tracked in the
patch, not only the untracked upstream checkout.

### Model and learning controls

- A frozen legacy behavior bridge preserves the original policy, with an
  initially zero typed OOC residual (`typed_residual_ooc_v3`) and separate
  actor/critic parameters. The critic receives entity/menu context. Optional
  combat residual `typed_combat_residual_v2` also freezes its reference.
- OOC macro/micro controls, gamma=1/lambda comparisons, original TRAIN rehearsal,
  actual choice exposures, unique-world counts, optimizer steps, actor-bearing
  wins, gradient norms before/after clipping, KL/TV/drift and critic calibration
  are recorded explicitly.
- Phase-separated inference fixed a tiny batching-dependent near tie
  (about 5.72e-6) that changed a reference greedy trajectory.
- MPS subnormal-probability/log-probability diagnostics produced NaNs; diagnostic
  calculations now use CPU float64 and fail closed on nonfinite results.
- Act-balanced actor loss is an explicit condition, not a retrospective metric.
  Sparse-reward training uses separate evaluation workers with unchanged
  reference rewards.
- Persistent bounded workers, complete fault-world quarantine, source/resource/
  deadline gates and retained evaluated checkpoints improve recovery and
  auditability. Learner and champion remain separate.

Important remaining gap: the new world-first frontier sampler is used by the
diagnostic probe. It has **not** silently changed PPO continuation sampling.
`strategic_data.continuation_tasks` still selects a latest start per world using
floor. Any training integration must be a newly declared condition.

### Counterfactuals, source banks and native controls

- `strategic_counterfactual.py` and `probe_strategic_counterfactual.py` run every
  legal first OOC action under a frozen continuation. Current protocol is
  `frozen_continuation_ooc_menu_v3`. Reference overrides must match full traces,
  rewards and endpoints. Faults, cuts and incomplete menus invalidate evidence;
  forced actions cannot enter on-policy PPO data as sampled behavior. No
  automatic training targets are generated.
- `early_frontiers.py`, `collect_early_strategic_frontiers.py` and
  `complete_frontier_replays.py` collect/replay exact earlier TRAIN starts.
  Version `learned_ooc_early_menu_frontiers_v2` distinguishes menu stages and
  selection purposes, including card-only rewards after collecting gold and
  late Act 1 boss relics. Source faults quarantine the entire world. Child
  replay completion preserves parent data/hashes.
- `strategic_world_then_start_v1` chooses worlds before starts, with stable
  independent RNG streams and duplicate removal. Menu-rich worlds no longer
  dominate diagnostic selection.
- Earlier per-action native controls discarded winning plan continuations;
  the built-in agent retains them. Those results now have the narrower scope
  stated explicitly. `privileged_winning_plan_reuse_v2` optionally caches only
  winning sequences, checks legal bits/native description/turn on every step,
  and clears on deviations, configuration changes, reset and battle boundaries.
  Cache data remain private diagnostic/controller state. This is a separate
  privileged comparator, not an exact reproduction of the built-in agent.
- Cache v1 incorrectly compared enriched public descriptions with native ones,
  causing 2,577 conservative invalidations. V2 uses the native formatter;
  regression tests and preserved v1 results document the correction.
- Earlier short prototypes initially recorded source hashes without source
  bytes. Exact launch sources were recovered and verified against those hashes.
  `source_snapshot.py` now saves executed environment/policy/script sources,
  vocabulary, metadata and patch bytes. Runtime/models/native builds still
  require separate preservation.

### Reporting and performance

`reporting_v3.py`, `report_nonllm_v3.py` and the dependency-free HTML dashboard
separate historical 0.2/current 0.3 profiles, fresh/conditional outcomes, paired
survival, actual actor exposures/loss mass, drift/calibration and manifest-pinned
counterfactuals. Historical nonfinite values are unavailable fields rather than
invalid JSON or silently favorable values.

The collector skips discarded public combat scoring when a private native
controller will choose the action and distinguishes externally executed native
decisions from automatic `native_steps`. A matched 16-world timing pair reduced
policy CPU time 4.065→2.753s, but wall time was 19.679→20.894s: **no end-to-end
speedup established**. The plan controller took 15.006s on those worlds, but
changed trajectories; that is not a matched-workload speed claim.

The preserved static dashboard is `data/nonllm_v3/report_0310/index.html`. Its
last export includes the 49-menu and four-source native probes; later six-world
card/boss-relic results are in manifests and the latest documents, and may be
absent from that export. The server is stopped.

### Separate, unadopted potion candidate

The native validator allowed OOC Blood/Fruit/Entropic potion use and occupied
slot discard, but public menus omitted them. Execution also fell through into
the screen handler after potion actions. A separate candidate adds legal choices
after existing screen choices, returns after potion execution, throws on invalid
use, and exposes typed potion effects. Bounded checks include Blood healing 16
at max HP 80, Fruit +5 max HP, Fire rejected OOC, invalid/repeated discard, and
REST potion use without resting.

Files: `patches/sts_lightspeed_ooc_potions_candidate.patch`,
`external/sts_lightspeed_v040`,
`scripts/investigations/probe_ooc_potions_candidate.py`,
`data/nonllm_v4/preflight/ooc_potions_candidate_acceptance.json`.
The candidate patch includes the API-3 base and applies to clean upstream, not
on top of the accepted patch. **It is not the default interface.** Adoption needs
a new interface/native profile, full screen/policy integration and regenerated
banks. See [candidate record](ooc_potions_candidate.md).

## Experiments and negative results

All counts below are development or TRAIN diagnostics unless explicitly called
the original frozen final. Do not pool profiles/cohorts or treat repeated DEV
evaluations as independent replications. Detailed commands, identities and
intermediate checkpoints are in [the chronological study record](nonllm_v3_summary.md).

### Historical 0.2 pilots

| Condition | Result / interpretation |
| --- | --- |
| Very low LR, 3e-5, four macro chunks | All 192 reference traces retained; execution check, not learning |
| LR .003, eight macro chunks | Temporary 158/21/0; ended 156/18/0 |
| Actor-bearing start macro / matched micro | 154/16/0 versus 157/18/0 |
| MC critic, five epochs | TRAIN MSE .645→.536; holdout .178→.203; rejected |
| Broader 114-frontier lambda .95 seed 0 | 157/19/0 |
| Same bank lambda 1 seed 0 / seed 1 | 159/19/0 / 156/19/0; no full-win gain |
| Short combat BC seed 0, 160 steps | 164/15/1 on main DEV; 164/17/1 on independent DEV versus 157/16/0 |
| Same short BC seed 1 / long 1,600-step fit | 152/17/0 / 134/8/0; severe instability/overfitting |
| Learner-state DAgger, 9,650 labels, budget 200 / 1,000 | 157/13/0 / 156/13/0 |
| Six-condition consensus / matched one-hot | 155/12/0 / 154/11/0 |

The favorable short BC condition supplied only 2/384 wins versus 0/384 controls,
with mixed progression and optimizer-seed failure; it was not promoted.
Of 637 teacher states, 212 had unanimous search-seed labels; adding hidden-draw
conditions yielded 287 unanimous states, 252 of which already agreed with the
baseline. Consensus alone did not establish useful corrections.

On 68 matched TRAIN tasks (32 fresh, 36 conditional), learned combat had
1/32 fresh + 1/36 conditional wins; per-action native budget 200 had 0/32 + 6/36,
and budget 1,000 had 0/32 + 7/36. These support continuation differences, not a
claim that all native experts fail or that conditional wins estimate fresh wins.

### Current 0.3 pilots

The fresh current-profile DEV192 reference is **156/20/0** (mean floor 28.198).
It is a new cohort, not the historical 2/192 cohort. All following strategic
panels had 192 valid worlds, no full-game gain, and no promotion:

| Eight-chunk condition | Act 1 | Act 2 | Full wins |
| --- | ---: | ---: | ---: |
| Macro lambda 1, uniform actor loss, seed 0 | 159 | 18 | 0 |
| Macro lambda 1, act-balanced, seed 0 | 158 | 20 | 0 |
| Same act-balanced condition, seed 1 | 156 | 17 | 0 |
| Sparse training reward, act-balanced, seed 0 | 157 | 20 | 0 |

Uniform/balanced first buffers had exactly matching 57 trajectories. Raw
Act 1/2/3 actor exposures were 1,276/935/237; balanced loss mass was 816/816/816.
This verifies weighting, not superiority. Every evaluated checkpoint is saved.

A new 65-task, 9,436-label combat panel used six search/draw conditions. All
teacher-query/no-query trajectories and rewards matched, without faults.
Heuristic stable preferences normalized root scores, averaged equivalent-copy
logits, required minimum visits/agreement/gap, and reported actual exposures.
Seed 0 gave 158/15/0, seed 1 155/15/0, matched one-hot 156/14/1. A correction-only
hinge improved held-out ordering but regressed to 135/10/0. None was promoted;
proxy fit improvements did not predict game performance.

### Counterfactual and earlier-source results

| Diagnostic | Main result |
| --- | --- |
| All 49 existing late OOC menus, 140 branches | All valid, zero faults, exact 49 controls; three reference conditional wins, zero alternative win-gain menus, three floor-gain menus, 33 ties; 43.37s |
| Fresh-process repeat of four sources | All 16 baseline/branch traces and outcomes identical |
| Four-source per-action native-1,000 comparison | 12 valid branches, exact controls; one reference changes defeat→victory, zero alternative win gains; 8.65s |
| Initial earlier 16-world source | 89 exact starts, five Act 3 starts from only one world; zero fresh wins |
| Stage-aware timing controls | 222 starts; all 16 source traces/rewards/endpoints matched across scoring paths |
| Winning-plan v2, 16 worlds | 15/3/0 source outcomes; 259 exact starts; 690 searches + 3,981 cache hits, zero invalidations/exhaustions/predicted-win defeats |
| Fresh-process repeat, same 16 worlds | Every full trace, reward, endpoint, diagnostic counter and all 259 starts match |
| Independent winning-plan source, 64 TRAIN worlds | 60/17/0 source outcomes, zero faults/cuts, 45.53s; 2,634 searches + 15,298 cache hits, zero invalidations/exhaustions/predicted-win defeats |
| Completed child of that bank | All 1,051 starts replay; 60 Act 1 boss-relic starts, 780 Act 2 starts, 211 Act 3 starts from 17 independent worlds, 77 card-only rewards |
| World-first six-card-menu comparison | Public/private each 30 valid branches and six exact controls; zero wins/gain menus; one public floor gain disappears under private continuation |
| World-first six-boss-relic comparison | Public/private each 24 valid branches and six exact controls; private continuation has two winning alternatives in one menu, public has zero wins |

The most useful concrete interaction is TRAIN world **431000063**. Original OOC
chooses Coffee Dripper and dies at floor 33 with either combat controller. Sozu
or Snecko Eye instead reaches floor 50 with learned combat but still loses;
the privileged plan controller wins at floor 51 with either alternative. This
supports a joint strategic/combat diagnosis in that reached state. It is one
fixed hidden future, not a general relic ranking, a public-agent win or an
accepted teacher target. Earlier negative panels remain preserved.

The complete bank broadens source coverage; its 64 fresh private-controller
games still had zero wins. Multiple menus from one source remain correlated.
Counterfactual policy seeds vary continuation choices, **not hidden futures**.
Public-history-consistent hidden-future uncertainty remains unresolved.

See [earlier bank/controller record](early_strategic_frontiers.md) and
[initial counterfactual record](strategic_counterfactual_v1.md).

## Validation and accepted identities

Last code acceptance: **565 host/native tests in 17.548s, OK, one optional
generated-bank skip**, plus the mandatory bounded native smoke (seed 3, Act 1,
50 battle simulations) terminated normally after 21 steps with no error.
Logs: `/private/tmp/nonllm_world_sampling_suite.log` and
`/private/tmp/nonllm_world_sampling_smoke.log`. Subsequent empirical commits only
recorded results; the handoff is prose/status work, not another code revision.

- Old/new canonical transfer: 192/192 action/control/reward/endpoint matches.
- Current profile: two fresh processes, 192/192 full action-JSON hashes match.
- Frozen micro/macro control: eight exact trajectories/reward matches.
- Phase-separated zero-residual control: all 192 reference traces match.
- Accepted native patch: fresh/reverse apply checks pass.
- Native 64-world bank: 23,278 rows / 970 fights, zero faults, 31.01s.
- Coverage audit: 6,550 OOC + 16,728 combat rows, zero unknown IDs/status/kinds
  or unresolved joins; 2,514 unsupported damage-target fields explicitly unknown.
  This is sampled encoding coverage, not proof every mechanic is represented.
- All 49 current late frontiers and all 1,051 completed earlier starts replay.
  Parent/child source snapshots contain 116/117 verified files respectively.

| Component | Accepted current identity |
| --- | --- |
| Runtime | Python 3.12.13; Torch 2.14.0; NumPy 2.5.4; MPS available |
| Environment / public / native | 0.3.0 / `sts_env_public_v3` / API 3 |
| Encoder | `public_entities_v4` |
| Reference model | `data/nonllm_v4/models/common_initial_semantics_v3.pt` |
| Model SHA-256 | `230c24dece9298e96163fcef35ad21ad87dcbaebb83326c9ed509a4e91928b63` |
| Parameters | 5,250,438 total; 359,105 OOC actor; 293,569 critic |
| Vocabulary | `public_vocabulary_v3.json`; `c2159812bd3e052a0801e1d29d7374cb9835bfc312267a6f73fd355aefb6cf82` |
| Native patch SHA-256 | `3db8b5c53898865d9bfdf9ebd0d79bcf6682b4f6b5ba39aff6eb922348657934` |
| Native module SHA-256 | `1d7a245a41b3b85c07bfff8138a8cdbb47aa90f9668f86d8005510f6a389d4b7` |
| Python-interface SHA-256 | `0e278636129ab1cf85bbe12283cc57a3616b707079782988e6418bfe4b6260c8` |
| Native upstream | `7476a81954020087da31d41d16fddf475746ec2d` |
| Native dependencies | pybind11 v2.13.6; JSON `0b345b20c888f7dc8888485768e4bf9a6be29de0` |

Use `external/sts_lightspeed_v030/build`; `external/sts_lightspeed` points to that
source. Original 0.2 source is preserved at `external/reference_code_v020`
(commit `e7145f6`), its binary at `external/sts_lightspeed_v020`, and data at
`data/nonllm_v3`. The original selected checkpoint also remains in the curriculum
worktree as `data/nonllm_native_revision/campaign/joint_act3_0030.pt`
(SHA prefix `57f1e961`). Frozen historical environment tags are unchanged.

## Artifact map for the next agent

Paths below are relative to the investigation worktree. Consult manifests for
exact commands, cohort identities, model/native hashes and source snapshots.

| Purpose | Path |
| --- | --- |
| Initial critique and proposal | `docs/nonllm_training_critique_2026-10-12.md`, `docs/nonllm_v3_plan.md` |
| Morning report / chronological results | `docs/nonllm_overnight_report_2026-10-13.md`, `docs/nonllm_v3_summary.md` |
| Compatible reference evaluation | `data/nonllm_v4/evals/common_initial_semantics_v3_dev192.json` |
| Current native training bank | `data/nonllm_v4/banks/native_train64_sim1000` |
| Existing 49 late starts | `data/nonllm_v4/banks/strategic_ooc_starts64_sim1000/starts.jsonl` |
| Current 65-task teacher panel | `data/nonllm_v4/banks/policy_teacher_panel65` |
| Original TRAIN rehearsal/calibration | `data/nonllm_v4/calibration/original_rehearsal_v4/calibration_train.jsonl` |
| Current strategic pilots | Names `semantic_macro_lam1_s0_8`, `semantic_actbalanced_lam1_s0_8`, `semantic_actbalanced_lam1_s1_8`, `semantic_sparse_actbalanced_s0_8` under `data/nonllm_v4` |
| Initial earlier bank | `data/nonllm_v4/banks/early_original_native1000_probe16` |
| Matched stage timing banks | `.../banks/early_stage_native1000_timing16_legacy`, `.../banks/early_stage_native1000_timing16_fast` |
| Cache v1 / corrected v2 / repeat | `.../banks/early_stage_native1000_plan16`, `.../banks/early_stage_native1000_plan_v2_16`, `.../banks/early_stage_native1000_plan_v2_16_repeat` |
| Independent 64-world source, partially verified original | `data/nonllm_v4/banks/early_stage_native1000_plan_v2_64` |
| **Use this completed bank** | `data/nonllm_v4/banks/early_stage_native1000_plan_v2_64_complete/starts.jsonl` |
| First counterfactual panels | `data/nonllm_v4/counterfactual/ooc_menu_greedy_panel4`, `ooc_menu_greedy_panel4_repeat`, `ooc_menu_greedy_all49`, `ooc_menu_native1000_panel4` |
| Matched six-world cards | `data/nonllm_v4/counterfactual/early_cards_world6_public`, `early_cards_world6_plan1000` |
| Matched six-world boss relics | `data/nonllm_v4/counterfactual/early_bossrelic_world6_public`, `early_bossrelic_world6_plan1000` |
| Transfer/determinism acceptance | `data/nonllm_v4/preflight/semantic_transfer_acceptance.json`, `current_profile_determinism.json` |
| Static report/screenshots | `data/nonllm_v3/report_0310/index.html`, `dashboard_postallocation.jpg`, `dashboard_counterfactual.jpg` |

The rehearsal set has 65 TRAIN tasks (32 fresh, 24 Act 2, nine Act 3), plus 24
disjoint TRAIN holdouts. Four train wins comprise one fresh/three conditional;
holdout has zero wins. There are 2,307 training and 1,730 holdout rows. Do not
interpret rows or continuation outcomes as independent fresh-game trials.

Principal implementation modules are under `src/sts_policy/`: `entities.py`,
`entity_model.py`, `legacy_bridge.py`, `combat_residual.py`, `combat_preferences.py`,
`strategic_learning.py`, `strategic_collection.py`, `strategic_control.py`,
`strategic_data.py`, `strategic_counterfactual.py`, `early_frontiers.py`,
`frontier_sampling.py`, `native_plan.py`, `source_snapshot.py`, `reporting_v3.py`
and `workers.py`. CLIs are under `scripts/`, particularly
`run_protected_strategic_v3.py`, `train_combat_preferences_v4.py`,
`collect_early_strategic_frontiers.py`, `complete_frontier_replays.py`,
`probe_strategic_counterfactual.py` and `report_nonllm_v3.py`.

Recent commits, newest first: `698ef29` boss-relic interaction; `a4cf923`
world-first sampling/card pilot; `3b74b87` winning plans/earlier banks/snapshots;
`b58d0fb` initial earlier collection; `e735a73` privileged continuation/report;
`88e4abf` complete-menu counterfactuals; `3b56616` morning/sparse control;
`f12f6ea` act balance/potion candidate; `924b597` preference controls;
`67b858a` encoding audit/frontiers; `d7fe6ba` profile reporting; `2973f3b`
accepted API-3 semantics. Earlier records retain failed conditions rather than
rewriting them into the later result.

## Recommended next work, after a new allocation

1. **Diagnose the reached boss-relic/combat interaction first.** Reconstruct
   world 431000063 and the public Sozu/Snecko final-boss states; compare public
   and validated privileged combat continuations. Then confirm strategic gains
   across independent TRAIN worlds. Do not train on DEV wins or generalize the
   single relic case into a ranking.
2. **Establish useful supervision before more long PPO or BC.** Validate
   continuation competence and sequence-level/deck-strategy decisions. Current
   fixed-world, first-choice probes and root preferences are inadequate as a
   confident public teacher. Hidden-future resampling must respect public
   history/known draw placement; policy-seed resampling is not a substitute.
3. **Use genuinely independent late-game coverage.** Explicitly integrate
   world-first sampling into a new learning condition, retaining fresh games
   and original-policy rehearsal. Report worlds, actual actor exposure and
   optimizer updates. Use multiple optimizer seeds, disjoint confirmation and
   a final cohort embargo; retain full-game and Act 1 checks jointly.
4. **Complete conformance/interface work where it changes available strategy.**
   Decide on potion-menu migration with a new profile and full integration.
   Audit remaining unsupported mechanics and simulator departures before
   claiming transfer to the real game.
5. **Keep operations bounded and reviewable.** Declare a new finite compute
   window, source/model/native identities and a stopping/handoff point. The old
   protected-controller default deadline is 2026-10-13 07:30 and is expired.
   Faster hardware is secondary to finding reliable learning signal. Preserve
   this branch's local artifacts before any cleanup.

The improvement target should be repeatable **fresh full-game wins with public
learned control**, confirmed on independent worlds while retaining early-game
competence. Conditional/private wins and lower proxy loss are useful diagnostic
evidence, but do not meet that target.

## Commands for a future authorized agent

These are reference commands, **not pending jobs**. No further command below is
being launched during this stop/handoff. Run from the worktree, with fresh output
directories for new experiments.

```sh
cd /Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI

# Accepted host/native suite and bounded smoke.
PYTHONPATH=src STS_REQUIRE_SIMULATOR=1 STS_LIGHTSPEED_BUILD=$PWD/external/sts_lightspeed_v030/build .venv/bin/python -m unittest discover -s tests -t .
PYTHONPATH=src STS_LIGHTSPEED_BUILD=$PWD/external/sts_lightspeed_v030/build .venv/bin/python scripts/run_rollout.py --agent random --seed 3 --max-act 1 --battle-simulations 50

# Refresh the static report (no learning).
PYTHONPATH=src .venv/bin/python scripts/report_nonllm_v3.py --data data/nonllm_v4 --compare-data data/nonllm_v3 --output data/nonllm_v3/report_0310

# Example bounded, current-profile TRAIN diagnostic; choose a new output path.
PYTHONPATH=src STS_LIGHTSPEED_BUILD=$PWD/external/sts_lightspeed_v030/build .venv/bin/python scripts/probe_strategic_counterfactual.py --checkpoint data/nonllm_v4/models/common_initial_semantics_v3.pt --starts data/nonllm_v4/banks/early_stage_native1000_plan_v2_64_complete/starts.jsonl --selection world-balanced --selection-seed 0 --decision-family card_only_rewards --per-act 3 --seconds 120 --panel-seconds 20 --output /NEW/OUTPUT/PATH
```

For the boss-relic diagnostic, use `--acts 1 --per-act 6` and omit the card-only
family filter. For the separate privileged comparator, add
`--native-combat-simulations 1000 --native-plan-cache`. Check current CLI help
before launching. Historical 0.2 reproduction must instead use
`PYTHONPATH=external/reference_code_v020/src:external/reference_code_v020` and
`STS_LIGHTSPEED_BUILD=$PWD/external/sts_lightspeed_v020/build`; never mix that
source/runtime profile with current banks.

The old v1/v2 campaigns remain paused with their own recorded remaining budgets.
Nothing in this handoff authorizes resuming them or opening final cohorts.
