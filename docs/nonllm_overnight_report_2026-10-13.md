# Overnight findings and next study

13 October 2026. Scope: Ironclad, ascension 0, ordinary Acts 1–3. Work is isolated
on `codex/nonllm-critique`. **We now have a substantially more trustworthy
training setup, but no candidate establishes reliable full-game improvement.**
All bounded strategic pilots finished, the last at 06:25:46 UTC. Models,
optimizer recovery, task identities and rejected outcomes are preserved. No
learner was promoted; previous campaigns and final cohorts stay preserved.

## What the evidence says

The original curriculum established strong Act 1 competence (20/21 held-out
full-control clears) but zero full-game wins in 21 requests, including one
simulator fault. The later v2 campaign retained 4.03 million transitions yet
regressed its development baseline from 2/192 wins to 0/192. Phase crossover
implicates out-of-combat drift in the broad progression loss; combat changes
also lose rare wins. Extending those runs without changing the learning signal
was poorly justified.

The new repaired-profile reference scores **156 Act 1 clears, 20 Act 2 clears,
zero full wins out of 192 fresh development worlds**. This is a new cohort,
not the old 2/192 panel. Current strategic controls are directly paired:

| Eight-chunk condition | Act 1 | Act 2 | Full wins |
| --- | ---: | ---: | ---: |
| Reference | 156 | 20 | 0 |
| Macro credit, λ=1, uniform actor loss, seed 0 | 159 | 18 | 0 |
| Macro credit, λ=1, equal actor-loss mass by act, seed 0 | 158 | 20 | 0 |
| Same act-balanced condition, seed 1 | 156 | 17 | 0 |
| Sparse training reward, act-balanced, seed 0 | 157 | 20 | 0 |

Every evaluation finishes 192 valid worlds. Seed-0 gains do not replicate;
removing progress shaping does not establish a win gain in this short study.
This does not prove that macro credit, act balancing or sparse rewards cannot
work. It rejects scaling these particular pilots as demonstrated improvements.
The sparse arm uses separate evaluation workers to preserve the reference
evaluation configuration. Evidence: `data/nonllm_v4/pilots/*/status.json` and
their saved paired evaluations.

A short earlier combat imitation fit under the preserved 0.2 profile wins
**2/384** across two disjoint development cohorts versus 0/384 for its reference.
However, a second optimizer seed regresses and ten-times-longer fitting falls
to 134/8/0 on the first 192 worlds. Current-profile stable preference fits also
lose Act 2 clears. A correction-only fit improves held teacher ordering while
collapsing live play to 135/10/0. Better proxy loss is not a reliable learning
signal here. No candidate qualifies for promotion.

The native search teacher is useful but is not a public expert oracle. On 637
learner states, only 212 equivalent action labels are unanimous across three
search seeds. Its best-sequence choice often differs from the highest visited
root mean. Hidden draw permutations also change labels. Some disagreement may
be between equally good actions, so instability alone does not prove teacher
error. On a matched small TRAIN control, stronger privileged search helps
conditional later-act success but produces no fresh wins. Both deck strategy
and combat supervision need outcome validation.

## Repairs and new capabilities

**Action and observation fidelity.** The public menu previously merged a
free-once Strike with an ordinary Strike because their descriptions matched;
their actual resulting energy is 3 versus 2. Distinct legal instances and exact
teacher routing are now preserved. The typed entity encoder represents card
upgrades/costs/counters, enemy HP/block/intents/powers, source/target relations,
numeric offers, reward alternatives and feasible routes. Unsupported numerical
effects have explicit unknown masks.

Further reached-state audits found identical inputs for upgrading bottled and
ordinary deck instances, missing Prayer Wheel reward groups, and stale combat
gold. The accepted unreleased 0.3/public-v3/native-API-3 revision repairs these,
including coherent live HP/max-HP and explicit historical entry snapshots.
The original policy has a frozen compatibility bridge; incompatible banks or
learned residuals fail closed rather than being silently reinterpreted.

**Learning and retention.** The new actor and critic have separate parameters;
the critic sees entity/menu context. Strategic collection can compress combat
into macro transitions while freezing combat. Original-policy rehearsal,
choice-only actor losses, λ controls and explicit act weighting are available.
Eight of ten initial TRAIN wins began at the final boss with no OOC choices;
the curriculum now selects genuine earlier strategic decisions instead.
Current strategic starts replay exactly in 49/49 cases, though only nine
independent Act 3 source worlds remain a serious coverage limitation.

**Operations and metrics.** Native workers persist across bounded chunks.
The controller saves permanent evaluated checkpoints and distinguishes learner
from champion. Fault worlds are quarantined; resource/deadline and source
identity checks protect recovery. Logging counts actual optimizer exposures,
unique rows, actor-bearing wins, phase/act distribution, pre/post clipping
gradients, final policy drift and return calibration. MPS subnormal-probability
NaNs in drift metrics are repaired with float64 diagnostics. Phase-separated
inference repairs a tiny padding-dependent near-tie that changed an otherwise
frozen control trace.

The report separates fresh wins from conditional continuation wins, paired
gains from losses, current from historical profiles, and teacher metrics from
live outcomes. Historical nonfinite metrics are explicitly unavailable with
source links. CPU collection is already roughly a thousand primitive steps
per second in measured runs; hardware is not the dominant demonstrated barrier.
Vectorized preference fitting takes about 15–16 seconds for 40,960 exposures.
These throughput figures have different workloads and are not a speedup ratio.

**Validation.** Latest suite: 540 host/native tests, one optional generated-bank
skip; required bounded smoke without error. All 192 canonical reference
controls/rewards/endpoints transfer unchanged, and two fresh current-profile
processes match all 192 full action-JSON hashes. The new TRAIN bank has 64 valid
worlds, 23,278 labels and zero faults; its complete identity/join audit passes.
This certifies the tested execution properties, not complete game fidelity.

**Potion candidate.** A separate native patch exposes missing legal OOC potion
consumption/discard actions and fixes execution fallthrough into the ordinary
screen handler. Blood/Fruit Juice, discard and invalid-action mechanics pass
bounded tests. It has not replaced the default binary or menus. Adoption needs
an explicit interface version, integration coverage and a policy prior for new
choices. See [candidate details](ooc_potions_candidate.md).

## Recommended route toward reliable full-game wins

1. **Build and validate stronger strategic supervision first.** Use diverse
   TRAIN decks and learner-reached map/reward/shop/rest choices. For each sampled
   decision, compare legal choices with matched future randomness, several
   public-consistent hidden futures and learned continuation. Track full-run
   return and retention, not just short search scores. Native privileged combat
   is a separate diagnostic control. Validate whether the proposed teacher
   improves subsequent live play before collecting a large imitation bank.
2. **Expand independent late-act coverage.** Collect more distinct Act 2/3 decks,
   bosses, scaling/AoE/block profiles and low-HP situations. Nine Act 3 source
   worlds and tiny held teacher subsets cannot support broad generalization.
   Start near meaningful strategic choices where both successes and failures
   occur, then move earlier as conditional competence improves. Keep fresh
   full games and Act 1 rehearsal throughout; count conditional wins separately.
3. **Make combat supervision outcome-based.** Compare candidate actions under
   matched continuations and uncertainty, respecting what the public history
   actually reveals. Search root means are adaptively sampled, biased proxies.
   Use soft/equivalent-action targets only when live continuation outcomes
   support them. Fit short protected residuals, isolate combat versus strategy,
   and check several optimizer seeds before longer training.
4. **Use a stricter promotion protocol.** A candidate must improve fresh full-game
   wins on development data, retain progression on paired controls, then confirm
   on a larger disjoint development cohort across several learning seeds. Freeze
   selection before opening final worlds. Until then, keep the verified original
   transfer as reference. A single new win or lower teacher loss is insufficient.
5. **Finish remaining fidelity work alongside the study.** Adopt the OOC potion
   candidate through explicit migration; add real-game conformance fixtures and
   targeted public-history features where action outcomes show a need. The
   simulator is not generally certified by deterministic replay alone.

The immediate priority is a small outcome-validated OOC teacher pilot with
broader independent late-act starts. Another long PPO run, larger model or
larger GPU currently has less supporting evidence. Architecture and reward
alternatives remain hypotheses, not accepted cures.

The [initial critique](nonllm_training_critique_2026-10-12.md),
[implementation plan](nonllm_v3_plan.md), [dated evidence](nonllm_v3_summary.md)
and [accepted interface](public_entities_v4.md) give more detail. Local models,
banks and builds are gitignored: copy them before removing this worktree.
