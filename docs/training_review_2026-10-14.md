# Training setups and evidence — 14 October 2026

The best retained learned player is round-2 OOC PPO update 20. Its combat weights are inherited from the initial campaign. No later combat student has established better complete-game play. On the current 0.1.4 simulator condition, learned-only wins 219/4,096 repeated DEV worlds and 213/4,096 fresh confirmation worlds. Native search at Act 2–3 bosses and elites wins 333/4,096 on that fresh confirmation; adding ordinary later-act hallway search wins 98/512 in a smaller repeated DEV pilot, without fresh confirmation of the broader scope.

This review distinguishes actual training from play-time search and diagnosis. Older numerical results belong to their recorded simulator, cohort, horizon and evaluation conditions; they are not a single comparable leaderboard. Simulator faults are separate from game defeats. Repeated DEV selection is not held-out evidence. No further training is authorized by this review.

## 1. Common architecture, inputs and objective

Scope is Ironclad, ascension 0, ordinary Acts 1–3; Act 4 and other characters are outside the demonstrated training coverage. The selected original architecture is a large candidate-scoring MLP with no menu-context branch: 2,298,882 parameters per combat or out-of-combat (OOC) model, about 4.6 million total. The models are separate across phases. Within each model, actor and critic share the state encoder. The selected no-context critic does not consume action embeddings. An optional mean-context variant was screened but was not selected.

`public_candidate_v2` encodes 256 state features and 128 features per legal candidate: public scalar/keyword features, costs, damage, block, target properties, card/potion rules and signed hashed bags. The scorer evaluates legal candidates rather than predicting a fixed global action ID. Menu indices are routing metadata. Seeds, native action bits, hidden draw order, search scores and diagnostic info are excluded from learned policy inputs.

This is a purpose-built numerical network, not an LLM, recurrence or an entity-attention model. Public information is compressed and incomplete; a large parameter count does not recover omitted information. Actor changes can also alter the decks/states seen by the other phase. Full-game victories are the headline objective, with progression and retention as secondary diagnostics.

Source: [original handoff](experiment_handoff.md), [encoding](nonllm_encoding.md), [initial campaign](nonllm_summary.md).

## 2. First curriculum: behavioural cloning, DAgger, then PPO

The first campaign moved through isolated combat, OOC with native combat support, integrated Act 1, and finally full Acts 1–3 control. Behavioural cloning fitted teacher choices; combat DAgger added teacher labels on learner-visited states; masked PPO then optimized live play. Six model-size/menu-context screens and multiple learning-rate/initialization controls were run. The selected combat settings included learning rate 1e-4, PPO clip .1, gamma 1 and entropy .01. The final integrated checkpoint is `joint_act3_0030.pt`.

A native automatic-combat character-initialization defect invalidated old auto-combat OOC/lookahead evidence. It was fixed in environment 0.1.1, and affected labels were regenerated. Old full-control Ironclad combat was retained only after scoped checks. This was a simulator-condition boundary, not a silent update to historical evidence.

The frozen first-study held-out results were:

| Assessment | Learned successes / requested | Faults |
| --- | ---: | ---: |
| Isolated Act 1 fights | 145/160 | 0 |
| Isolated later-act fights | 25/32 | 0 |
| Act 1 OOC with native combat | 17/21 | 0 |
| Act 1 full learned control | 20/21 | 0 |
| Full three-act control | 0/21 | 1 |

The last row is 20 ordinary defeats plus one simulator fault, not 21 losses. The specialist composition also had zero victories, without faults. Strong Act 1 competence was supported. Isolated later-act fighting did not translate into reaching and completing those acts with learner-built decks. These final cohorts are now historical/exposed and cannot be reused as untouched tests.

Source: [final evaluation](nonllm_final_evaluation.md), [completion audit](nonllm_completion_audit.md).

## 3. Longer joint curricula: v1 and v2

V1 mixed fresh starts, later-act continuations and Act 1 rehearsal. Its early stages selected on the Act 2 subgoal, with frequent evaluations and restoration of the stage incumbent. The audit found a serious distinction between compute performed and learning retained: 526 completed chunks and 36,217 optimizer updates were recorded, but the selected checkpoint contained only six new chunks and 511 retained updates. A two-win candidate on the 48-world full-game panel was rejected because its Act 2 result was worse. The selected model remained 0/48 full-game wins. This showed a selection/objective mismatch and inefficient repeated branching, not a clean demonstration that 36,217 accumulated updates failed.

V2 restarted from the original morning checkpoint and separated a continuously advancing learner from a confirmed champion. All tasks targeted Act 3 victory. The fixed 128-episode mix was 90 fresh starts, 25 Act 2 continuations and 13 Act 3 continuations. It retained gamma 1, GAE lambda .95, floor reward .05, terminal +/-1 and zero terminal-HP bonus. Optimizer state was carried between ordinary chunks. Screening used 48 DEV worlds and confirmation 192; those remained development panels, not independent final tests.

V2 genuinely accumulated updates, but two larger confirmations gave 0/192 wins against the original 2/192. Act 1 retention also declined. On 190 common-valid DEV worlds, the phase crossover was:

| Combat / OOC weights | Act 1 clears | Act 2 clears | Full wins |
| --- | ---: | ---: | ---: |
| Original / original | 151 | 13 | 2 |
| New / original | 151 | 22 | 0 |
| Original / new | 130 | 5 | 0 |
| New / new | 130 | 12 | 0 |

OOC changes caused the broad early progression loss on this panel. Combat changes improved average Act 2 progression but still lost the two original winning trajectories. Small local PPO KL did not prevent accumulated drift from the original. Critic interference through shared state features is a plausible mechanism, not experimentally established. Both longer studies remain paused; their allocations and final cohorts were not restarted by the later experiments.

Source: [v1 audit](long_curriculum_learning_audit.md), [v2 protocol](long_curriculum_v2_protocol.md), [phase diagnosis](long_curriculum_v2_forgetting_diagnosis.md).

## 4. The successful OOC macro-credit PPO recipe

The 13 October investigation isolated OOC learning instead of continuing joint training. The combat network was frozen and used historical greedy collection. Combat rewards between consecutive OOC choices were accumulated into an OOC macro transition, with the next OOC critic used for bootstrap. This removed cross-phase critic transitions and reduced combat-sampling noise in strategic credit.

Core settings were 768 games per iteration, learning rate 3e-4, three PPO epochs, minibatch 4,096, clip .2, entropy 0, KL guard .05, gamma 1, and filtering of Act 4 key actions that cannot help an Act 3 task. Later execution used chunking, 1,024-row microbatches and multiple collectors. The historical first recipe used lambda .95 and a scaled-greedy combat mode; later overnight controls explicitly used exact combat argmax. Historical greedy tie behaviour must not be described as universally deterministic.

The first recipe improved paired mean floor by roughly 3.1 floors by iterations 36–40, and a second initialization seed showed similar early progress. It was a real improvement over the initial composite, but plateaued. Rare wins and collapsed strategic choices remained problems.

The important distinction is that freezing combat weights did not by itself solve OOC credit. Macro transitions and adequate batches made the learning signal more useful. This is the principal successful PPO configuration in the later lineage.

Source: [13 October findings](rl_findings_2026-10-13.md), preserved `ooc_main_s1` launch log/config.

## 5. Strategic demonstrations plus long-horizon OOC credit

The policy had nearly eliminated useful alternatives: in a diagnostic sample it never smithed, never chose an offered elite and never bought an affordable removal. Logit gaps made some alternatives almost impossible to sample. We measured rules through paired menu overlays before distilling them.

Useful measured rules were smithing at high HP and choosing an offered elite at high HP. Smithing at low HP was harmful; removals, indiscriminate card skipping, potion conservation and several relic-avoidance rules did not justify adoption. The retained strategic demonstration rules were smith at HP >=85% and take an offered elite at HP >=90%.

OOC distillation used cross-entropy on rule labels plus beta 1 KL rehearsal against the preceding policy, with combat tensors unchanged. Smith-only distillation used 2,877 labels, learning rate 3e-4, batch 256, 12 epochs and a 10% whole-world validation split. Its learned behaviour approximately matched the live overlay without needing the rule at deployment. Round-2 smith/elite distillation used 5,198 labels and 16 epochs at the same learning rate/batch; label-collection and fitting parents are separately recorded.

The lambda comparison addressed what happened after distillation. Lambda .95 PPO gradually erased smithing and lost about .54 mean floors by iteration 16. Lambda .99 was better but drifted. Lambda 1 retained the behaviour more effectively and improved mean floor by about .88 at iteration 12 over the distilled starting policy. Same-trajectory smith-versus-rest advantage probes changed sign in relevant HP buckets when critic bootstrap dependence was removed. This supports critic/credit bias as an explanation for that measured behaviour, not a universal claim that lambda 1 solves all strategy learning.

The retained lineage is:

`initial composite -> OOC PPO update36 -> smith distillation -> lambda 1 OOC PPO update12 -> smith/elite distillation -> round2 lambda 1 OOC PPO update 20`.

Historical old-backend round-2 results rose to about 3.8% wins. Later checkpoints through update 32 did not establish a reliable additional gain. On corrected 0.1.3, the matched 4,096-world comparison gave update 20 220 wins versus update 12 146. The current single-observation 0.1.4 reference gives 219/4,096 DEV and 213/4,096 fresh confirmation. The historical percentages and current figures must not be treated as one unchanged-condition curve.

Source: [findings](rl_findings_2026-10-13.md), [pipeline audit](rl_pipeline_audit_2026-10-13.json), [overnight log](rl_overnight_2026-10-14.md).

## 6. Focused combat PPO and earlier search imitation

Fight-bank PPO replayed learner-reached starts, updating combat while freezing OOC. The sparse version used 1,024 fights per iteration, terminal +/-1 plus .3 HP retention on wins, gamma 1/lambda 1, clip .2, entropy 0, value coefficient .25, KL guard .05 and minibatch 4,096. The mix emphasized Act 2 bosses/elites while retaining earlier and later fights. Worlds were held out as groups; periodic full-game evaluation checked transfer.

V1 used learning rate 1e-4/three epochs and showed only an inconclusive +.29 mean-floor change at iteration 10. V2 continued from that checkpoint at 3e-4/four epochs; it changed both LR and epochs, so it was not a pure LR ablation. It degraded whole-game floor and Act 1 boss survival. Dense shaping added a potential using .3 player-HP fraction minus .5 enemy-HP fraction, included hallways and used lambda .95. It remained essentially flat in full-game and boss-panel outcomes. The implemented terminal potential should not be casually called policy-invariant shaping.

Earlier imitation also failed. Approximately 51k privileged native-search DAgger labels worsened full-game progression and Act 2 boss pass. Approximately 18.7k RNG-resampled boss labels reduced the Act 2 panel from 29.5% to 23.3%. These older datasets/search backends have fault and information-boundary qualifications. They are not interchangeable with the corrected teacher-played data below.

Possible mechanisms include narrow-state forgetting, sparse credit, incomplete representation, noisy teacher labels and failure to execute coherent plans after copying isolated choices. The results do not uniquely identify one cause. The claim supported by the tests is that these particular recipes did not improve full-game play.

Source: [combat findings](rl_findings_2026-10-13.md), preserved `combat_bossrl_v1/v2`, `combat_v3_shaped`, and combat-distillation configs.

## 7. Clean-backend continuations: exact control and card exploration

The new overnight control restarted from update 20 with a fresh optimizer and corrected 0.1.3. Combat used exact argmax and remained bit-identical. Settings were eight iterations, 768 games each, gamma/lambda 1, LR 3e-4, three epochs, 4,096 minibatch/1,024 microbatch, entropy 0 and KL .05. The matched variant changed only card-reward sampling temperature to 4; rest/map/shop sampling and deployed argmax stayed unchanged.

Each collected 6,144 valid games, without faults. The control retained 387 optimizer updates and the card-temperature pilot 380. Evaluation on the same 1,024 DEV subset gave parent 57, control 58 and card-temperature 57. There was no demonstrated win improvement. The card-temperature condition also lost progression relative to the control. These runs show the previous successful recipe had plateaued at this checkpoint under these bounds; they do not prove all further OOC learning is impossible.

Additional strategy tests were rejected: suppressing the Neow boss-relic swap 29/1,024, forcing the policy's removal target 41, and forcing an unupgraded Strike removal 49, against parent 57. A plausible strategic story was not sufficient to adopt a rule.

Source: [overnight log](rl_overnight_2026-10-14.md), fixed plans, collection receipts and checkpoint audits.

## 8. Corrected teacher-played combat imitation and endgame retention

A separate clean TRAIN corpus contains 7,770 completed fights and 19,660 sampled public states from 512 worlds. A corrected native teacher then actually played 128 paired fights: 32 each Act 2/3 boss and elite. All 3,217 saved labels and trajectories were independently replayed, including public-state hashes, selected actions, teacher steps and outcomes.

Search improved remaining HP in Act 2 bosses and Act 2/3 elites, by roughly 8–10 HP per fight on average with nominal paired intervals excluding zero. Act 3 boss results were inconclusive. We therefore used 2,460 labels from BOSS2/ELITE2/ELITE3 and 1,072 balanced rehearsal states; BOSS3 teacher targets were excluded. Input/validation worlds were separated: 349 fit and 87 held.

All three pilots started afresh from update 20, froze all 20 OOC tensors, used AdamW LR 1e-4/weight decay .01, batch 256, four epochs, beta 1 KL rehearsal and seed 17:

| Pilot | Supervised target | Optimizer steps | Full-game DEV wins /1,024 |
| --- | --- | ---: | ---: |
| Soft votes | Complete native vote distribution | 32 | 50 |
| Hard action | Exact played plurality/tie rule | 32 | 52 |
| HP-curated hard action | 1,404 labels from better-HP fights; other 1,056 states moved to rehearsal | 20 | 46 |
| Parent | Original update 20 | — | 57 |

The curated condition retained all 3,532 observations/worlds but used fewer supervised batches; it is not an equal-update causal test of filtering. Remaining HP omits potion use, max-HP changes and future-run value. All pilots had zero faults and failed their win gates. None is promoted.

The soft model's held vote CE improved 3.934->1.663, while agreement with the actual teacher action declined 44.4%->42.5%. We also found a 3.7% validation tie-rule mismatch. Exact candidate-feature aliases were rare: 20/2,460 observations, with only one later identical candidate chosen by the teacher. A separate 32-state sanity model fitted all selected targets; it is explicitly diagnostic-only, not evidence of generalization or a candidate player.

The requested public-context ablation kept the curated model everywhere except the Act 3 final boss, where the original head was restored. It gave 61/1,024 versus curated 46 (paired p=.00813) and parent 57 (p=.724). Recorded earlier boss HP matched on every world; all 780 unswitched worlds matched outcomes/steps. It recovered damage from fine-tuning but did not establish superiority to the parent. Secret Portal makes five legitimate final bosses occur before floor 50: the original act/room gate correctly includes them. A fixed-floor diagnostic undercounted entrants and initially caused a false scope/victory concern, which was explicitly retracted after public replays. The narrower floor 50 secondary run also gave 61 wins and is retained separately.

Source: [complete overnight log](rl_overnight_2026-10-14.md), paired decision/acceptance receipts and verified review archive.

## 9. Fight-value learning: diagnostic models, not deployed policies

Two small classifiers fitted the 512-world corpus with fixed 20 epochs, AdamW LR .001/weight decay.01, batch 1,024, seed 17 and whole-world 409-fit/103-held split. Each fight was balanced rather than treating its multiple sampled states as independent fights. Legacy input was the 256 state features plus mean 128 candidate features. The augmented model added a zero-initialized residual branch over 35 explicit public enemy/card fields, including enemy max HP/block and temporary card flags.

Act 2 boss entry Brier (lower is better, 70 held fights) was .2469 for the fit-derived group prior, .2382 legacy and .2348 augmented. Within-fight prediction was stronger. Act 3 boss entry Brier (18 held fights) was .1516 prior versus .2980/.3041 learned, worse than the simple reference. These labels describe outcomes under the frozen policy, not optimal fight value. Neither predictor was deployed in planning, and their scores do not establish game improvement.

## 10. Play-time search: strongest current improvement, no training

Canonical RNG-resampled native search uses 24 samples times 1,000 simulations per eligible decision, votes over actions and single-observation learned argmax when it abstains. It canonicalizes unordered piles, reseeds battle RNGs and derives sampling seeds from public observation hashes. The actor weights do not change. Other latent native fields/equal-public-key ties remain uncertified; this is a defined native-assisted simulator condition, not a certified public-only planner.

| Same-cohort comparison | Learned only | Boss search | Boss+elite | Boss+elite+ordinary hallway |
| --- | ---: | ---: | ---: | ---: |
| 4,096 repeated DEV | 219 (5.35%) | 251 (6.13%) | 344 (8.40%) | Not run |
| 4,096 fresh frozen confirmation | 213 (5.20%) | Not run | 333 (8.13%) | Not run |
| 512 repeated DEV prefix | 29 (5.66%) | 38 (7.42%) | 42 (8.20%) | 98 (19.14%) |

Fresh boss-plus-elite confirmation has paired win p=9.14e-13 and floor gain 1.846, CI[1.610,2.072], zero faults. The broader hallway pilot has a large paired gain over boss-plus-elite, but remains a smaller repeatedly used DEV panel. No broader-scope extension or more training was launched after the requested report checkpoint. Search acts only from Act 2 onwards; the hallway condition adds ordinary MONSTER rooms, not all event fights.

## 11. What is established and what is not

Established: the first campaign learned Act 1; later OOC macro credit, measured rule distillation and lambda 1 produced the retained stronger lineage; recent unchanged OOC continuation plateaued; combat adaptation can harm endgame performance despite better progression; native search materially improves the declared condition and the boss/elite gain survives fresh confirmation.

Not established: a generally better newly trained combat policy; broadly calibrated Act 3 value prediction; a cause unique to critic interference or representation; superiority of the composed 61-win player over parent 57; fresh generalization of the 19.14% hallway pilot; fully public-state-only native search; general game mastery.

Historical RNG/launch gaps cannot be repaired by archiving. New jobs record source/native identities, exact settings and inputs, ancestry, optimizer/RNG states and fault/replay information. The current code/docs are pushed on the two owned research branches. Checkpoints/datasets/binaries are Git-ignored local archives and require a separate explicitly approved publication/backup path; a Git push alone does not copy them.
