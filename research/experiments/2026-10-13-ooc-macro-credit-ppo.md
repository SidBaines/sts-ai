# Out-of-combat PPO with macro credit, rule distillation and lambda 1

**Status:** finished 14 October 2026. **Verdict:** freezing combat, distilling
measured strategic rules and using long-horizon credit produced the champion:
round-2 PPO update 20, later measured at 5.20% win rate on 4,096 fresh
confirmation worlds in environment 0.1.4.

## Question

Joint training was regressing; could better credit and demonstrations improve
strategy? The approaches were [out-of-combat PPO with macro
credit](../strategies.md#out-of-combat-ppo-with-macro-credit) and [distilling
strategy rules](../strategies.md#distilling-strategy-rules).

## Design

Combat stayed fixed while out-of-combat (OOC) choices learned from whole
games: Ironclad, Ascension 0, Acts 1–3, starting from the [long
curricula](2026-10-12-long-curricula.md) composite.

- **Credit:** a macro transition sums rewards from one OOC choice to the next,
  crossing combat. Next-state value comes from the OOC critic, the network
  predicting future reward. Combat weights stayed unchanged. Collection used
  scaled-greedy combat; evaluation later used exact argmax.
- **PPO:** proximal policy optimization, with 768 new training games per
  update, learning rate 0.0003, three epochs, minibatches of 4,096, clipping
  0.2, no entropy bonus and a Kullback–Leibler divergence guard of 0.05.
  Rewards stayed at 0.05 per floor, +1 for victory and −1 for defeat, without
  discounting. Act 4 keys were filtered.
- **Rules and labels:** menu overlays changed choices only when a rule fired.
  The retained rules were smithing (upgrading a card) at ≥85% of maximum HP
  (hit points) and taking an offered elite at ≥90% HP. Smith distillation used
  2,877 labels plus 4,404 rehearsal decisions, 12 epochs, learning rate
  0.0003, batch 256 and 10% whole-world validation. Cross-entropy fitted
  labels; a divergence penalty of weight 1 preserved other choices. Round 2
  used 5,198 labels and 16 epochs at the same learning rate and batch size.
  Labels came from the smith-only policy, then were fitted into its lambda-1
  PPO update-12 descendant.
- **Credit comparison:** generalized advantage estimation (GAE) combines
  rewards with predicted future values. The sweep compared lambda 0.95, 0.99
  and 1 from the smith-distilled policy on the same training worlds. At 1,
  complete-game returns supply credit; the critic is only an advantage
  baseline.
- **Evaluation and compute:** historical environment 0.1.1 study results used
  1,024 reused evaluation worlds, 1,024 fresh confirmation worlds and a
  4,096-world win set, all from the training split. Win rate became primary
  and mean floor secondary; comparisons paired worlds. The first eight-update
  pilot took about 42 minutes on 12 CPU workers; no total budget is recorded.
- **Decision rules and approval:** after the pilot, a replication proposal
  required at least +0.5 mean floors across seeds. Later thresholds and
  checkpoints were selected during the study. Sid specified Act 3 victory as
  the goal on 13 October; no separate dated approval or prospective final win
  gate is recorded.

## Results

Round-2 update 20 became the champion. A later matched **Exploratory** check
on 4,096 DEV worlds in environment 0.1.3 separated it from update 12, with
zero faults. Intervals below are 95%.

| Round-2 checkpoint | Wins of 4,096 | Mean floor | Paired mean floor difference vs update 12 [95% interval] |
|---|---:|---:|---:|
| Update 12 | 146 (3.56%) | 35.01 | — |
| **Update 20** | **220 (5.37%)** | **35.65** | **+0.64 [+0.35, +0.92]** |

The paired win test gave p = 4 × 10⁻⁶; no win-difference interval is reported.
The same retained update 20 later won 219/4,096 DEV worlds (5.35%) and
213/4,096 fresh confirmation worlds (5.20%) under single-observation inference
in environment 0.1.4, with zero faults. These are separate simulator
conditions.

Earlier OOC gains survived **Confirmed** checks on 1,024 requested fresh
worlds per agent on the historical build; differences below use worlds valid
for both candidate and original.

| Agent | Mean floor | Paired difference vs original [95% interval] |
|---|---:|---:|
| Original composite | 28.82 | — |
| OOC PPO, update 36 | 31.79 | +2.99 [+2.39, +3.57] |
| Update 36 plus distilled smith rule | 33.12 | **+4.29 [+3.67, +4.87]** |

Smith distillation added +1.34 floors [+0.85, +1.83] over update 36 on these
fresh worlds, without a rule at deployment. A second training seed replicated
early OOC gains: +1.37 floors [+0.85, +1.90] at update 12 against the original
on 1,024 requested reused worlds (**Exploratory**).

Menu tests showed which strategic changes deserved demonstration. All rows
below are **Exploratory**, with paired differences against the stated parent.

| Overlay | Parent; requested worlds | Mean floor difference [95% interval] |
|---|---|---:|
| Smith at HP ≥85% | OOC update 28; 1,024 | +1.35 [+0.95, +1.79] |
| Smith at HP ≥50% | OOC update 28; 1,024 | −0.57 [−1.05, −0.10] |
| Suppress potion purchases while removal is affordable | OOC update 28; 1,024 | +0.12 [−0.36, +0.60] |
| Take an offered elite at HP ≥90% | Smith-distilled policy; 4,096 | −0.06 [−0.29, +0.16] |
| Skip every card reward after Act 1 | Smith-at-70% overlay; 1,024 | −1.63 [−2.04, −1.22] |
| Reject Fusion Hammer and Coffee Dripper | Round-2 distilled policy; 4,096 | −0.43 [−0.53, −0.34] |

The elite overlay won 94 games against 77 on 4,096 requested worlds (paired p
= 0.10). Pooling with an earlier 1,024-world test on a different parent gave p
≈ 0.015, motivating adoption. Potion conservation showed no clear difference
on 1,024 requested worlds.

Lambda 0.95 erased smithing. This **Exploratory** update-16 comparison against
the smith-distilled start used 1,024 requested reused worlds; sampled smith
rates came from training games.

| GAE lambda | Mean floor difference [95% interval] | Sampled smith rate at update 16 |
|---|---:|---:|
| 0.95 | −0.54 [−1.07, −0.01] | 14.5% |
| 0.99 | +0.72 [+0.16, +1.24] | About 35% |
| 1 | +0.80 [+0.25, +1.36] | About 45% |

Lambda 1 at update 12 added +0.87 floors [+0.59, +1.13] over the distilled
start on the requested 4,096-world win set; its win gain was inconclusive
(paired p = 0.58). After smith/elite distillation and more lambda-1 PPO,
round-2 update 12 won 151/4,016 valid worlds (3.76%).

Earlier gradient probes found little agreement between disjoint 64-game
halves: cosine similarity, measuring direction agreement, was 0.07 for combat
and 0.17 for OOC.

## Conclusion

PPO refined useful habits supplied by demonstrations. In 119 games at update
28, it never smithed at 557 rest sites, took an elite at 211 offered menus or
removed a card at 571 affordable menus. Extremely low sampling probabilities
explained this exploration barrier.

Joint-training probes found noisy gradients and likely combat-boundary critic
bias toward avoiding fights. On the same 512 sampled games, macro
smith-minus-rest advantage at 70–85% HP changed from −0.049 at lambda 0.95 to
+0.083 at 1. The likely explanation is that the critic represented healing
better than an upgrade's delayed payoff; lambda 1 reduced that dependence.

Caveats:

- Macro credit, frozen greedy combat and larger batches changed together;
  their individual contributions were not established by completed ablations.
- Later selection used reused worlds. Historical sets were not official
  DEV/final cohorts.
- Historical faults were excluded, not defeats; exact paired denominators are
  sometimes absent. Scaled-greedy combat could break ties randomly.
- Environment 0.1.1, corrected 0.1.3 and 0.1.4 results are not one learning
  curve.
- The later lineage had one training seed per stage. Lambda 1 was noisier and
  not shown superior to 0.99; it did not discover smithing without
  demonstrations in 16 updates, and still allowed smithing to drift downward
  in round 2.
- The recipe is reconstructable, but exact historical replay is not
  established.

## Next

Prioritize measured rules and a critic that sees deck quality, with
fresh-world confirmation. Checkpoints through round-2 update 32 established no
reliable further gain. See [overnight tests](2026-10-14-overnight-tests.md)
for continuations and [combat PPO](2026-10-13-combat-ppo.md) for combat
training.

## Log

- 13 Oct: diagnosed noisy joint updates and credit bias at combat boundaries.
- 13 Oct: isolated OOC PPO improved progression and replicated early across
  seeds.
- 13 Oct: distilled high-HP smithing; the combined gain survived fresh worlds.
- 13 Oct: smithing regressed under lambda 0.95; the sweep motivated lambda 1.
- 13 Oct: distilled smith/elite choices and resumed OOC PPO without live
  rules.
- 14 Oct: corrected-environment comparison retained round-2 update 20 as
  champion.

Code: `src/sts_rl/ooc_trainer.py`, `src/sts_rl/distill.py`,
`src/sts_rl/common.py`, `scripts/rl/train_ooc.py`,
`scripts/rl/eval_policy.py`, `scripts/rl/collect_rule_labels.py`,
`scripts/rl/distill_ooc.py`; audit snapshot `1481be4` (not exact launch
binding). Data: runs `ooc_main_s1`, `distill_smith85_i36`, `ooc_pfd_lam1`,
`distill_r2_lam1i12`, `ooc_r2_lam1`, `fresh_eval`, `win_eval` (local).
