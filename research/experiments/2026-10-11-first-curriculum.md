# First curriculum: imitation, DAgger and PPO, from fights to full games

**Status:** finished 12 October 2026. **Verdict:** full learned control
cleared 20 of 21 frozen final Act 1 worlds, but neither retained player won a
full game.

## Question

The question was whether teacher imitation followed by reinforcement learning
could teach both fighting and strategy, progressing from fights to full games:
[curriculum imitation and PPO](../strategies.md#curriculum-imitation-and-ppo).
The practical target was Ironclad, Ascension 0, ordinary Acts 1 to 3.

## Design

The campaign trained specialists, then trained both together without native
combat support.

- **Policy:** each legal action was scored from 256 public state features and
  128 candidate features. Six screens crossed three model sizes with or
  without mean menu context, an average representation of the other choices.
  They selected 2.3 million parameters per network without menu context. The
  composite contained separate combat and out-of-combat (OOC) networks.
- **Data:** banks held 49,944 Act 1 labels and 560,620 expanded multi-act
  labels. Worlds were hash-split 80% training, 10% development and 10% final
  before fitting. Only training subsets were fitted. Combat copied the
  privileged search teacher, normally at 200 simulations; OOC copied the
  native script.
- **Learning:** behavioural cloning fitted teacher choices. DAgger added
  search labels on learner-visited combat states, mixed with earlier teacher
  examples. Proximal policy optimization (PPO) trained on sampled play over
  legal actions. Stages were isolated fights, OOC with native combat,
  integrated Act 1, then full Acts 1 to 3. Both networks were updated in the
  integrated stages.
- **Settings:** ten combat controls covered learning rate, clipping,
  exploration, discount, rewards, health bonuses and learning from scratch.
  Selected combat used learning rate 0.0001, probability-ratio clip 0.1,
  discount 1 and entropy coefficient 0.01. Corrected OOC used learning rate
  0.0003 and floor-progress reward weight 0.05, with the same clip, discount
  and entropy and no HP bonus. Native-resolved combat counted as one external
  OOC decision for discounting.
- **Evaluation:** selection used 32 OOC and 48 integrated DEV worlds, plus
  fight panels. Final assessment used 160 Act 1 fights from 20 worlds, 32
  later-act fights from 25 worlds, and 21 worlds each for OOC Act 1,
  integrated Act 1 and full games. Metrics were fight wins, act clears,
  full-game win rate, mean floor, HP cost and simulator faults. References
  used matched tasks: random, public heuristic and privileged native, whose
  OOC choices were scripted. Fight/OOC assessments used specialists; full
  games tested both the frozen joint player (`joint_act3_0030.pt`) and
  unmodified specialist composition. Integrated Act 1 used the joint player.
- **Selection and budget:** the plan preceded substantial training and
  selected on DEV success, HP cost and progression, then froze candidates
  before final evaluation. Complete requested cohorts and separate fault
  accounting were required; no numerical full-game improvement threshold was
  recorded. Sid approved about ten hours of laptop compute on 11 October.
  Selection froze after training ended; no tuning followed final results.

## Results

Full learned control was clearly better than native in Act 1: **Confirmed**
evidence from frozen final cohorts on environment 0.1.1. Success means the
fight win, act clear or full-game victory indicated in each row.

| Frozen assessment | Learned successes / requested | Native successes / requested | Paired success difference, points [95% interval] |
|---|---:|---:|---:|
| Isolated Act 1 fights | 145/160 | 149/160 | −2.3 [−6.2, +0.7] |
| Isolated Acts 2 to 3 fights | 25/32 | 26/32 | −2.0 [−14.0, +10.0] |
| Act 1 OOC, native combat | 17/21 | 6/21 | +50.0 [+30.0, +70.0] |
| Act 1 full learned control | **20/21** | 6/21 | **+65.0 [+45.0, +85.0]** |
| Full three-act learned control | **0/21** | 0/21 | Uninformative: all valid games failed |

World-bootstrap 95% intervals weight common valid worlds equally: 20 for Act 1
fights, 25 for later fights, and 20 for each Act 1 comparison with native.
Counts retain requested tasks. Native faulted once in each whole-run cohort;
the joint player had 20 defeats and one simulator fault in full games. Other
learned assessments had no faults. No final assessment was truncated;
duplicate Act 1 fight panels are counted once.

![Act 1 full-control development clears on the corrected environment](../figures/2026-10-11-first-curriculum/native011_act1_development.png)

Act 1 clears on 48 fixed DEV worlds during joint training on environment
0.1.1; red markers flag faults or cut-off games.

The following screens are **Exploratory**, with no reported paired intervals.

- **Combat:** the retained specialist won 44/48 Act 1 fights from 21 DEV
  worlds and 30/32 later-act fights from 30 DEV worlds on environment 0.1.0.
  Comparing 50, 200 and 1,000 teacher simulations on 48 fights from 21 DEV
  worlds retained 200: 1,000 brought no consistent improvement.
- **Corrected OOC:** fresh imitation cleared 12/32 DEV worlds. Subsequent PPO
  reached 29/32. Moving the policy-divergence stop check before the optimizer
  step gave 31/32 with zero faults versus 29/32 with one fault in matched
  60-second pilots; the former was retained. Random, native and public
  heuristic references cleared 3, 15 and 22 of the same 32 worlds.
- **Corrective OOC imitation failed:** on 256 corrected training frontiers,
  lookahead changed 92 labels using up to four candidates and 30-decision
  continuations. From the same frozen OOC parent, 20-update scripted/lookahead
  fits gave 19/32 DEV clears with one fault and 16/32 with none: both were
  rejected despite the lookahead proxy improving.
- **Integrated play:** with the selected OOC network, the retained combat head
  cleared 21/32 DEV worlds, against 11/32 and 13/32 for two health-bonus heads
  (one fault each), and 7/32 for shared imitation. Combat histories differed.
  Joint Act 1 training improved the composed baseline from 29 to 45 clears on
  a separate fixed 48-world panel. Short later-act bridges and
  credit-assignment pilots failed to improve the 48-world full-game DEV
  result. The specialist composition also had zero final victories: 21
  defeats, no faults.

Against the public heuristic on 21 final worlds, full learned control cleared
20 versus 4: +76.2 points [+57.1, +90.6], Confirmed. OOC with native combat
cleared 17 versus 13: +19.0 points [−4.8, +42.9], Confirmed but inconclusive.

## Conclusion

The curriculum established Act 1 play; isolated later-act fighting did not
establish full-game competence with learner-built decks. The contributions of
imitation, DAgger and PPO were not isolated.

Caveats:

- Environment 0.1.0 left the automatic-combat character uninitialized,
  corrupting class-specific card/potion generation. All earlier
  automatic-combat OOC outcomes, transitions and lookahead labels were
  quarantined. Environment 0.1.1 fixed this and affected data were
  regenerated. Earlier full-control Ironclad combat passed scoped
  compatibility checks, without certifying other simulator behaviour.
- DEV worlds were reused; imitation had initialization repeats, but the final
  curriculum was not replicated across training seeds. Final fight comparisons
  remain inconclusive; zero wins in 21 games cannot establish a true zero win
  rate.
- Two early integration checks used one world from the final split, outside
  the final cohorts, so the final split was not entirely untouched.
- Final OOC/full-game horizons were 2,048 decisions; DEV OOC used 1,024 and
  full games 4,096. HP cost was net starting minus ending HP; act-transition
  healing obscures earlier damage and limits the OOC lookahead proxy.
- The teacher saw hidden state; the encoder could omit useful public
  information. Faults were excluded from learning and paired estimates but
  retained in counts.

## Next

Target later-act strategy, learner-induced fight states and longer credit
assignment on new worlds. [Longer curricula](2026-10-12-long-curricula.md)
continued from this player.

## Log

- 11 Oct: Sid approved the curriculum; size and context screens selected the
  large network.
- 12 Oct: the automatic-combat defect forced an OOC restart on environment
  0.1.1.
- 12 Oct: corrected OOC PPO was retained; corrective imitation and health
  alternatives were rejected.
- 12 Oct: short later-act bridges failed; more time went to three-act
  training.
- 12 Oct: selection froze; final evaluation showed Act 1 competence and no
  full-game wins.

Code: `src/sts_policy/`, `scripts/train_nonllm.py`,
`scripts/run_nonllm_campaign.py`, `scripts/evaluate_nonllm_baselines.py` and
`scripts/report_nonllm_campaign.py` (before the experiment-folder convention;
no single commit recorded). Data: runs `nonllm_v1/campaign`,
`nonllm_native_revision/campaign` and `nonllm_native_revision/final_report`
(local).
