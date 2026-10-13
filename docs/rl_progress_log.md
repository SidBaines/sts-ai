# RL progress log (most recent first)

**Live dashboard: <http://127.0.0.1:8791>** (all runs, eval curves, paired gain vs original, learning health)
· queue: `.venv/bin/python scripts/rl/queue_runner.py status`
· runs and logs: `data/rl_runs/<name>/log.jsonl`
· background: [rl_investigation_2026-10-13.md](rl_investigation_2026-10-13.md)

If the dashboard is down: `.venv/bin/python scripts/rl/dashboard.py start`. If the
queue runner is down: `.venv/bin/python scripts/rl/queue_runner.py start` (it pauses
runs on battery and resumes on AC).

Reference point for everything below: the original composite (`joint_act3_0030.pt`),
greedy, dead-action filter, 1,014 valid TRAIN-eval worlds (seeds from 5e9):
**29.27 ± 0.26 mean floor**, Act 1 boss pass 88.1%, Act 2 boss reach 48.9% / pass
23.4%, 1 win.

### Where we are (pinned; updated in place)

All results are greedy, paired vs the original on ~1,010 worlds.

| | Mean floor | Paired vs original | Wins | Notes |
| --- | --- | --- | --- | --- |
| Original | 29.27 | — | 1–2 | |
| OOC PPO, update 28 | 31.92 | +2.69 [2.12, 3.24] | 11 | E8 recipe; seed 2 replicates (+1.37 at update 12 vs +1.57) |
| OOC PPO, update 36 (`ooc_main_s1/iter_0036.pt`) | 32.34 | **+3.14** [2.58, 3.70] | 11 | still improving |
| Update 28 + "smith whenever HP ≥ 85%" | 33.26 | ≈ +4.0 | 16 | rule: +1.35 [0.95, 1.79] over update 28 |
| **Update 36 + rule distilled into the network** (`distill_smith85_i36/distilled.pt`) | **33.31** | **+4.05** [3.43, 4.65] | 19 | no rule at deployment; **fresh worlds: +4.29 [3.67, 4.87], wins 3 → 22** |

**Resumed 18:53.** Running: `ooc_ppo_from_distilled` (to 16), 4,096-world win evals, and combat RL v3 (soon).
Stopped:
- `ooc_main_s1` at update 45. This was the control lineage.
- `ooc_branch_smith70`: PPO trained with the smith rule.
- Rule-sweep evals: elites, card skips, and main@36 + rule.
- `strat8` (exploration) is stopped at update 38.

**Main open problem:** the OOC policy's logits are saturated, and
strategic-decision credit is noisy (per-decision effect ~0.02 vs noise
~0.15). Exploration alone teaches smithing very slowly; paired-eval rule
tuning is fast.

--- | --- | --- | --- | --- |
| Original | 29.27 | — | 1 | |
| OOC PPO, update 28 (`ooc_main_s1/iter_0028.pt`) | 31.92 | **+2.69** [2.12, 3.24] | 11 | E8 recipe |
| …plus the "smith at ≥70% HP" rule | **33.06** | ≈ +3.8 | 13 | +1.15 [0.70, 1.60] over update 28 |

**Resumed 18:53.** Running: `ooc_ppo_from_distilled` (to 16), 4,096-world win evals, and combat RL v3 (soon).
Stopped:
- `ooc_main_s1` at update 45. This was the control lineage.
- `ooc_rep_s2`: seed-2 replicate from the original.
- `ooc_branch_strat8`: exploration on rest/map/shop menus.
- `ooc_branch_smith70`: PPO with the smith rule.

Branches vs main are compared at updates 36/40/44 with
`scripts/rl/compare_evals.py`.

**Main open problem:** the OOC policy's logits are saturated, so PPO cannot
explore strategic options. Exploration is the lever.

---

## 2026-10-13 19:18 BST — Elite-seeking raises wins at no floor cost (4,096-world win set); potions-at-boss lead

- **Elite-seeking (take an offered elite path at HP ≥ 90%)**, on top of the
  distilled agent, 4,096 fresh worlds:
  - mean floor −0.06 ± 0.12 (null);
  - **wins 77 → 94 (1.91% → 2.34%)**, discordant 39/56, McNemar p = 0.10;
  - the earlier 1,024-world test on a different base and worlds was 13 → 22
    (p = 0.049).
  - Pooled across the two independent world sets: 43 vs 69 discordant worlds,
    sign test **p ≈ 0.015**.
  - **Adopt for the win objective.** The 80% threshold is being tested.
- **Potions at boss entry** (distilled-agent bank, 1,012 Act 2 boss fights;
  correlational):

  | Potions held at entry | Act 2 boss win rate |
  | --- | --- |
  | 0 | 23% |
  | 1 | 23% |
  | 2 | 39% |
  | 3 | 45% |

  - **36% of Act 2 boss fights start with no potions**, even though 66% of
    shop purchases are potions.
  - Won fights also started with more HP (90% vs 75%).
  - Causal test queued: no potion drinking in hallway fights while HP ≥ 30%
    (`--save-potions-above 0.3`).
- **Combat v3** after 3 updates: panel Act 2 boss 33.3 → 34.8%, Act 3 boss
  17.5 → 20%. Too early to call.

---

## 2026-10-13 19:09 BST — Win-set baseline 1.91%; PPO keeps the distilled behaviour and builds on it; combat v3 running

- **Best agent (distilled) on the 4,096-world win set: 1.91% win rate**
  (77/4,041); mean floor 33.19.

  | Stage | Rate |
  | --- | --- |
  | Pass the Act 2 boss | 31% |
  | Reach the Act 3 boss | 11.5% |
  | Beat the Act 3 boss, once reached | 16.6% |

  Completing the game ≈ reach the Act 3 boss × beat it. Both boss gates
  matter.
- **`ooc_ppo_from_distilled`** (plain PPO, no rule), paired vs its distilled
  start:

  | Update | Paired (floors) | Wins (19 at start) |
  | --- | --- | --- |
  | 4 | +0.08 ± 0.25 | 20 |
  | 8 | +0.40 ± 0.26 (CI [−0.11, +0.92]) | 25 (p = 0.38) |

  Sampled smith rate stays at 41–43%. **Plain PPO keeps the learned behaviour
  and appears to build on it.** Not yet significant.
- **Combat v3** (`combat_v3_shaped`) is running.
  - New bank from the distilled agent: 21,051 fights, 16,860 of them hallway.
  - Baseline held-out panel (1,537 fights): Act 2 boss 33%, Act 3 boss 17.5%,
    Act 2 elite 74%, Act 3 elite 75%, Act 1 boss 94%, hallways 94–100%.
  - Full-game paired evals at updates 10, 20 and 30 vs the distilled agent.

---

## 2026-10-13 18:55 BST — Resumed; branch pushed; win-rate measurement; combat RL v3 (dense credit) prepared

- **Pushed** `claude/slay-spire-rl-debug-da6d22` to origin. All code, docs and
  the log are on GitHub. `data/` (checkpoints, evals, run logs) is gitignored
  and stays local.
- **Resumed** `ooc_ppo_from_distilled` from update 4, with 4 collectors and the
  machine to itself. Main is not resumed: it was flat from update 36 to 40 and
  its job as the control is done.
- **The reward already weights wins heavily.** A run returns 0.05 × floor − 1
  on death and about +3.55 on a win, so **a win is worth ~41 floors**; +1
  point of win rate ≈ +0.4 mean floor. The real issue for the new objective
  is **measurement**: at a ~2% win rate, 1,024 worlds cannot resolve 1-point
  changes.
  - New `paired_wins` gives an exact McNemar test on discordant worlds, and
    `compare_evals.py` prints it.
  - A fresh **4,096-world win set** (TRAIN seeds from 6e9) is reserved for
    win-rate claims.
- **Queued on the win set:** distilled vs distilled + elite-seeking at HP ≥ 90%
  and ≥ 80%. Elite-seeking raised wins 13 → 22 (p = 0.049) at 1,024 worlds.
- **Combat RL v3 prepared** (`combat_trainer --shaping-hp/--shaping-enemy`).
  - Per-step potential shaping: φ = 0.3·HP fraction − 0.5·enemy HP fraction,
    with GAE λ = 0.95, so a lost fight still credits damage dealt and a won
    fight credits HP kept.
  - Regression-tested: the Monte-Carlo return telescopes to
    terminal + φ(final) − φ(start).
  - It trains on a new bank from the best agent that **includes hallway
    fights**, so the shared combat head does not regress elsewhere.

---

## 2026-10-13 17:54 BST — PAUSED (laptop closing); objective set to Act 3 boss wins; resume notes

- **Objective (from Sid):** "complete the game" means an **Ascension 0 Act 3
  boss win** (`victory`). Win rate is now the headline metric; mean floor is
  secondary. The current reward is floor-progress based, so it does not target
  wins directly. Next: weight victories in the reward and re-test
  elite-seeking (−0.26 floors, but wins 13 → 22).
- **Stopped cleanly** (all checkpoints are atomic). The queue runner is
  stopped and no jobs are running.

  | Run | State | Last eval |
  | --- | --- | --- |
  | `ooc_main_s1` | stopped after update 45 (`latest.pt` = 45) | update 40: 32.29, +3.10, 11 wins |
  | `ooc_branch_smith70` | **done** (update 44) | update 44: 33.01, +3.71, 24 wins |
  | `ooc_ppo_from_distilled` | stopped after update 4 (`latest.pt` = 4) | update 4: 33.36, +4.09 vs original, 20 wins; vs the distilled start +0.08 ± 0.25. Smithing kept (sampled 37% → 41%) |

- **To resume:**
  1. `.venv/bin/python scripts/rl/queue_runner.py start`
  2. Re-add main as `ooc_main_s1_ext6`, with the same command as `ext5` plus
     `--resume`.
  3. Re-add `ooc_ppo_from_distilled` as `ooc_ppo_from_distilled_r1`, with the
     same command plus `--resume`.
- Then: eval `ooc_ppo_from_distilled` at 4/8/12/16, and decide on a
  win-weighted reward.

---

## 2026-10-13 17:45 BST — Where the best agent dies: the Act 2 boss (44% of runs)

- **Death locations on fresh worlds** (counts per 1,024):

  | Agent | Act 1 boss | Act 2 hallway | Act 2 boss | Act 3 hallway | Act 3 boss | Win |
  | --- | --- | --- | --- | --- | --- | --- |
  | Original | 126 | 382 | 391 | 74 | 24 | 3 |
  | Update 36 | 92 | 267 | 466 | 94 | 74 | 11 |
  | Distilled | 77 | 241 | **449** | 123 | 94 | 22 |

  Act 2 boss pass rate: 20.5% → 27.8% → **34.6%**. Entry HP 62 → 65 → 67.
- **Act 2 boss win rate by identity** (update-8 fight bank, before smithing):

  | Boss | Win rate |
  | --- | --- |
  | Bronze Automaton | 18% |
  | Collector | 25% |
  | Champ | 27% |

  Entry HP is ~78% for all three. No single boss is the problem; it is deck
  strength plus combat play.
- **Open decisions for Sid:**
  1. Objective: mean floor or win rate. Elite-seeking trades −0.26 floors for
     wins 13 → 22.
  2. krill-mill compute.
  3. Next lever for the Act 2 boss: combat RL with dense (HP/damage) credit,
     more deck-strength levers via the overlay → distil loop, or test-time
     public-information search in boss fights.

---

## 2026-10-13 17:44 BST — Confirmed on fresh worlds: distilled network +4.29 floors, wins 3 → 22

To rule out selection bias from reusing the same eval worlds, the key agents
were re-evaluated on **1,024 fresh TRAIN worlds** (seeds from 5.6e9, never used
before):

| Agent | Mean floor | Paired vs original | Wins |
| --- | --- | --- | --- |
| Original | 28.82 | — | 3 |
| OOC PPO, update 36 | 31.79 | **+2.99** [2.39, 3.57] | 11 |
| Update 36 + distilled smith rule | **33.12** | **+4.29** [3.67, 4.87] | **22** |

- Distilled vs update 36: **+1.34 [0.85, 1.83]**. This is larger than on the
  reused worlds (+0.96), so there is no sign of optimistic selection.
- **PPO from the distilled checkpoint** (no rule, no extra exploration) keeps
  smithing: sampled smith rate 37.4% → 39.3% over updates 1–2; training-game
  floor 32.5 → 33.0.

---

## 2026-10-13 17:31 BST — Distillation captures the whole rule; main flat over updates 36→40; card picks matter

- **Distilled network vs the rule it learned from:** update 36 + smith ≥85%
  applied at evaluation scores 33.40 (+1.05 [0.64, 1.45] vs update 36). The
  distilled network, with no rule, scores 33.31: **−0.08 ± 0.23**, the same.
  The network absorbed the rule's full value.
- **Update 40:**
  - main 32.29 (+3.10 vs original; vs update 36: −0.05 ± 0.23, flat);
  - `smith70` 33.06 (+3.78; **+0.78 [0.21, 1.41] vs main@40**; vs the
    distilled network −0.25 ± 0.28).

  Training *with* the rule for 7 updates is worth about the rule itself. No
  extra compounding is visible yet.
- **Skip every card reward after Act 1:** **−1.63 [−2.04, −1.22]** vs
  smith ≥70%. The policy's Act 2 card picks carry a lot of value; deck bloat is
  not the problem.
- **PPO from the distilled checkpoint, update 1:** sampled smith rate 37.4% of
  rest sites (main ~0.2%); training-game floor 32.5 (main ~31.4).

---

## 2026-10-13 17:23 BST — The network now smiths on its own (distilled): 33.31, +4.05 vs the original; PPO-from-distilled launched

- **Distilled the smith ≥85% rule into update 36's OOC head.**
  - Data: 2,877 rest-site labels (48% smith) plus 4,404 rehearsal decisions with
    a KL anchor; 12 epochs at lr 3e-4.
  - Held-out rest-site agreement with the rule rose 0.50 → 0.82–0.90.
    Agreement with the original on other decisions: 0.96.
- **Greedy, no rule at deployment: 33.31 mean floor.**
  - Paired **+0.96 [0.46, 1.45]** vs update 36.
  - **+4.05 [3.43, 4.65] vs the original**; wins 2 → 19.
  - **Best pure-network agent so far:**
    `data/rl_runs/distill_smith85_i36/distilled.pt`.
- **`ooc_ppo_from_distilled` launched** (seed 3, 16 updates). It runs the E8
  recipe from the distilled checkpoint, **with no rule and no extra
  exploration**. Watch its `smith_rate` and paired evals: they show whether
  plain PPO keeps, refines or unlearns the behaviour.
- **Skip every Act 3 card reward:** −0.09 ± 0.04 vs smith ≥70%; wins 13 → 9.
  Late card-taking is fine.

---

## 2026-10-13 17:12 BST — Elite-seeking trades mean floor for wins; rule-distillation experiment started

- **Overlay "take an offered elite path at HP ≥ 90%"** on top of smith ≥70%,
  paired vs smith ≥70% alone:
  - mean floor **−0.26 ± 0.18** (CI [−0.62, +0.10]);
  - **wins 13 → 22**. Of 17 worlds where only one variant won, 13 favour
    elites (sign test p ≈ 0.05);
  - more Act 1 deaths (70 → 83) and Act 3 deaths (94 → 107).
  - Elites buy late-game strength (relics) at early-death risk.
  - **Objective question for Sid:** we optimise a mean-floor progress reward.
    If the goal is win rate, wins should carry more reward weight, and
    elite-seeking may then be right.
- **Rule-as-initialisation experiment** (answering "can RL learn this without
  the rule?"):
  1. Collect rest-site labels from update 36 following the smith ≥85% rule
     (`scripts/rl/collect_rule_labels.py`; 600 TRAIN games from seed 8.5e9).
  2. Distil the labels into the OOC head with a KL anchor on other decisions.
  3. Run PPO from the distilled checkpoint **without** the rule.
  4. Track whether it keeps, refines or unlearns smithing.

---

## 2026-10-13 17:07 BST — Correction: main had not plateaued (+3.14 at update 36); smith ≥85% ≈ ≥70%

- **Correction to the 16:36 entry:** `ooc_main_s1` update 36 is **32.34,
  paired +3.14 (CI [2.58, 3.70])**, up from +2.65 at update 32. The "plateau"
  call at update 32 was premature, based on one noisy eval step. Main is still
  improving.
- **At matched update 36, vs main:**
  - `strat8` (exploration): −0.37 ± 0.26.
  - `smith70` (rule + PPO): +0.35 ± 0.28.

  Neither is significant.
- **Smith threshold 85%:** +1.35 (CI [0.95, 1.79]) vs update 28, and
  +0.18 ± 0.17 vs the 70% threshold. **The rule is robust from 70% to 85%;
  50% is harmful (−0.57).**
- **Queued:** main update 36 + smith ≥85% applied at evaluation only. This
  separates the rule's value from the value of training with it.
- **Elite and card-skip overlay evals** are next in the queue.

---

## 2026-10-13 17:02 BST — Replicate holds at update 12; the smith threshold is sharp (50% hurts, 70% helps)

- **Seed-2 replicate, update 12: +1.37 (CI [0.85, 1.90])**. Seed 1 was +1.57.
  The two seeds track each other within error across updates 4, 8 and 12.
- **Smith-threshold sweep**, paired vs update 28:

  | Smith whenever HP ≥ | Paired (floors) |
  | --- | --- |
  | 50% | **−0.57** [−1.05, −0.10] |
  | 70% | **+1.15** [0.70, 1.60] |
  | 85% | running |

  Smithing in the 50–70% band costs more than resting. This independently
  confirms the advantage diagnosis (low-HP smiths are very costly), and it is
  why undirected exploration taught PPO "smith is bad".
- The exploration branch `strat8` was stopped at update 38 to free CPU for the
  sweeps. Its `latest.pt` is kept, so it can resume.

---

## 2026-10-13 16:55 BST — Queue-runner bug killed the main run at update 36; fixed and resumed

- **Bug:** the stall watchdog looked for each run's log at
  `data/rl_runs/<queue name>/log.jsonl`. Resumed and branched runs write
  elsewhere: `ooc_main_s1_ext4` writes to `ooc_main_s1/`. For those, "last
  activity" was the launch time, so the watchdog killed healthy runs 40
  minutes after launch.
- `ooc_main_s1_ext4` was killed mid-update 36. Update 35's checkpoint was
  intact (atomic save). Its queued rescore then started in its place.
- **Fix:** activity is now read from the `--out` log and `stdout.log`.
  Log-less evaluation jobs get 3× the limit.
- The runner was restarted on the fixed code, and main resumed from update 35
  as `ooc_main_s1_ext5`. About 20 minutes of main-run time was lost.
- No other run was affected. The other renamed runs were all cancelled by me
  within 40 minutes.

---

## 2026-10-13 16:52 BST — Exploration alone is too noisy to learn smithing; tuning the strategy rule with paired evals instead

- **Update-36 evals**, paired:

  | Run, update 36 | Mean floor | vs original | vs main@32 |
  | --- | --- | --- | --- |
  | `strat8` (exploration) | 31.92 | +2.69 | **+0.08 ± 0.25** |
  | `smith70` (rule + PPO) | 32.70 | **+3.43** [2.81, 4.05]; wins 2 → 21 | +0.80 [0.30, 1.31] |

  `smith70` at update 36 vs update 28 + rule (33.06): −0.35 ± 0.28. Four PPO
  updates on top of the rule have not added anything yet.
- **Why exploration does not teach smithing.** With `strat8`'s update-36
  weights (T=8), sampled smith fell to ~2% of rest sites; PPO pushes it down.
  Macro advantages by HP, ~1,750 rest sites:

  | HP at rest site | Smith advantage (n) | Rest advantage |
  | --- | --- | --- |
  | <50% | **−0.41** (3) | −0.016 |
  | 50–70% | −0.14 (6) | +0.012 |
  | 70–85% | −0.09 (6) | +0.008 |
  | ≥85% | +0.03 (7) | −0.010 |
  | 100% | −0.01 (11) | −0.030 |

  - Exploration samples smith at every HP. The costly low-HP smiths dominate,
    and the network generalises "smith is bad" across HP.
  - The true high-HP benefit (~0.02 reward units, ~0.4 floors per decision)
    sits under advantage noise of ~0.15. It would take ~200 samples per HP
    bucket to resolve.
  - **Temperature exploration is the right idea but far too sample-hungry for
    small-effect strategic decisions under this credit noise.**
- **Pivot: tune strategy rules with paired evals.** 1,024 shared worlds give
  SE ≈ 0.25 floors, much lower noise than per-decision credit. Queued vs
  update 28:
  - smith threshold 50% and 85% (bracketing the 70% that gave +1.15);
  - elite-seeking (take an offered elite path at HP ≥ 90%) on top of smith at
    70%.
- **Seed-2 replicate stopped at update 12** after replicating (+1.46 at
  update 8). Its update-12 checkpoint is being evaluated separately.

---

## 2026-10-13 16:36 BST — The recipe replicates (seed 2 +1.46 at update 8); main plateaus; exploration has a cleaner gradient

- **Seed-2 replicate update 8: 30.70, paired +1.46 (95% CI [0.96, 1.98])** vs the
  original. Seed 1 was +1.14 at update 8. **The E8 recipe replicates across
  training seeds.**
- **Main has plateaued (WRONG, see correction above: update 36 is +3.14):** update 28 +2.69, update 32 +2.65 (CI [2.08, 3.21]).
  Training-game floor is flat at ~31.0–32.0. This fits the exploration limit:
  PPO has exhausted what it can change among options it samples.
- **The exploration branch's gradient is cleaner.** Half-batch gradient cosine
  over its first three updates is 0.30, 0.31 and 0.37, against −0.39 to +0.22
  for main over the same updates. Its training-game floor rose 30.0 → 31.0
  despite the exploration noise. First greedy eval at update 36.
- **Smith-rule branch:** training-game floor 32.6 vs main ~31.7, so the rule
  also helps the sampled policy. First eval at update 36.

---

## 2026-10-13 16:23 BST — Removal overlay is null; exploration confirmed; rule-augmented lineage launched

- **Removal overlay** ("no potion buys while a removal is affordable"), paired
  vs update 28: **+0.12 ± 0.25 (CI [−0.36, 0.60])**. Null on mean floor. Wins
  11 → 17, but too few to read. Potion-heavy shopping is not the leak; never
  upgrading is.
- **The exploration branch explores:**

  | At update 33 (T=1 sampling, training games) | Main | `strat8` |
  | --- | --- | --- |
  | Smith rate | 0.0% | **5.2%** |
  | Elite paths | 0.0% | 0.4% |
  | Removals | 0.0% | 1.0% |
  | Potion share of shop actions | 69% | 48% |
  | Collect floor | 31.0 | 30.0 |

  Collect floor is training-time sampled play, not the greedy eval.
- **Rule-augmented lineage launched: `ooc_branch_smith70`.** It is main plus
  the "smith at ≥70% HP" rule, applied in both training and evaluation. It
  tests whether PPO keeps compounding on top of the +1.15 rule. Compared paired
  against main and `strat8` at 36/40/44.

---

## 2026-10-13 16:18 BST — Smithing at ≥70% HP alone is worth +1.15 floors; exploration branch launched

- **Overlay "smith whenever HP ≥ 70%"**, paired on 1,024 worlds vs update 28:
  **+1.15 ± 0.23 floors (95% CI [0.70, 1.60]); mean 33.06.**
  - Act 1 boss pass 94% (from ~89%); Act 2 boss reach 68.5%; Act 2 boss pass
    30.2%; Act 3 boss reach 11.4%.
  - One collapsed habit (never upgrading) costs more than a floor, which confirms
    the exploration diagnosis.
- **Why PPO cannot find this, measured:**
  - At update 28 the OOC logit gap from each strategic option to the policy's
    choice has median **−18.7** for smith, −18.6 for elite paths, −21 for
    removal and −36 for card skips. The sampling probability is ~1e-8.
  - The combat head is *not* saturated: median top-action probability 0.82,
    median top-2 gap 1.8. Its slow RL comes from sparse reward, not lack of
    exploration.
- **Exploration branch: `ooc_branch_strat8`.** It continues from main update 32
  on the same training worlds, with a **strategic temperature of 8**: OOC logits
  on rest-site, map and shop menus are divided by 8 in both collection and the
  PPO update.
  - At T=8 a median-gap smith option is sampled ~9% of the time instead of
    never. Greedy deployment is unchanged.
  - Compared paired against main at updates 36, 40 and 44.
- **Seed-2 replicate update 4: +0.43 paired (CI [−0.00, 0.87]).** Seed 1 was
  +0.24 at update 4, so the replicate is consistent so far.
- **lr 6e-4 branch cancelled** after 2 updates to free CPU for the exploration
  branch. The lr question is deferred, not answered.
- **Running next:** the removal overlay ("no potion buys while a removal is
  affordable") vs update 28. Then a rule-augmented lineage: main + smith rule
  during training, to test whether PPO keeps compounding on top of it.

---

## 2026-10-13 16:12 BST — The OOC policy's strategy has collapsed; removing one dominated choice is worth +0.60 floors

- **Strategic choices**, greedy update 28, 119 games (TRAIN-eval worlds):

  | Choice | How often it was available | How often it was taken |
  | --- | --- | --- |
  | Upgrade at a rest site ("smith") | 557 rest sites, 98 at full HP and 180 at ≥85% | **0** |
  | Map path into an elite | 211 map choices | **0** |
  | Card removal at a shop | 571 shop menus where it was affordable | **0** (77% of purchases were potions) |
  | Skip a card reward | — | 14.5% in Act 1, 7% in Act 2, 4% in Act 3 |

  Upgrading, thinning the deck and fighting elites for relics are core to
  strong play. PPO cannot discover them because it almost never samples them
  (smith is ~1% even under T=1 sampling). This is an **exploration problem**,
  not a credit-assignment one.
- **Dominated-action filter** (new, `--dominated-filter`): drop "rest" at full
  HP, where it heals 0, so smith is the only choice. Dream Catcher is excepted.
  - Paired on 1,024 worlds vs update 28: **+0.60 ± 0.15 floors (95% CI [0.32,
    0.89])**; wins 11 → 16; mean 32.50; Act 3 boss reach 8.9%.
  - A strictly-better menu with a significant gain says upgrades matter a lot.
- **Running next (paired vs update 28):**
  - Smith whenever HP ≥ 70%.
  - Skip potion buys while a card removal is affordable.

  If these are large, the plan is an **exploration prior**: a fixed logit bonus
  for smith, elite-path and removal candidates, keyed off the encoder's action
  flags and room one-hot. It is applied the same way in collection, PPO and
  deployment, so PPO can learn to override it where it is wrong. That beats
  hard-coding rules.
- **Load:** three training trees plus evals have slowed the main run to
  ~190–250 s per update.

---

## 2026-10-13 16:05 BST — Main run +2.69 at update 28; forced-play is null; lr branch launched

- **`ooc_main_s1` update 28: 31.92, paired +2.69 (95% CI [2.12, 3.24])** vs the
  original. Updates 24 → 28 added +0.14.
- **`forced_play` is null.** Paired on 971 bank fights (greedy, update-24
  policy), the win-rate change was within noise for every group:

  | Fight | Change (pp) |
  | --- | --- |
  | Act 2 boss | −1.0 ± 1.6 |
  | Act 2 elite | 0.0 ± 1.3 |
  | Act 1 boss | +2.0 ± 1.6 |
  | Act 3 boss | +1.1 ± 1.9 |
  | Act 3 elite | +2.5 ± 3.9 |

  - Pooled: +0.4 pp. The wasted end turns are mostly harmless (Defend with no
    incoming damage, and the like). Left off by default.
  - This is consistent with the investigation (E5): Act 2 boss losses are mostly
    decided before the fight, by deck and entry HP. **OOC stays the lever.**
- **Learning-rate branch, `ooc_branch_lr6e4`:** a copy of update 28, same seed
  (so the same training worlds), at **lr 6e-4 vs 3e-4**. Paired evals at
  updates 32 and 36 against the main run will tell us whether learning can go
  faster. It runs at nice 10 alongside the seed-2 replicate.
- **Bug caught before it bit:** resuming restored the optimizer's saved lr,
  which silently overrode `--lr`. The config now wins, and the resume event logs
  both values. No earlier run was affected.
- **Machine:** 3 training trees, ~85% CPU, 19 GB of Python RSS, 67% memory free.

---

## 2026-10-13 16:00 BST — 2× throughput (parallel collectors); seed-2 replicate running; combat diagnosis

- **Collection was bound by one Python thread.** The trainer's main process
  ran at 96% CPU while 14 game workers idled at ~13% each and the machine sat
  50% idle. Profile: observation encoding ~44% of main-thread time, policy
  forward ~16%.
  - New `CollectorPool` (`src/sts_rl/parallel_collect.py`): K persistent
    collector processes, each with its own workers and a copy of the policy.
  - Greedy evaluation records are identical to single-process
    (`scripts/rl/check_parallel_collect.py`).
  - `ooc_main_s1` resumed at update 28 with 4 collectors. **Collection 159–179 s
    → 106 s** per 768 games, while a second training run shares the machine.
  - Also: atomic `latest.pt`, and `--update-threads` for the PPO phase.
  - Commits `dbc3f9e`, `e0dd889`.
- **Replicate running: `ooc_rep_s2`.** Same E8 recipe, seed 2, from the
  original checkpoint, 24 updates, at nice 10. It tests whether the +2.5 floors
  is the recipe or seed luck.
- **Combat RL diagnosis:**
  - Over 38 bank-PPO updates (~39k fights), the *sampled* training win rate on
    the boss/elite mix moved only 0.51 → 0.55. Win/loss reward over long boss
    fights is too sparse to learn from at this scale.
  - Splicing combat v1@10 onto OOC update 24 changes nothing: **+0.09 ± 0.21**
    paired.
- **Combat behaviour audit**, 60 Act 2 boss fights, frozen greedy combat head:
  - **11% of "end turn" choices (59/523, ~1 per fight) leave energy unspent with
    an affordable card in hand.** Examples: ending the turn instead of a 9-damage
    Strike, or with a Defend playable.
  - Potions are used fine: 66 uses, and only 1 loss with a usable potion left.
  - A win/loss reward barely penalises this. Dense HP/damage credit would.
  - Testing a policy-side `forced_play` filter now: hide "end turn" while energy
    remains and a safe card (no self-damage, curse or status) is playable. The
    test is paired on ~970 bank fights.

---

## 2026-10-13 15:50 BST — OOC PPO at +2.55 floors; combat boss RL stopped (hurts full games); update-phase memory fixed

- **`ooc_main_s1` update 24: 31.77 ± 0.27 mean floor, paired +2.55 (95% CI
  [2.00, 3.08]) vs the original.** It is still climbing.

  | Update | Paired gain (floors) | Wins / ~1,016 | Act 2 boss pass | Act 3 boss reach |
  | --- | --- | --- | --- | --- |
  | 4 | +0.24 | 3 | 21.1% | 4.3% |
  | 8 | +1.14 | 6 | 21.8% | 4.3% |
  | 12 | +1.57 | 10 | 26.4% | 6.8% |
  | 16 | +1.75 | 10 | 27.7% | 7.4% |
  | 24 | **+2.55** | 8 | 27.0% | 6.7% |

  Act 2 boss *reach* rose from 49% to 64%. An independent 4-worker re-evaluation
  of update 24 gave a bit-identical 31.774, so determinism holds.
- **Combat boss RL (v2, lr 3e-4) stopped at update 26: it hurts full games.**
  - Full-game paired results vs its start: **−0.51 ± 0.22** at update 10 and
    **−0.87 ± 0.23** at update 20.
  - Paired deaths per ~1,016 games, start vs update 10: Act 1 boss **98 → 137**.
    At update 20, Act 2 hallway (which includes Act 2 elites) went 318 → 352.
  - The 120-per-group held-out panel showed "+5–9 points on Act 2 elites", but
    that was within noise. It cannot resolve a 4-point Act 1 boss drop.
  - **The fight bank replays faithfully:** 300 bank fights replayed under the
    generating checkpoint reproduce the in-run outcome 300/300. So the panel is
    not biased; it is too small. The training also degraded a group it trained
    on (Act 1 boss, 10% of the mix).
  - Lessons for any combat v3:
    - include hallway fights and a stronger anchor;
    - use a panel of ≥500 per group;
    - judge only by full-game paired evaluations.
  - v1 at lr 1e-4 was +0.29 ± 0.20 at update 10, a weak positive.
- **Spliced check queued:** OOC update 24 + combat v1@10 vs OOC update 24 alone,
  1,024 paired worlds (`data/rl_runs/combos/`).
- **Memory:** after the earlier fix, the OOC trainer still spiked **2 → 12.7 GB
  during the PPO update**. A 4,096-row minibatch of padded OOC menus runs a
  single forward/backward. `update_ppo(micro_batch=1024)` now accumulates the
  same gradient in chunks (regression-tested against the full minibatch; KL
  guard preserved). The trainer was restarted from update 25 on this code
  (`ooc_main_s1_ext3`). 473 unit tests pass. Committed `dbc3f9e`.

## 2026-10-13 15:15 BST — Memory incident fixed (it crashed Chrome); runs resumed

- **What happened:** Chrome was force-quit by macOS under memory pressure at 14:25.
  A subagent traced the cause to my `ooc_main_s1` trainer, not the dashboard.
  It had grown to 19–24 GB, and swap reached 24.5/25.6 GB.
  `collect_rollouts` holds every transition of a 768-game batch with full
  observation dicts, then the evaluation held 1,024 more games on top.
- **Fix:** chunked collection and evaluation, keeping only OOC choice
  transitions, with their observations slimmed after encoding. Measured peak
  for one 512-game update plus a 512-game eval: **13.6 GB → 5.9 GB**.
- The trainer was stopped at update 20 (state saved). It **resumed from update
  20 on the fixed code** as `ooc_main_s1_ext`, continuing to update 48. Combat
  RL was also restarted from its own checkpoint (update 14) to pick up the
  chunked evaluator.
- The dashboard was not leaking (heap flat over ~10 h-equivalent of polling).
  The subagent still made it much lighter: charts update in place, a 1.1 KB
  poll when nothing changed, and no polling while the tab is hidden. Committed.
- **`ooc_main_s1` before the stop** (old evaluator): update 12 eval 30.82
  (+1.57 paired); training-time floor 30.5–31.2 at updates 13–20.
- **Combat RL:**
  - v1, lr 1e-4: slow. Update 10 full-game paired +0.29 ± 0.20 vs `main_s1@8`;
    wins 7 → 11 / ~1,016.
  - Continued as v2 at lr 3e-4. At update 14 the held-out panel shows **Act 2
    elite 68% → 77.5%**; Act 2 boss still ~24%.
- **Strategy probes** on `main_s1@8`, paired over ~510 worlds:
  - **Masking Neow's boss-relic swap: −1.99 ± 0.44 floors.** The policy's
    near-universal swap is right.
  - Act-1-only elite seeking: +0.23 ± 0.28, i.e. nothing.
  - Hand-rules are not where the headroom is; RL is.

## 2026-10-13 14:15 BST — Clean confirmation of the OOC gain; boss/elite combat RL launched

- **Exact-argmax re-score of `ooc_main_s1` update 8: 30.42 ± 0.26 floors,
  paired +1.17 ± 0.26 (95% CI [0.65, 1.70]) vs the original.** Wins 2 → 7 per
  ~1,010; Act 1 boss pass 88.5 → 90.2%; Act 2 boss reach 48.5 → 57.6%, pass
  23.2 → 22.5%. (`data/rl_runs/baselines/main_s1_i8_argmax_1024.json`.)
- **Fight bank:** 3,968 learner-reached boss/elite starts from 1,500 runs of
  that policy on fresh TRAIN worlds (`data/rl_runs/banks/fights_main_s1_i8.jsonl`).

  | Fight | Act 1 | Act 2 | Act 3 |
  | --- | ---: | ---: | ---: |
  | Elite win rate | 99.1% | **70.5%** | 59.3% |
  | Boss win rate | 90.1% | **23.4%** | 17.8% |

- **`combat_bossrl_v1` running** (dashboard: held-out fight win-rate chart):
  - PPO on the combat head only (OOC frozen) from `main_s1@8`.
  - One fight per episode from the bank's training worlds, 1,024 fights per
    update. Reward ±1 for win/loss + 0.3 × HP fraction on wins; λ=1.
  - Mix: 35% Act 2 boss, 35% Act 2 elite, 15% Act 3, 15% Act 1 (retention).
  - Metrics: greedy held-out-world fight panels every 2 updates; paired
    1,024-game evaluation every 10 updates against `main_s1@8`.
- Queue runner now has lanes (`--lane`), so a combat job runs alongside the OOC job.
- Next: when `ooc_main_s1` finishes, take its best checkpoint into a longer OOC
  run. Then alternate OOC PPO and combat RL, so strategy adapts to the stronger
  fighter.

## 2026-10-13 14:05 BST — OOC PPO replicates and grows; card teacher and combat imitation both fail

- **`ooc_main_s1` (E8 recipe, seed 1) works and keeps improving.**

  | Update | Greedy eval floor | Paired vs original | Act 2 boss reach / pass | Wins |
  | --- | ---: | ---: | ---: | ---: |
  | 4 | 29.51 | +0.24 ± 0.24 | 52.3% / 21.1% | 3 |
  | 8 | **30.44** | **+1.14 ± 0.26** | 57.8% / 21.8% | 6 |

  Training-time floor went 29.0 → 30.2 over updates 1–8; the original scores
  28.1 under the same conditions. This replicates E8 (+0.71 at update 8) on a
  new seed and exceeds it. Smithing appeared (0% → 2–5%), probably because
  the dead-action filter removed the Ruby-key option at campfires.
  These numbers come from the older, noisier evaluator; an exact-argmax
  re-score is queued for when the run ends.
- **Card-reward teacher, played directly, does not help.** Partial 148/512
  worlds: −0.54 ± 0.59 floors (stopped early to free CPU). A reliability test
  explains why: at 8 boss + 2×4 elite fights per half, option-score
  differences correlate only 0.51 across halves (argmax agreement 48% vs
  chance 27%). The overlay's 4-replay panel was therefore mostly noise
  (reliability ~0.35). The teacher skips 19% of card rewards vs the policy's 6%.
  Not dead: distilling many labels averages noise, and more replays raise
  reliability. Parked behind cheaper wins.
- **Combat imitation of native search hurts.**
  - Data: 51k DAgger labels from 600 learner games, on boss, elite and
    low-HP Act 2 fights.
  - Labels: search at 2,000 simulations, 3 shuffled hidden draw orders. Only
    35% of votes are unanimous; the policy matches the vote majority 42% of
    the time.
  - Distilled with a rehearsal KL: held-out vote agreement 0.46 → 0.52.
  - Live play: **−0.34 ± 0.22 floors**. Act 2 boss pass **23.2% → 13.4%**,
    while Act 2 boss reach rose 48.5% → 51.7%.
  - This replicates the Codex finding. Imitating unstable, partly privileged
    per-decision search labels breaks plan coherence in long fights. Dropping
    imitation for combat.
- Compute note: krill-mill (Hetzner) has 12 nearly idle cores. I have **not**
  used it, since it is a shared agent box. Say if you want me to use it; it
  would roughly double throughput.
- Next: let `ooc_main_s1` run to update 24. Build focused **boss-fight RL** for
  the combat head: PPO on many short fights from learner-reached boss starts,
  low variance and cheap (~0.07 s per fight).

## 2026-10-13 13:45 BST — Fight panel validated; card-reward teacher test running

- Baseline is now exact-argmax: **29.22 ± 0.26** (n=1,013), Act 1 boss 88.5%,
  Act 2 boss reach 48.5% / pass 23.2%, 2 wins (`orig_argmax_1024.json`). Argmax
  results are identical at 6 vs 14 workers.
- **Panel evaluator works** (`src/sts_rl/panel.py`). From any out-of-combat state
  it replays once, then plays each panel fight on a `GameContext.copy()`: the act
  boss at full HP, the run's next elites and hallway fights at current HP. It
  uses the frozen combat head with greedy play and salted combat seeds, and skips
  the game's post-battle code (which crashed on scratch states). ~0.07 s per fight.
- **Validation (X1)**: predicting whether the run actually beats the act boss,
  from a mid-act state (200 states per act, 4 samples):

  | | Act 1 (fl 6–12) | Act 2 (fl 22–30) |
  | --- | ---: | ---: |
  | HP only, AUC | 0.64 | 0.63 |
  | panel boss win (greedy, full-HP boss), AUC | 0.74 | 0.70 |
  | panel boss + elite, AUC | 0.74 | **0.76** |
  | split-half reliability (2 vs 2 samples) | 0.68–0.81 | 0.57–0.85 |

  Sampled combat or current-HP boss fights saturate in Act 2 (0.59). Greedy with
  a full-HP boss is the setting used from here on. ~2% of Act 2 fights hit the
  simulator's own undefined-behaviour flag; they are excluded.
- **X2 card-reward teacher.** The original policy plays everything, except that
  when its argmax is a card choice, the teacher scores every card option and
  skip on identical fights (boss×4, 2 elites×2, 2 hallway) and plays the best.
  Teacher-off control reproduces the baseline 23/23. The teacher overrides 58%
  of card choices. **512-world paired run in progress**
  (`data/rl_runs/_panel/teacher_cards_512.json`).
- `ooc_main_s1` update 4: 29.51 ± 0.26 (old noisier evaluator; will re-score).
  Training-time floor 28.4 → 29.6 over updates 1–6.

## 2026-10-13 13:30 BST — Fight-panel binding built and verified; evaluation determinism bug fixed

- **Native binding for the fight panel is built.**
  - `patches/sts_lightspeed_rl_panel.patch` (25 lines, applied on top of the
    unchanged 0.1.1 study patch) exposes `GameContext.enter_battle(encounter)`
    and the run's upcoming hallway and elite encounter lists.
  - Build: `scripts/rl/build_native_rl.sh` → `external/sts_lightspeed_rl/build`,
    using an isolated `.venv-native` for cmake. The frozen venv and binary are
    untouched.
  - The study patch reproduces byte-for-byte (`77fddbd…`). **The new binary
    plays identically to the frozen one on 125/125 worlds** (same final floor
    and step count), so it is a safe drop-in.
- **Bug (mine), now fixed:** the first version of `evaluate()` implemented
  "greedy" as logits×10⁴ and sampled, rather than an exact argmax. Exact ties
  (e.g. two identical Strike copies on a card-select screen) then broke
  randomly, and runs diverged chaotically: ~20% of worlds changed between
  repeats. With exact argmax, evaluation is fully deterministic at a fixed
  worker count.
  - The E8 numbers in the investigation used a true argmax and stand.
  - `ooc_main_s1` started with the old evaluator. Its paired numbers are
    unbiased but noisier; I will re-score its saved checkpoints with argmax.
- Recomputing the baseline with exact argmax
  (`data/rl_runs/baselines/orig_argmax_1024.json`); queued runs will use it.

## 2026-10-13 13:10 BST — Took over as primary agent; infrastructure up, first queue running

- New package `src/sts_rl/` (composes the frozen `sts_policy`/`sts_env`, edits none of it):
  - `evaluate.py`: deployment-style greedy eval on fixed worlds with per-world
    records, boss reach/pass/entry HP, and paired comparison.
  - `ooc_trainer.py`: the E8 recipe generalised. Every load-bearing choice is a
    flag. It logs half-batch gradient cosine, displacement vs the Adam random-walk
    line, and behaviour stats (smith rate, card skip, map choices, fights/run)
    every update.
  - `common.py`: worker/task helpers, phase-greedy forward, and a dead-action
    filter (Act 4 keys removed from the menu the policy sees).
- `scripts/rl/queue_runner.py`: detached queue and power-aware babysitter.
  `scripts/rl/dashboard.py`: live dashboard. `scripts/rl/eval_policy.py`: eval CLI.
- Baseline recomputed with the dead-action filter (above). Without the filter it
  was 29.19; the filter itself is worth ~+0.07, as in the overlay probe.
- Queue (in order):
  1. `ooc_main_s1`: E8 recipe, seed 1, 24 updates. Replicates E8 at update 8
     and shows whether gains continue.
  2. `abl_sample_combat`: sampled combat during collection.
  3. `abl_micro_credit`: GAE through the combat critic.
  4. `ooc_main_s2`: second seed.
  5. `abl_small_batch`: 128 games/update, matched total games.
- Next while these run: the fight-panel deck-strength evaluator (X1).
