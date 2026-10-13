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
