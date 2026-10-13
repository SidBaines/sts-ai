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

**Objective:** complete the game, i.e. an Ascension 0 Act 3 boss win
("victory"). Win rate is the headline; mean floor is secondary.

| Agent | Win rate (4,096-world win set) | Mean floor | Notes |
| --- | --- | --- | --- |
| Original (`joint_act3_0030.pt`) | ~0.3% (3/1,009 fresh) | 28.8 | |
| OOC PPO, update 36 (λ = 0.95) | ~1.1% (11/1,009 fresh) | 31.8 | |
| + distilled smith rule (`distill_smith85_i36/distilled.pt`) | 1.92% | 33.19 | |
| + PPO λ = 1.0, 12 updates (`ooc_pfd_lam1/iter_0012.pt`) | 2.09% | **34.08** | best mean floor |
| + distilled elite rule (`distill_r2_lam1i12/distilled.pt`) | 2.32% | 33.59 | |
| + PPO λ = 1.0, 4 updates (`ooc_r2_lam1/iter_0004.pt`) | 2.62% | 33.92 | vs distilled p = 0.018 |
| **+ PPO λ = 1.0, 12 updates** (`ooc_r2_lam1/iter_0012.pt`) | **3.76%** | **34.58** | **best**; vs update 4 p = 0.001 |

**What works:**
- The E8 OOC PPO recipe, now with **λ = 1.0**.
- Measuring strategy rules with paired evals, then distilling them into the
  network and continuing PPO.

**What we learned:**
- The OOC logits are saturated, so PPO never explores strategic options.
- λ = 0.95 critic bootstrapping biases credit against delayed payoffs and
  unlearns smithing.
- Combat PPO on fight banks does not move full-game outcomes, sparse or dense.

**Biggest wall:** the Act 2 boss (~31% pass) and the Act 3 boss (~17% pass). Public search adds +8 points at the Act 2 boss (bank fights, p ≈ 0.016).

--- | --- | --- | --- | --- |
| Original (`joint_act3_0030.pt`) | 0.30% (3/1,009) | 28.8 | — | fresh worlds |
| OOC PPO, update 36 | 1.09% (11/1,009) | 31.8 | +2.99 [2.39, 3.57] | fresh worlds |
| **Update 36 + distilled smith rule** (`distill_smith85_i36/distilled.pt`) | **1.91%** (77/4,041) | 33.2 | +4.29 [3.67, 4.87] | fresh 1,024 + win set 4,096 |
| …plus the elite rule (HP ≥ 90%) as an overlay | **2.34%** (94/4,023) | 33.1 | ≈ +4.2 | win set; wins p ≈ 0.015 pooled across 2 world sets |

**What works:**
- The E8 OOC PPO recipe: +3 floors, replicated across seeds.
- Measuring strategy rules with paired evals, then distilling them into the
  network: smith ≥85% gives +1.3 floors; elite ≥90% gives more wins.

**What we learned:**
- The OOC logits are saturated, so PPO never explores strategic options.
- With λ = 0.95 macro credit, the critic biases against delayed-payoff
  choices. Plain PPO unlearns distilled smithing (−0.54 by update 16).
  λ = 1.0 credit has the right sign.

**Running:**
- λ = 1.0 and λ = 0.99 PPO from the distilled agent.
- Discovery from scratch (exploration + λ = 1).
- Combat v3 (dense credit).

**Biggest wall:** the Act 2 boss (31% pass) and the Act 3 boss (17% pass).

--- | --- | --- | --- | --- |
| Original | 29.27 | — | 1–2 | |
| OOC PPO, update 28 | 31.92 | +2.69 [2.12, 3.24] | 11 | E8 recipe; seed 2 replicates (+1.37 at update 12 vs +1.57) |
| OOC PPO, update 36 (`ooc_main_s1/iter_0036.pt`) | 32.34 | **+3.14** [2.58, 3.70] | 11 | still improving |
| Update 28 + "smith whenever HP ≥ 85%" | 33.26 | ≈ +4.0 | 16 | rule: +1.35 [0.95, 1.79] over update 28 |
| **Update 36 + rule distilled into the network** (`distill_smith85_i36/distilled.pt`) | **33.31** | **+4.05** [3.43, 4.65] | 19 | no rule at deployment; **fresh worlds: +4.29 [3.67, 4.87], wins 3 → 22** |

**Running (since 19:50):** λ sweep from the distilled agent, discovery-from-scratch, and combat v3. See the latest entries.
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

**Running (since 19:50):** λ sweep from the distilled agent, discovery-from-scratch, and combat v3. See the latest entries.
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

## 2026-10-13 23:24 BST — Confirmed: 3.76% win rate (4,096-world win set), double the distilled agent

**Round-2 PPO, update 12** (`ooc_r2_lam1/iter_0012.pt`), win set:

| vs | Mean floor | Paired floors | Wins | McNemar p |
| --- | --- | --- | --- | --- |
| Round-2 PPO update 4 | 34.58 vs 33.92 | **+0.66 [0.39, 0.96]** | 106 → **151** (2.64% → **3.76%**) | **0.001** |
| Distilled (smith) agent | 34.58 vs 33.19 | +1.38 [1.05, 1.69] | 75 → 150 (1.88% → 3.76%) | < 0.001 |

- **The agent now completes the game (A0 Act 3 boss win) in ~1 run in 27.**
  That is ~12× the original (~0.3%).
- The 1,024-world readings (52 wins at update 12, 44 at update 16) were
  consistent with this; the win set confirms it.
- Training-game floor kept rising after update 16: 35.2–35.3 at updates
  19–20. The run continues to update 32.

---

## 2026-10-13 23:03 BST — Round-2 PPO keeps climbing: 34.7 mean floor; wins ~4–5% at 1,024 worlds (win-set check queued)

`ooc_r2_lam1` (smith + elite distilled, then PPO with λ = 1.0), standard
1,024 worlds:

| Update | Mean floor | Wins | vs distilled agent (wins 19) |
| --- | --- | --- | --- |
| 4 | 33.65 | 36 (3.6%) | p = 0.016 |
| 8 | 33.93 | 35 (3.5%) | p = 0.026 |
| 12 | **34.66** | **52 (5.2%)** | +1.35 [0.70, 1.95] floors; **p < 0.001** |
| 16 | **34.69** | 44 (4.4%) | +1.39 [0.78, 2.02]; p = 0.001 |

- From update 4 to update 12: +1.03 [0.48, 1.58] floors; wins 36 → 52
  (p = 0.05).
- Sampled smith rate is drifting down (31% → 25% by 16). Elite paths are
  steady at 6–7%. Watch the smith drift.
- **Queued:** win-set (4,096) evals of updates 12 and 16; the run is extended
  to update 32 (`ooc_r2_lam1_ext`).
- The 1,024-world win counts are noisy (±7). The 4,096-world set is the
  arbiter.
- **Per-act elite overlay** (Act 1 at ≥70%, Acts 2–3 at ≥90%) on update 4,
  win set: −0.09 ± 0.05 floors; wins 106 → 110 (p = 0.59). Null.

---

## 2026-10-13 22:51 BST — Distilling public-search votes into the combat head hurts the Act 2 boss (−6.2 points)

- **Labels:** 320 games from the best agent; every boss-fight decision labelled
  by public search (24 samples × 1,000 simulations), as soft vote targets.
  - 18,683 labels plus 3,166 rehearsal decisions.
  - The policy agreed with the search plurality on only 45.5%.
- **Distillation** (combat head only, KL anchor, 4 epochs, lr 1e-4):
  - held-out agreement with the plurality 0.445 → 0.503;
  - rehearsal argmax agreement 0.90.
- **Paired bank panel, base vs distilled combat head:**

  | Fight group | n | Base | Distilled | Change (pp) | Fights changed |
  | --- | --- | --- | --- | --- | --- |
  | **Act 2 boss** | 400 | 29.5% | **23.3%** | **−6.2 ± 2.3** | 89 |
  | Act 3 boss | 157 | 16.6% | 15.3% | −1.3 ± 3.0 | 22 |
  | Act 1 boss | 300 | 91.7% | 93.0% | +1.3 ± 1.5 | 20 |
  | Act 2 elite | 300 | 77.7% | 78.7% | +1.0 ± 1.7 | 25 |
  | Act 2 hallway | 400 | 96.3% | 96.8% | +0.5 ± 0.6 | 6 |
  | Act 3 hallway | 300 | 95.3% | 94.0% | −1.3 ± 0.9 | 8 |

- **This replicates the earlier imitation failure, with public labels this
  time**, so privileged information was not the cause.
  - The network reproduces only ~half of the search's choices.
  - Partial, per-decision imitation breaks the coherence of multi-turn plans:
    search helps when it *plays* the fight, not when its choices are copied
    one at a time.
  - Not adopted. Combat improvement from search needs either test-time search
    or a different training signal: value targets, search inside the RL loop,
    or a model that can represent the plan.
- The full-game search-augmented eval (test-time public search at the Act 2
  and Act 3 bosses) is still running.

---

## 2026-10-13 22:28 BST — New best on wins: round-2 PPO update 4, 2.62% (p = 0.018 vs distilled)

4,096-world win set:

| Agent | Mean floor | Wins | Paired floors vs distilled | Wins vs distilled (McNemar) |
| --- | --- | --- | --- | --- |
| Distilled (smith) | 33.19 | 1.87% | — | — |
| λ = 1.0 PPO, update 12 | 34.08 | 2.09% | +0.87 [0.59, 1.13] | p = 0.58 |
| Round 2 (+ elite distilled) | 33.59 | 2.26% | +0.41 | — |
| **Round 2 + λ = 1.0 PPO, update 4** (`ooc_r2_lam1/iter_0004.pt`) | **33.92** | **2.62%** (105/4,013) | **+0.71 [0.39, 1.04]** | **75 → 105, p = 0.018** |

- vs its own round-2 start: +0.30 [0.01, 0.57] floors; wins 91 → 105
  (p = 0.27).
- **The loop works:** measure strategy rules → distil them → PPO with
  unbiased credit. Wins went ~0.3% (original) → 1.9% → 2.3% → **2.6%**.
- `ooc_r2_lam1` is healthy after the battery pause: 14 faults per ~760
  games, normal. Sampled elite-path rate is rising (5.4% → 8%), smith ~33%,
  and training-game floor is 33.97 at update 7.

---

## 2026-10-13 22:11 BST — Back on AC; jobs resumed; public-search expert iteration prepared

- **Back on AC around 22:09.** The queue runner resumed `ooc_r2_lam1` (from
  update 4) and started the two held evals:
  - the full-game search-augmented eval, `searcheval_r2_a23_s64`, 2,048
    win-set worlds;
  - the round-2 win-set check, `win_r2ppo_i4`.
- **The paused Act 3 boss search job was killed, not resumed.** Its resume
  watcher died with the previous session, and its workers' wall-clock
  timeouts would have expired during the pause anyway. The full-game eval
  covers Act 3 boss fights.
- **`collect_combat_labels.py --public --boss-only`** is prepared. It will
  collect public-search vote labels in boss fights on the learner's own state
  distribution, then distil them into the combat head (soft targets plus a KL
  anchor). The earlier failed imitation used privileged labels; this is the
  fair test of search-driven combat improvement.

---

## 2026-10-13 21:25 BST — Public search +8 points at the Act 2 boss (p ≈ 0.016); laptop on battery, compute paused

- **Public-information search, 64 samples × 1,000 simulations, 238 Act 2 boss
  fights:** **37.4% vs greedy 29.4%**, +8.0 points.
  - 38 fights flipped to wins and 19 to losses; sign test p ≈ 0.016.
  - 0.68 s per decision.

  | Samples | Gain over greedy |
  | --- | --- |
  | 8 | +4.2 points |
  | 24 | +7.5 points |
  | 64 | **+8.0 points** |

  - Two of 240 fights errored: a simulator UB fault, and a draw-index card
    select (Secret Technique/Weapon). The latter now abstains and falls back to
    the policy.
- **Queued (held while on battery):**
  - **full-game search-augmented eval**: round-2 agent with public search in
    Act 2 and 3 boss fights, 2,048 win-set worlds, paired vs round 2;
  - public search on 158 Act 3 boss fights. Started, then paused with SIGSTOP
    for battery; a watcher resumes it on AC.
- **The laptop went on battery (~21:20).** The queue runner paused
  `ooc_r2_lam1` at update 4 and holds new jobs.
- **Round-2 PPO update 4** (smith + elite distilled, then λ = 1.0 PPO),
  1,024 standard worlds vs the distilled agent:
  - **wins 19 → 36 (1.88% → 3.56%)**, discordant 14/31, McNemar p = 0.016;
  - mean floor +0.34 ± 0.31.

  This is the strongest win signal so far. It needs win-set confirmation,
  which is queued once on AC.
- **Discovery from scratch, update 16:** +0.33 ± 0.29 vs its main@36 start.
  Smith rate stayed ~3–4%, so **no discovery** in 16 updates.

---

## 2026-10-13 21:21 BST — Avoiding conflicting boss relics hurts; λ = 0.99 ≈ λ = 1.0 at update 16

- **Avoid Fusion Hammer and Coffee Dripper** (both conflict with smithing or
  resting), on top of round 2, win set: **−0.43 ± 0.05 floors (CI [−0.53,
  −0.34])**; wins 93 → 85.
  - Both are energy relics. The extra energy outweighs losing smith or rest,
    so the policy's choices were right.
  - Negative result; the overlay stays off.
- **From distilled, update 16** (1,024 worlds):
  - λ = 0.99: 34.05, +0.72 [0.16, 1.24];
  - λ = 1.0: 34.11, +0.80 [0.25, 1.36].

  At this horizon both avoid the λ = 0.95 collapse, which was −0.54 at
  update 16.
- **Discovery from scratch** finished 16 updates: smith ~3–4% throughout.
  **Exploration + unbiased credit did not discover smithing in 16 updates.**
  The update-16 eval is pending.
- **New evaluator: `scripts/rl/eval_search_boss.py`.** It plays full games
  greedily, but uses public search in boss fights. It is smoke-tested; full
  runs wait on the 64-sample public-search estimate.

---

## 2026-10-13 21:10 BST — Public-information search: most of the privileged gain was clairvoyance, but a real gain remains

- **New native bindings** (RL patch only; the study patch is unchanged):
  `BattleContext.copy()` and `reseed_rngs(seed)`.
  - `PanelWorker.search_votes_public` searches deep copies with the draw order
    shuffled **and** every battle RNG stream re-seeded (enemy AI, card
    randomness, reshuffles, potions, monster HP).
  - The build lives in `external/sts_lightspeed_rl/build_pub`, beside the build
    the running jobs use. All 473 tests pass, and replays reproduce 300/300
    bank fight outcomes.
- **Same 120 Act 2 boss fights**, greedy combat head 27.5%:

  | Combat controller | Win rate | Flipped to win / loss | Seconds per decision |
  | --- | --- | --- | --- |
  | Privileged search, 8 × 1,000 (true RNG streams) | 44.2% | 27 / 7 | 0.13 |
  | Public search, 8 × 1,000 | 31.7% | 14 / 9 | 0.11 |
  | Public search, 24 × 1,000 | **35.0%** | 18 / 9 (p ≈ 0.12) | 0.26 |

  - **Roughly two-thirds of the "search headroom" was knowledge of hidden
    randomness.**
  - The public gain grows with samples (+4.2 → +7.5 points).
  - Running: 64 × 1,000 on 240 fights, for a decisive estimate.
- **If public search holds at ~+8–10 points, it is affordable at test time.**
  ~0.3–0.7 s per decision, and 3 boss fights per run. It is also a cleaner
  expert-iteration teacher than the privileged search used before.
- Other runs at update 12:
  - λ = 0.99: 33.61 (+4.39 vs original).
  - Discovery (exploration + λ = 1): 32.6 (+3.39); smith ~3–4%, no
    discovery.

---

## 2026-10-13 21:00 BST — Combat headroom at the Act 2 boss: search wins 44% vs greedy 27.5% (privileged upper bound)

- **Act 2 boss fights from the distilled-agent bank.** Each decision is the
  plurality vote of native search over 8 shuffled draw orders × 1,000
  simulations:

  | Fights | Greedy combat head | Search | Flipped to win | Flipped to loss |
  | --- | --- | --- | --- | --- |
  | 120 | 27.5% | **44.2%** | 27 | 7 |

  - Sign test p ≈ 0.001.
  - **0.13 s per decision**: cheap.
  - 40 fights at 4 × 200 simulations gave 30% vs 25%. More search helps.
- **Caveat: this is privileged.** The native wrapper re-seeds only the
  searcher's own RNG ("cannot touch the battle RNGs"). Its simulations use the
  true enemy-AI and card-randomness streams; only the draw pile is shuffled.
  So 44% is an upper bound, not a deployable agent.
- **Still:** about 30% of lost Act 2 boss fights are winnable by better play
  from the same starting state. Combat is a real lever for wins.
- **Next:**
  1. Add public-information search: reseed all battle RNG streams per sample
     in the clone (new binding in `patches/sts_lightspeed_rl_panel.patch`; the
     study patch stays unchanged).
  2. Measure its boss win rate.
  3. If it holds up, use it at test time in boss fights and/or as an
     expert-iteration teacher for the combat head.
- Discovery-from-scratch at update 12: smith rate ~3–4%, no discovery.

---

## 2026-10-13 20:56 BST — Win set: λ = 1.0 gives +0.87 floors (tight); elite-seeking keeps trading floors for wins

4,096-world win set (TRAIN seeds from 6e9), paired vs the distilled agent:

| Agent | Mean floor | Paired floors | Wins (of ~4,010) | McNemar p |
| --- | --- | --- | --- | --- |
| Distilled (smith ≥85%) | 33.19 | — | 77 (1.92%) | — |
| λ = 1.0 PPO, update 12 | **34.08** | **+0.87 [0.59, 1.13]** | 84 (2.09%) | 0.58 |
| Round 2 = λ1@12 + distilled smith + elite | 33.59 | +0.41 [0.11, 0.73] | **93 (2.32%)** | 0.18 |
| Distilled + elite ≥90% overlay | 33.12 | −0.06 | 94 (2.34%) | 0.10 |

- **λ = 1.0 is a robust mean-floor gain**, confirmed at 4,096 worlds. It
  translates into only a small win gain so far.
- **Elite-seeking consistently trades ~0.4–0.5 floors for +10–17 wins per
  4,000.** The direction holds in every comparison, but no single one is
  significant (round 2 vs λ1@12: −0.47 floors, 82 → 93 wins, p = 0.39).
- λ = 1.0 at update 16, standard set: +0.80 [0.25, 1.36] vs distilled. It is
  holding.
- **Launched:**
  - `ooc_r2_lam1`: PPO λ = 1.0 from the round-2 agent.
  - Win-set test of avoiding Fusion Hammer and Coffee Dripper on top of
    round 2.

---

## 2026-10-13 20:51 BST — λ sweep at update 12: credit bias scales with 1 − λ; round-2 distillation (smith + elite) done

| From distilled, update 12 | Paired floors vs start | Wins /~1,008 (start 19) | Smith rate |
| --- | --- | --- | --- |
| λ = 0.95 | −0.15 (→ −0.54 at 16) | 20 | 23% (→ 14.5%) |
| λ = 0.99 | +0.32 [−0.25, +0.91] | 27 | 34–38% |
| λ = 1.0 | **+0.88 [0.35, 1.45]** | 16 | 45% |

- Smith retention orders exactly with λ. Floors favour λ = 1.0.
- Wins at 1,024 worlds disagree (λ = 0.99 27 vs λ = 1.0 16, p = 0.07), which
  is exactly why win claims need the 4,096-world set. A win-set eval of
  λ = 1.0 update 12 is running.
- **Round-2 distillation** of smith ≥85% + elite ≥90% into λ = 1.0
  update 12 (`data/rl_runs/distill_r2_lam1i12/distilled.pt`):
  - 3,750 rest and 1,448 elite labels;
  - held-out rule agreement 0.79 → 0.94; rehearsal agreement 0.97;
  - win-set eval running.
- **Added an `--avoid-boss-relics` overlay.** Fusion Hammer disables smithing
  and was taken in ~10% of runs; Coffee Dripper disables resting. Test queued
  after the current evals.
- The machine is saturated (0% idle): three trainers plus two 4,096-world
  evals.

---

## 2026-10-13 20:27 BST — λ = 1.0 fixes the unlearning and improves: 34.17 mean floor (+0.88 over the distilled start)

| From the distilled agent, same training worlds | Update 4 | Update 8 | Update 12 | Smith rate at update 12 |
| --- | --- | --- | --- | --- |
| λ = 0.95 (E8 recipe) | +0.08 | +0.40 | −0.15 (→ −0.54 at 16) | 23% (→ 14.5% at 16) |
| **λ = 1.0** | +0.23 | +0.46 | **+0.88 [0.35, 1.45]** | **45%** (stable) |

- **34.17 mean floor at λ = 1.0, update 12: a new best.** Wins 19 → 16 at
  1,024 worlds, which is noise; a win-set check will follow.
- This confirms the critic-bias diagnosis end to end. Credit bootstrapped with
  λ = 0.95 unlearns a delayed-payoff behaviour worth ~1.2 floors. Monte-Carlo
  credit keeps it and keeps improving.
- **Recipe change: OOC PPO should use λ = 1.0**, or close to it; λ = 0.99 is
  running.
- **Discovery from scratch** (exploration + λ = 1, from update 36): smith rate
  3–5% through update 8. No discovery yet.
- **Combined labels collected** for the next distillation, from 789 games:
  3,790 rest-site labels (1,419 smith) and 1,458 elite-option map decisions
  (583 labelled elite).

---

## 2026-10-13 20:17 BST — Combat v3 (dense credit) is also null; λ = 1.0 at update 8 matches λ = 0.95

- **Combat v3 stopped at update 20.**
  - Full-game paired vs its start: **−0.20 ± 0.22** at update 10, **+0.08 ±
    0.24** at update 20; wins 19 → 15.
  - Panel: Act 2 boss 33.3 → 33.8%; Act 3 boss 17.5 → 25% (n = 40, noise).
  - Sampled training win rate flat at ~0.64.
  - Dense potential shaping with GAE did not make fight-level PPO learn
    usefully at ~20k fights.
  - **Combat needs a different method.** Candidates: much larger scale;
    imitation of a *public-information* search (sampled hidden draw orders and
    enemy RNG) rather than privileged search; or test-time search.
- **λ = 1.0 vs λ = 0.95 from the distilled agent, update 8:**
  - λ = 1.0: 33.73, +0.46 vs start, wins 19 → 26;
  - λ = 0.95: 33.69, +0.40;
  - difference +0.05 ± 0.27.

  The λ = 0.95 run only declined after update 9, so updates 12 and 16 decide.

---

## 2026-10-13 19:47 BST — λ sweep and discovery-from-scratch running

Running, all paired on the same training worlds where possible:

| Run | Start | Credit | Exploration | Question |
| --- | --- | --- | --- | --- |
| `ooc_ppo_from_distilled` (done) | distilled | λ = 0.95 | none | unlearned smithing by update 16 (−0.54) |
| `ooc_pfd_lam1` | distilled | **λ = 1.0** | none | does unbiased credit keep and refine smithing? |
| `ooc_pfd_lam099` | distilled | λ = 0.99 | none | the middle ground |
| `ooc_discover_strat8_lam1` | main update 36 (never smiths) | λ = 1.0 | strategic T = 8 | **can PPO discover HP-conditional smithing or elites by itself?** |
| `combat_v3_shaped` | distilled | dense shaping, λ = 0.95 | — | combat learning speed |

- `ooc_pfd_lam1`, updates 1–4: smith rate 38 → 40 → 43 → 43%. This is the
  same as the λ = 0.95 run early on; its decline only began after update 9.
- Memory is 46% free with four trainers running. A low-memory alert is armed.

---

## 2026-10-13 19:40 BST — Confirmed: λ = 0.95 critic bootstrapping biases credit against smithing; λ = 1.0 does not

Smith-vs-rest macro advantages on the **same 512 sampled games** from the
distilled policy, under two GAE λ (all advantages are shifted positive because
the critic lags the improved policy; only the gaps matter):

| HP at rest site | λ = 0.95: smith − rest | λ = 1.0: smith − rest |
| --- | --- | --- |
| 100% | +0.032 (rest n = 42) | +0.189 |
| ≥85% | **−0.026** (~2σ) | +0.012 |
| 70–85% | **−0.049** (~2σ) | **+0.083** (~1.3σ) |

- **Under λ = 0.95, smithing at high HP looks worse than resting.** That is
  why PPO unlearns it.
- **Under Monte-Carlo returns (λ = 1.0) it looks better**, matching the paired
  evaluations (+1.2 floors). The cost is ~3× noisier advantages: SE 0.02–0.04
  vs 0.007–0.013.
- **Implication for the E8 recipe:** λ = 0.95 macro credit is biased against
  delayed-payoff OOC decisions (upgrades, and plausibly card picks and relic
  choices), because the critic cannot see their value. The original policy's
  "never smith" was likely learned through the same bias.
- `ooc_pfd_lam1` (PPO from distilled at λ = 1.0) is running. A middle ground
  (λ = 0.99) is the next candidate if λ = 1.0 is too noisy.
- **Combat v3, update 10:** full game −0.20 ± 0.22 vs its start (wins
  19 → 18); panel Act 2 boss 33.3 → 36.3% (~1 SE). Continuing to update 20.

---

## 2026-10-13 19:37 BST — Plain PPO *unlearns* smithing: evidence of biased OOC credit (λ = 0.95)

- **`ooc_ppo_from_distilled`** (E8 recipe, λ = 0.95, no rule), from the
  distilled checkpoint:
  - sampled smith rate 37–48% over updates 1–9, then **24% at 11 and 14.5% at
    16**;
  - greedy at update 16: smith at 15/52 full-HP rest sites and 8/84 at ≥85% HP.
    The distilled start smithed ~all of these;
  - paired vs the start: +0.08, +0.40, −0.16, then **−0.54 [−1.07, −0.01]** at
    update 16; wins 19 → 14.
- With ~40% smith sampling there are plenty of samples, so this is **bias, not
  just noise**. PPO's credit pushes against a choice that paired evaluation
  says is worth about +1.2 floors.
- **Hypothesis: critic bias under λ = 0.95.**
  - An OOC advantage is dominated by the critic's bootstraps over the next ~20
    decisions.
  - The critic sees HP directly but barely sees one upgrade (one "+" in the
    deck bag), so resting (+24 HP) always looks better.
  - The upgrade's payoff at the boss lies beyond the effective credit horizon.
  - The original policy's "never smith" habit may itself be a product of this
    bias.
- **Tests running:**
  1. `ooc_pfd_lam1`: the same run (same seed, so the same training worlds) with
     **λ = 1.0**, pure Monte-Carlo credit with the critic only as a baseline.
  2. Smith-vs-rest advantages by HP under λ = 0.95 vs 1.0 on the same 512 games
     from the distilled policy.

---

## 2026-10-13 19:29 BST — Elite threshold: 90% beats 80%; PPO from distilled is flat; combat v3 slow

- **Elite threshold, win set (4,096 worlds), vs the distilled agent:**

  | Take an elite path at HP ≥ | Floors | Wins | McNemar p |
  | --- | --- | --- | --- |
  | 80% | −0.26 [−0.51, −0.00] | 77 → 91 | 0.22 |
  | 90% | −0.06 [−0.29, +0.16] | 77 → **94** | 0.10 |

  Adopting **90%**.
- **`ooc_ppo_from_distilled`:** paired evals at updates 4, 8 and 12 read 33.36,
  33.69 and 33.15, against a start of 33.31. PPO **keeps** the distilled
  behaviour but does not significantly build on it within 12 updates.
- **Combat v3 (dense credit), update 7:**
  - panel Act 2 boss 33.3 → 34.3–34.8%, Act 3 boss 17.5 → 17.5–20%;
  - sampled win rate flat at ~0.64.

  Better than v2's trajectory, but still slow. First full-game eval at
  update 10.
- **Next:** distil smith ≥85% + elite ≥90% together. Then run PPO from that,
  and confirm on the win set.

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
  - Causal test, no potion drinking in hallway fights while HP ≥ 30%: **null**
    (+0.00 ± 0.24; wins 19 → 19). The correlation was confounding: stronger
    runs simply have more potions.
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
