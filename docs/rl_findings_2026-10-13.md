# Non-LLM RL for Slay the Spire: what works, what doesn't, and why

**Date:** 2026-10-13 (one long working session).
**Branch:** `claude/slay-spire-rl-debug-da6d22`.
**Audience:** the next agent or person picking this up.

This is the distilled record. Related documents:

- **Chronological detail:** every experiment, number and correction, in
  [rl_progress_log.md](rl_progress_log.md), most recent first.
- **The original diagnosis** of why RL was not working (E1–E10) and the first
  proposal: [rl_investigation_2026-10-13.md](rl_investigation_2026-10-13.md).
- **Operational gotchas** (memory, queue, builds, search helpers):
  [scripts/rl/CLAUDE.md](../scripts/rl/CLAUDE.md).

Numbers below are greedy (exact argmax) and **paired**: each comparison uses the
same world seeds. Win-rate claims use the 4,096-world win set.

---

## 1. Where we are

**Objective:** complete the game, defined as an Ascension 0 Act 3 boss win
(outcome `victory`). Act 4 and higher ascensions are out of scope for now.
Win rate is the headline metric; mean floor is secondary.

| Agent | Win rate | Mean floor | Notes |
| --- | --- | --- | --- |
| Original (`joint_act3_0030.pt`) | ~0.2–0.3% (2/1,013 standard, 3/1,009 fresh) | 28.8–29.2 | not run on the win set |
| OOC PPO, E8 recipe, update 36 (`ooc_main_s1/iter_0036.pt`) | 1.09% (11/1,009 fresh) | 31.8 | |
| + "smith at HP ≥ 85%" distilled (`distill_smith85_i36/distilled.pt`) | 1.91% (77/4,041) | 33.19 | |
| + PPO λ = 1.0, 12 updates (`ooc_pfd_lam1/iter_0012.pt`) | 2.09% (84) | 34.08 | best mean floor at the time |
| + "elite at HP ≥ 90%" also distilled (`distill_r2_lam1i12/distilled.pt`) | 2.32% (93) | 33.59 | |
| + PPO λ = 1.0, 4 updates (`ooc_r2_lam1/iter_0004.pt`) | 2.62% (105) | 33.92 | |
| **+ PPO λ = 1.0, 12 updates** (`ooc_r2_lam1/iter_0012.pt`) | **3.76% (151/4,016)** | **34.58** | vs update 4: wins p = 0.001 |
| + PPO λ = 1.0, 16 updates (`ooc_r2_lam1/iter_0016.pt`) | 3.79% (152) | 34.50 | level with update 12 |

The best agent completes the game in about 1 run in 27. That is roughly 12×
the original, and about 2× the agent that only had the upgrade rule distilled.

**What is learned and what is not:**

- **Only the out-of-combat (OOC) head has ever been improved.** It handles the
  map, card rewards, rest sites, shops, events and relics.
- **The combat head is the original, frozen and unchanged** in every agent in
  the table. Every attempt to train it failed to improve full games (§4.2).
- The upgrade and elite behaviours were **found by measured hand rules and then
  distilled into the network**. PPO did not discover them; it keeps and refines
  them. No rule runs at play time.

**Where runs end** (distilled agent, 1,024 fresh worlds):

| Where | Runs ending there |
| --- | --- |
| Act 1 boss | 77 |
| Act 2 hallway | 241 |
| **Act 2 boss** | **449** |
| Act 3 hallway | 123 |
| Act 3 boss | 94 |
| Win | 22 |

- Act 2 boss pass rate is ~31–35%. Act 3 boss pass rate, given reach, is ~17%.
- **The two boss gates are the wall.**

**Still running when this was written:**

- `ooc_r2_lam1` continuing to update 32. Training-game floor was ~34.7–35.3
  at updates 19–24.

**Play-time search, full-game result** (§7.2): adding public search at the Act 2
and 3 bosses to the round-2 *starting* agent, paired on 2,019 win-set worlds:

| | Act 2 boss pass | Wins |
| --- | --- | --- |
| No search | 38.0% | 53 (2.63%) |
| Public search | **46.1%** | **69 (3.42%)** |

- McNemar p = 0.085; mean floor +0.78 [0.54, 1.02].
- It has not yet been tried on top of the best network.

---

## 2. The system in one page

- **Environment:** `sts_lightspeed` simulator via `sts_env`; full runs with
  `max_act=3`. Reward mode `progress`: 0.05 per floor, plus a terminal ±1
  for a win or loss.
  - A run returns ≈ 0.05·floor − 1 on death and ≈ +3.55 on a win.
  - **So a win is worth about 41 floors**, and +1 point of win rate ≈ +0.4
    mean floor. The reward is already very win-heavy; wins are just rare.
- **Policy:** a composite of two candidate-scoring actor-critic heads (combat
  and OOC, ~2.3M parameters each).
  - Each scores every legal action given the state, using the
    `public_candidate_v2` encoder: 256 state dims, and 128 per action
    (32 keyword flags, 32 numeric features including a room one-hot for map
    nodes, and 64 hashed bag-of-words).
  - Legal action indices refer to the current menu.
- **Training code:** `src/sts_rl/`.

  | Module | Role |
  | --- | --- |
  | `ooc_trainer.py` | OOC PPO |
  | `combat_trainer.py` | fight-bank PPO |
  | `distill.py` | label → head distillation with a KL anchor |
  | `panel.py` | scratch fights; search votes, privileged and public |
  | `parallel_collect.py` | `CollectorPool` |
  | `evaluate.py` | argmax evals, `paired`, `paired_wins` |
  | `common.py` | workers, menu filters and overlays, `phase_forward` |

  Command-line wrappers live in `scripts/rl/`.
- **World-seed namespaces** (all TRAIN split; DEV and final cohorts were never
  used):

  | Seeds from | Use |
  | --- | --- |
  | 5e9 | **standard eval set**, 1,024 worlds; reused for selection |
  | 5.6e9 | **fresh confirmation set**, 1,024 worlds |
  | 6e9 | **win set**, 4,096 worlds |
  | 7e9+ | training (offset by seed and iteration) |
  | 8e9 / 8.5–8.6e9 | OOC teacher and rule labels |
  | 9e9 / 9.5e9 | combat labels |
  | 1.1e10 / 1.12e10 | fight banks |

- **Infrastructure:**
  - `scripts/rl/queue_runner.py` is a detached job queue with lanes. It
    pauses jobs on battery, keeps the machine awake on AC, and has a stall
    watchdog.
  - `scripts/rl/dashboard.py` serves on port 8791.
  - Native builds: `external/sts_lightspeed_rl/build` (play) and
    `build_pub` (adds `BattleContext.copy/reseed_rngs`).

---

## 3. What works, and why

### 3.1 The E8 OOC PPO recipe

The settings:

- combat head frozen, and played **greedy** during collection;
- **OOC-only macro credit**: the rewards between consecutive OOC decisions are
  summed, and the bootstrap uses the OOC critic at the next OOC decision;
- 768 games per update; lr 3e-4; 3 epochs; minibatch 4,096; clip 0.2;
- no entropy bonus; KL guard 0.05;
- dead-action filter: drop the Act 4 key actions ("recall", "take key"),
  which cannot help when `max_act=3`.

**Why it works:** it removes the self-inflicted noise that sank the earlier
campaigns (investigation E1–E4):

- Sampling combat during collection cost ~2 floors and made OOC returns mostly
  combat luck.
- Crossing the combat critic in GAE biased OOC credit toward avoiding fights.
- Small batches gave gradients that were ~90% noise, which Adam turned into a
  random walk.

**Evidence** (seed 1, paired vs the original):

| Update | 8 | 16 | 24 | 28 | 36 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| Paired floors | +1.14 | +1.75 | +2.55 | +2.69 | +3.14 | +3.10 |

A seed-2 replicate tracked it: +0.43, +1.46 and +1.37 at updates 4, 8 and 12
(seed 1: +0.24, +1.14, +1.57). Progress flattened around +3.1 by updates
36–40.

**Use λ = 1.0, not 0.95, for the OOC credit** (§3.5). The E8 recipe as first
run used 0.95.

### 3.2 Paired, exact-argmax evaluation, sized for the question

- **Exact argmax**, not logit scaling. Scaled "greedy" breaks exact ties (two
  copies of the same card) at random, and the run then diverges chaotically.
  Argmax evals are bit-reproducible across worker counts and builds.
- **Pairing on world seeds.** Two policies' per-world floors correlate 0.66,
  and 57% of worlds end identically between nearby checkpoints. Paired SE is
  ~0.25 floors at 1,024 worlds, against ~0.27 SE for each mean alone, so
  differences of ~0.5 floors are resolvable.
- **Wins need more worlds.** At a ~2% win rate, 1,024 worlds resolve only
  multi-point changes (counts swing ±7). Use the 4,096-world win set and the
  exact McNemar test on discordant worlds (`paired_wins`, printed by
  `compare_evals.py`).
- **Selection bias.** The standard set was used to choose among many variants.
  The winners were confirmed on the untouched fresh set (distilled agent
  +4.29 [3.67, 4.87] vs original), and wins on the win set.

### 3.3 Measure strategy with menu overlays

An overlay restricts a menu at play time, e.g. "no rest at HP ≥ X". Paired on
1,024 worlds, this measures exactly what a strategic habit is worth. It is far
more sensitive than any per-decision learning signal, because the whole-game
comparison is paired and averages over thousands of decisions.

Results (`eval_policy.py --smith-above/--elite-above/...`):

| Overlay (vs the agent at the time) | Paired floors | Wins | Verdict |
| --- | --- | --- | --- |
| Drop "rest" at full HP (strictly dominated) | **+0.60** [0.32, 0.89] | 11 → 16 | real |
| **Smith whenever HP ≥ 70%** | **+1.15** [0.70, 1.60] | | real |
| **Smith whenever HP ≥ 85%** | **+1.35** [0.95, 1.79] | | real; ≈ 70% (+0.18 ± 0.17) |
| Smith whenever HP ≥ 50% | **−0.57** [−1.05, −0.10] | | harmful: low-HP smiths are costly |
| **Take an offered elite at HP ≥ 90%** | −0.06 ± 0.12 (win set) | **77 → 94** (p = 0.10; pooled with a 2nd set p ≈ 0.015) | wins up at no floor cost |
| Elite at HP ≥ 80% | −0.26 [−0.51, −0.00] | 77 → 91 | worse than 90% |
| Elite per act (Act 1 ≥70%, Acts 2–3 ≥90%) | −0.09 ± 0.05 | 106 → 110 | null |
| No potion buys while a removal is affordable | +0.12 ± 0.25 | | null |
| Skip all card rewards after floor 33 | −0.09 ± 0.04 | 13 → 9 | slightly harmful |
| Skip all card rewards after floor 17 | **−1.63** [−2.04, −1.22] | | harmful: Act 2 picks matter |
| No hallway potion use at HP ≥ 30% | +0.00 ± 0.24 | 19 → 19 | null |
| Never take Fusion Hammer or Coffee Dripper | **−0.43** [−0.53, −0.34] | 93 → 85 | harmful: the energy is worth more |
| Combat: hide "end turn" while energy and a safe card remain | +0.4 pp pooled (971 bank fights) | | null |

**Why the policy needed this.** At update 28, across 119 greedy games, the
policy:

- **never smithed** (0 of 557 rest sites, 98 of them at full HP);
- **never took an elite** (0 of 211 map choices that offered one);
- **never removed a card** (0 of 571 affordable shop menus).

The OOC logits were **saturated**: the median log-odds gap from these options
to the chosen one was −18 to −36, a sampling probability of ~1e-8. PPO can only
learn about actions it samples, so it could never discover them (§5.1).

### 3.4 Distil winning rules into the network

Pipeline:

1. `scripts/rl/collect_rule_labels.py` plays greedy games with the rule active,
   so the rule-policy's states are visited. It labels every rest-site (and
   optionally elite-option) menu with the rule's choice, and samples other OOC
   decisions for rehearsal.
2. `scripts/rl/distill_ooc.py` trains with CE on the labels plus β·KL to the
   original on the rehearsal decisions; the combat head is asserted unchanged.

Results:

- **Smith ≥ 85%:** 2,877 labels; held-out rule agreement 0.50 → 0.82–0.90;
  rehearsal agreement 0.96.
  - The distilled network, with no rule at play time, equals the rule:
    −0.08 ± 0.23 vs update 36 + overlay.
  - It is **+0.96 [0.46, 1.45]** over update 36 on the standard set, and
    **+1.34 [0.85, 1.83]** on fresh worlds.
- **Smith + elite** (round 2): 5,198 labels; agreement 0.79 → 0.94;
  rehearsal 0.97.

### 3.5 λ = 1.0 (Monte-Carlo) OOC credit: the most important algorithmic finding

**Symptom.** Plain PPO (E8, λ = 0.95), started from the distilled smithing
policy, **unlearned** the behaviour:

- sampled smith rate 37–48% over updates 1–9, then 24% at update 11 and
  14.5% at 16;
- paired floors +0.08, +0.40, −0.15, then **−0.54** at update 16.

**Mechanism: critic bias against delayed payoffs.**

- With λ = 0.95, an OOC advantage is dominated by the critic's bootstraps over
  the next ~20 decisions.
- The critic sees HP directly, but a single upgrade is one "+" in a hashed deck
  bag that it effectively cannot see.
- So "rest" (+24 HP) always scores above "smith" (+1 upgrade), even though the
  upgrade pays off at the boss, beyond the effective credit horizon.

**Direct test.** Smith-vs-rest advantages on the *same* 512 sampled games under
two λ:

| HP at rest site | λ = 0.95: smith − rest | λ = 1.0: smith − rest |
| --- | --- | --- |
| 100% | +0.032 | +0.189 |
| ≥85% | **−0.026** (~2σ) | +0.012 |
| 70–85% | **−0.049** (~2σ) | **+0.083** |

The sign flips. Monte-Carlo credit agrees with the paired evaluations. Its cost
is ~3× noisier advantages.

**Training confirms it.** From the distilled agent, on the same training worlds:

| λ | Update 4 | Update 8 | Update 12 | Update 16 | Smith rate at update 16 |
| --- | --- | --- | --- | --- | --- |
| 0.95 | +0.08 | +0.40 | −0.15 | −0.54 | 14.5% |
| 0.99 | +0.05 | −0.07 | +0.32 | +0.72 | ~35% (slow drift) |
| **1.0** | +0.23 | +0.46 | **+0.88 [0.35, 1.45]** | +0.80 | ~45% (stable) |

On the win set, λ = 1.0 at update 12 is **+0.87 [0.59, 1.13]** floors vs the
distilled start.

**Implication.** The original policy's "never smith / never elite" habits were
plausibly *learned through* this bias (and the GAE-through-combat bias,
investigation E3). **Default OOC runs should use `--gae-lambda 1.0`.**

### 3.6 The loop: measure → distil → PPO (λ = 1)

The working recipe, which took wins from ~0.3% to 3.8%:

1. Find a collapsed or suspicious habit: logit gaps, choice statistics
   (`ooc_choice_stats`-style scripts) and death locations.
2. Measure candidate rules with paired overlays. Keep only significant wins,
   and confirm them on fresh worlds or the win set.
3. Distil the winning rules into the OOC head.
4. Continue PPO with λ = 1.0, which keeps and refines the behaviour. Round 2:
   update 4 2.62% → update 12 **3.76%** (p = 0.001).
5. Repeat.

Watch for drift: in round 2 the sampled smith rate fell 31% → 25% over updates
4–16, even at λ = 1.0. Re-distil if it keeps falling.

### 3.7 Engineering that mattered

- **Throughput.** Single-process collection was bound by the trainer's main
  thread: Python observation encoding ~44% and the forward ~16%, while game
  workers were 87% idle. `CollectorPool` (K processes, each with workers and a
  policy copy) cut collection from ~160–180 s to ~106 s per 768 games at
  K = 4, with identical eval records.
- **Memory.** `collect_rollouts` keeps every transition. The trainer reached
  20+ GB and macOS force-quit Chrome. Fixes:
  - chunked collection and eval;
  - keep only OOC transitions;
  - slim observations: the encoder reads only a few text lines, and encodings
    stay identical;
  - micro-batched PPO (`update_ppo(micro_batch=1024)`), which gives the same
    update with bounded activations.

  Result: ~5.7 GB peak per trainer.
- **Branching runs.** Copy `latest.pt` into a new `--out` and `--resume` with
  the same seed: same training worlds, so paired comparisons. `--lr` now
  overrides the restored optimizer lr; it used to be silently ignored.

---

## 4. What didn't work, and why

### 4.1 OOC

- **Temperature exploration on strategic menus** (`--strategic-temperature 8`,
  OOC logits ÷8 on rest, map and shop menus in collection and PPO; argmax is
  unchanged).
  - It *did* explore: smith 0 → 5%, elite 0 → 0.4%, removal 0 → 1%.
  - Under λ = 0.95, PPO pushed them back down (smith 5.2% → 3.1% in 4
    updates), and greedy play gained nothing (−0.37 ± 0.26 vs main at the
    same update).
  - **Why:**
    - undirected exploration samples smith at *every* HP, and low-HP smiths
      are very costly (advantage −0.41), so they dominate the gradient;
    - λ = 0.95 bias pushes against smith at high HP too;
    - the true high-HP benefit (~0.02 reward units per decision) sits under
      advantage noise of ~0.15, needing ~200 samples per HP bucket.
- **Discovery from scratch** (exploration T = 8 + λ = 1.0, from update 36,
  16 updates): smith stayed at 3–5%; +0.33 ± 0.29 floors, n.s. **No discovery
  at this scale.** Unbiased credit is necessary but not sufficient. Exploration
  must be directed and the sample budget much larger.
- **Card-reward "teacher"** (score each card option by short fight replays of
  the resulting deck): option-score split-half correlation ~0.5 and argmax
  agreement 48% (chance 27%). Played directly: −0.54 ± 0.59 (partial run). The
  labels were too noisy at affordable replay counts.
- **A higher learning rate** (6e-4 branch) was cancelled after 2 updates for
  compute. **Unanswered.**
- **Card skipping, removals, potion conservation, relic avoidance:** see the
  overlay table. The policy's existing choices there were as good as or
  better than the obvious hand rules. The potion one is instructive:
  - Act 2 boss win rate rises from 23% with 0–1 potions to 39–45% with 2–3,
    but forcing potion conservation changed nothing.
  - The correlation was confounding: stronger runs simply have more potions.

### 4.2 Combat (every attempt failed to improve full games)

| Attempt | Result |
| --- | --- |
| Fight-bank PPO, sparse win/loss, lr 1e-4 (v1) | +0.29 ± 0.20 full game at update 10 (n.s.) |
| Same, lr 3e-4 (v2) | **−0.51** (update 10), **−0.87** (update 20); Act 1 boss deaths 98 → 137 |
| v1 combat head spliced onto a later OOC head | +0.09 ± 0.21 |
| Dense potential shaping (v3): φ = 0.3·HP − 0.5·enemy HP fraction, GAE 0.95, hallway fights included | −0.20 (update 10), +0.08 (update 20); Act 2 boss panel 33.3 → 33.8% |
| Imitate **privileged** native search (51k DAgger labels) | −0.34 ± 0.22; **Act 2 boss pass 23.2% → 13.4%** |
| Imitate **public** search (18.7k boss labels, 24 samples) | Act 2 boss panel **29.5% → 23.3%** (−6.2 ± 2.3) |
| "Never waste energy" combat filter | null |

Across 38 sparse-reward updates (~39k fights), the sampled training win rate on
the boss/elite mix moved only 0.51 → 0.55; v3 stayed flat at ~0.64.

**Why, as best we can tell:**

1. **Long-horizon, sparse credit.** A boss fight is ~40 decisions. Win/loss
   (or end-of-fight HP) is a weak, delayed signal for any one card play. Dense
   shaping did not fix it at ~20k fights. This is probably not an exploration
   problem: the combat head is *not* saturated (median top-action probability
   0.82, median top-2 gap 1.8).
2. **Narrow training distributions cause forgetting.** v2 trained on bosses
   and elites and degraded the Act 1 boss in full games. A 120-fight held-out
   panel was too small to see it (§6).
3. **Imitating search one decision at a time breaks plans.** Search picks an
   action as part of a multi-turn plan. The single-step candidate scorer
   reproduces only ~50% of those choices, and a policy that follows half of
   several different plans is worse than a coherent one. This happened with
   both privileged and public labels, so hidden information was not the cause.
   **Search helps when it plays the fight, not when its choices are copied.**
4. **Representation is unverified.** Whether the encoder captures what matters
   for multi-turn planning (draw-pile composition, enemy buff schedules,
   multi-turn damage races) has not been audited. See §7.3, item 1.

---

## 5. The key mechanisms

1. **Saturated logits block learning.** If an option's sampling probability
   is ~1e-8, PPO has no gradient toward it however good it is. Check logit
   gaps (`logit_gaps`-style scripts) before blaming the reward.
2. **Per-decision effect vs noise.** Strategic decisions move the return by
   ~0.02 reward units (~0.4 floors), against ~0.15 of advantage noise. Paired
   policy-level evaluation over 1,024 worlds (SE ~0.25 floors) is far more
   statistically efficient for a handful of strategic parameters than
   per-decision credit.
3. **The critic cannot see delayed value, so bootstrapped credit is biased
   against it.** Use Monte-Carlo credit (λ = 1) for OOC, or give the critic
   features that see deck quality (§7.1).
4. **Privileged search is mostly clairvoyance.** The study patch's
   `search_best_action` re-seeds only the searcher's own RNG; its simulations
   use the *true* battle RNG streams. With the draw order shuffled but the RNG
   true it wins 44% of Act 2 boss fights; with all streams re-seeded (public
   information) it wins 32–37%. About two-thirds of the apparent headroom was
   foreknowledge. **Never present the privileged search as an agent
   capability.**
5. **Deck and HP explain most Act 2 boss outcomes** (investigation E5).
   Combat play is a real but secondary lever: public search at 64 samples
   recovers ~8 points of win rate at that boss.

---

## 6. Pitfalls and bugs found (so you don't repeat them)

- **Small panels hide regressions.** At 120 fights per group the SE is ~4
  points; a real 4-point Act 1 boss drop was invisible. Judge combat changes
  by full-game paired evals, or panels of ≥400 fights per group.
- **1,024-world win counts are noise** at ~2–5% win rates.
- **Calling a plateau on one eval step.** I called the main run "plateaued" at
  update 32 (+2.65); update 36 was +3.14. Wait for two consecutive flat
  evals.
- **Resume ignored `--lr`.** `optimizer.load_state_dict` restored the saved
  lr. Fixed: the config now wins, and the resume event logs both values.
- **Queue watchdog killed renamed runs.** It read
  `data/rl_runs/<queue name>/log.jsonl`. Resumed and branched runs log under
  `--out`, so they looked dead after 40 minutes. Fixed: it reads the `--out`
  log and `stdout.log`.
- **Memory:** see §3.7. Check `top -o mem` and `sysctl vm.swapusage` before
  adding jobs. The Mac has 48 GB.
- **Battery.** The queue runner SIGSTOPs its jobs on battery and resumes them
  on AC. Ad-hoc `nohup` jobs are not covered, and a hand-rolled resume watcher
  dies with the agent session. **Queue every heavy job.**
- **Native rebuilds.** Overwriting a loaded dylib in place can kill running
  processes on macOS, so build new variants into a separate dir
  (`BUILD_DIR=build_pub scripts/rl/build_native_rl.sh`). Native source edits
  must be captured in `patches/sts_lightspeed_rl_panel.patch`.
- **Panel fights** must never run the game's post-battle code on scratch
  copies (`exit_battle` → `std::bad_function_call`). `panel.py` stubs it.
- **Spawn semantics.** Scripts using workers must be files with a
  `__main__` guard, not stdin.
- **Reward weighting is not the win problem.** A win is already worth ~41
  floors. The issue is rarity and measurement, not weight.

---

## 7. Future directions

### 7.1 Out-of-combat (where the gains have come from)

1. **Continue the loop.**
   - Run more λ = 1.0 updates on the best lineage (`ooc_r2_lam1` → 32).
   - Re-distil smith and elite if their sampled rates keep drifting.
   - Look for the next collapsed habit with logit-gap and choice-statistics
     scans: events, boss-relic choice, Neow, shop composition.
2. **Black-box search over strategy parameters.** A few interpretable
   thresholds (smith HP, elite HP per act, rest-before-boss HP) can be
   optimised by paired evaluation (cross-entropy method or coordinate ascent),
   then distilled. This is cheap and robust, because paired evals are the
   lowest-noise signal we have.
3. **Card picks** are the most frequent strategic decision and the least
   understood.
   - Hand rules do not fit them, and the fight-panel teacher was too noisy at
     4–8 replays.
   - Options:
     - many more replays per option (reliability grows with replays);
     - learned deck-strength value: train a "deck → boss win probability"
       regressor on many panel fights, then use it as a teacher or a critic
       feature;
     - simply more λ = 1.0 PPO compute.
4. **A better OOC critic.** Give it features that see deck quality, such as a
   distilled fight-panel value, upgrade counts, or a set encoder over the deck.
   With a critic that can see delayed value, λ < 1 becomes safe again, and the
   variance of λ = 1 can be avoided.
5. **Variance reduction.** World-grouped baselines (GRPO-style groups on the
   same world seed) and larger batches. Compute is the bottleneck (laptop
   only; krill-mill later).
6. **Avoid saturation at the source.** When pretraining or distilling, use
   label smoothing or an entropy floor on OOC heads, so later RL can still
   explore.

### 7.2 Play-time search (an avenue to explore; we may decide not to use it)

**What it is.** In boss fights, choose each combat action by the plurality
vote of native search over K sampled hidden worlds. For each sample:

1. deep-copy the battle (`BattleContext.copy`);
2. shuffle the draw pile;
3. re-seed every battle RNG stream (`reseed_rngs`): enemy AI, card randomness,
   reshuffles, potions, monster HP;
4. search N simulations.

This uses only information a player has. Code:
`PanelWorker.search_votes_public`, `scripts/rl/eval_search_boss.py`; it needs
`build_pub`.

**Measured** (Act 2 boss fights from the bank, paired with the greedy combat
head on the same starts):

| Controller | Win rate | Seconds per decision |
| --- | --- | --- |
| Greedy combat head | 27.5–29.4% | — |
| Public search 8 × 1,000 | 31.7% (n = 120) | 0.11 |
| Public search 24 × 1,000 | 35.0% (n = 120) | 0.26 |
| **Public search 64 × 1,000** | **37.4%** (n = 238; 38 vs 19 flips, p ≈ 0.016) | 0.68 |
| (Privileged search 8 × 1,000, not admissible) | 44.2% (n = 120) | 0.13 |

**Cost:** ~40 decisions per boss fight, so ~25–30 s per boss fight at
64 × 1,000.

**Full game** (`scripts/rl/eval_search_boss.py`, 64 × 1,000, search only in
Act 2 and 3 boss fights). Round-2 start agent, paired on 2,019 win-set worlds:

| | Act 2 boss pass | Act 3 boss reach | Act 3 boss pass | Wins |
| --- | --- | --- | --- | --- |
| No search | 38.0% | 282 | 18.8% | 53 (2.63%) |
| Search | **46.1%** | 341 | 20.2% | **69 (3.42%)** |

- McNemar p = 0.085. Mean floor +0.78 [0.54, 1.02].
- The Act 2 boss gain (+8.1 points) matches the fight-bank estimate exactly.
  The Act 3 boss gain is small.
- Cost: ~1.6 h for 2,048 games on 6 threads; 1.3% faults vs the usual ~1%.

**Open questions before adopting it:**

- The effect on top of the best network (`ooc_r2_lam1/iter_0012.pt`, 3.8%
  learned-only), and on a larger world set for a decisive win test.
- Why the Act 3 boss gains less. Perhaps those fights are more deck-limited;
  perhaps more samples are needed.
- Whether search should also cover elites.
- How far gains keep scaling with samples and simulations.
- **Whether "the agent" may use play-time computation at all.** This is a
  definitional choice for the project.

If adopted, report learned-only and search-augmented results separately. If
not adopted, it remains useful as a measurement of combat headroom and as an
in-the-loop teacher (§7.3, item 2).

### 7.3 Proposals for improving combat

Ordered roughly by expected value per effort.

1. **Audit the representation first** (cheap, de-risks everything else).
   - Probe the combat encoder: can it linearly decode draw-pile composition,
     incoming damage over the next 2 turns, enemy buff timers, and lethal-
     this-turn?
   - Compare the policy's mistakes with search's choices by situation type.
   - If key information is absent, no training method will fix the head.
2. **Search *in* the training loop (expert iteration done properly).**
   - Play boss fights with public search during collection, so the learner
     sees search-quality states.
   - Train the **value head** on the outcomes of those fights.
   - Train the policy on search visit distributions *and* on the fights' own
     outcomes, then iterate. The aim is AlphaZero-style improvement, where
     search and network improve each other.
   - Difference from the failed imitation: the labels come from fights played
     coherently *by search*, value targets carry multi-turn information, and
     search can stay in the loop at play time if adopted.
3. **Value-guided lookahead (cheap, public, plan-coherent within the
   horizon).**
   - Train a fight-state value V(s) = P(win) (or HP-adjusted) by supervised
     learning on many simulated fights. Fights cost ~0.07 s each, so 10^5–10^6
     are feasible.
   - At play time, do depth-limited lookahead to the end of the current turn
     (or 2 turns) in the simulator with re-seeded hidden randomness, scoring
     leaves with V.
   - It needs no plan memory, because the plan is recomputed each turn, and it
     is much cheaper than full search.
4. **Turn-level (macro) actions: make the policy choose whole turns.**
   - The unit of planning in Slay the Spire is the turn. Generate candidate
     full-turn sequences (beam search over card plays within the turn in the
     simulator; within-turn randomness is limited), and score each
     end-of-turn state with a learned value or policy network.
   - Train the scorer with RL on fight outcomes (credit per turn: ~10 turns
     per fight instead of ~40 actions) or by distilling search's
     `best_sequence` turn by turn.
   - This directly removes the "half a plan" failure mode.
5. **Models capable of multi-turn planning:**
   - **Set/sequence encoders:** attention over cards in hand, draw, discard and
     exhaust, and over enemies with their intents and buffs, instead of hashed
     bags. A transformer over the fight history, or a recurrent state, lets the
     policy carry a plan across decisions.
   - **Plan-conditioned policies:** at turn start, predict a plan embedding (or
     a target end-of-turn state), then condition per-action choices on it.
     This gives coherence by construction.
   - **Learned dynamics (MuZero-style)** are probably unnecessary, since we have
     an exact simulator. Its value would be cheaper rollouts and handling
     hidden information in latent space.
6. **Sequence-level imitation.** If imitating search again, imitate *whole
   turns* (or the best sequence), with DAgger at turn granularity, never
   independent per-decision votes.
7. **Scale and curriculum for fight RL.**
   - Much larger fight banks (boss-heavy, but always including hallway and
     elite fights to avoid forgetting).
   - Per-turn shaped credit.
   - Judge only by full-game paired evals and panels of ≥400 per group.

### 7.4 Measurement and process

- Claim wins only from the 4,096-world win set with McNemar. Confirm selected
  winners on fresh worlds. Keep the standard set for development.
- Use `scripts/rl/compare_evals.py A.json B.json …` for every comparison.
- Decide stop rules before long runs: for example, "two consecutive flat
  evals", or "full-game paired < 0 at update 10 → stop".
- Compute is the bottleneck. When krill-mill is available, run replicate
  seeds, larger batches and the combat projects there.

---

## 8. Reproduction: where things are

- **Data (local only; `data/` is gitignored):** `data/rl_runs/`.

  | What | Path |
  | --- | --- |
  | Runs | `<run>/log.jsonl`, `eval_XXXX.json`, `iter_XXXX.pt` |
  | Win set | `win_eval/` |
  | Fresh-world evals | `fresh_eval/` |
  | Overlay evals | `combos/` |
  | Fight banks | `banks/` |
  | Labels | `labels/` |
  | Queue state | `queue.json` |

- **Lineage of the best agent:**

  | Step | Checkpoint |
  | --- | --- |
  | Original | `joint_act3_0030.pt` |
  | E8 PPO, seed 1, update 36 | `ooc_main_s1/iter_0036.pt` |
  | Distil smith ≥85% | `distill_smith85_i36/distilled.pt` |
  | PPO λ = 1, seed 3, 12 updates | `ooc_pfd_lam1/iter_0012.pt` |
  | Distil smith + elite ≥90% | `distill_r2_lam1i12/distilled.pt` |
  | PPO λ = 1, seed 5, 12 updates | **`ooc_r2_lam1/iter_0012.pt`** |

- **Key commands.** Prefix every command with `STS_LIGHTSPEED_BUILD=…`, and run
  heavy ones through `scripts/rl/queue_runner.py add NAME [--lane L] -- …`.

  ```bash
  # OOC PPO (recommended settings)
  .venv/bin/python scripts/rl/train_ooc.py --out data/rl_runs/X --checkpoint CKPT \
    --baseline data/rl_runs/baselines/orig_argmax_1024.json --seed S --iterations 16 --eval-every 4 \
    --gae-lambda 1.0 --collectors 4 --workers 16 --update-threads 6
  # measure a strategy rule (paired vs a saved eval)
  .venv/bin/python scripts/rl/eval_policy.py --checkpoint CKPT --n 1024 --collectors 3 --workers 9 \
    --smith-above 0.85 --elite-above 0.9 --baseline BASE.json --output OUT.json
  # win-set eval
  .venv/bin/python scripts/rl/eval_policy.py --checkpoint CKPT --n 4096 --world-start 6000000000 ...
  # distil rules
  .venv/bin/python scripts/rl/collect_rule_labels.py --checkpoint CKPT --smith-above 0.85 --elite-above 0.9 --output LABELS
  .venv/bin/python scripts/rl/distill_ooc.py --records LABELS --checkpoint CKPT --out OUT --types rest,elite --epochs 16 --lr 0.0003
  # compare
  PYTHONPATH=src .venv/bin/python scripts/rl/compare_evals.py A.json B.json
  # public search (needs build_pub)
  BUILD_DIR=build_pub scripts/rl/build_native_rl.sh
  STS_LIGHTSPEED_BUILD=external/sts_lightspeed_rl/build_pub .venv/bin/python scripts/rl/eval_search_boss.py \
    --checkpoint CKPT --samples 64 --simulations 1000 --rooms BOSS --min-act 2 --baseline BASE.json --output OUT.json
  ```

- **Tests:** `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`
  (473 tests). These cover micro-batched PPO equivalence, menu filters,
  strategic rows and combat shaping.

## 9. Open questions

- Play-time public search raised Act 2 boss pass +8 points in full games (wins
  +30% relative, p = 0.085). Does it stack on the best network, and why is
  the Act 3 boss gain small?
- How far does the λ = 1.0 OOC loop go with more rounds and compute, and does
  smith drift need periodic re-distillation?
- Can a critic with deck-quality features make bootstrapped credit unbiased
  again, and cut λ = 1 variance?
- Does the combat encoder contain the information needed for multi-turn plans
  (§7.3, item 1)? This decides whether combat work starts with representation
  or with training.
- Is a higher learning rate faster for OOC PPO? The test was cancelled.
