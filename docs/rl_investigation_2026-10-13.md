# Why the non-LLM RL agent is not improving — investigation and proposal

13 October 2026, Claude (Opus 5.5), worktree `claude/slay-spire-rl-debug-da6d22`
(branched from `codex/long-curriculum-v2` @ `453c768`). Scope: the non-LLM PPO
composite (two 2.3M-parameter candidate-scoring MLPs, combat + out-of-combat
"OOC"), Ironclad A0, Acts 1–3. All measurements below are new, bounded,
TRAIN-split diagnostics on seed namespaces never used by the campaigns
(5e9+, 6e9+, 7e9+). No DEV/final data, campaign state, budget ledger or pinned
binary was touched. Scripts: `scripts/investigations/rl_debug_2026_10_13/`.

This builds on, and mostly agrees with, the Codex critique on
`codex/nonllm-critique` (`docs/nonllm_training_critique_2026-10-12.md`), which
already covers the action-dedup bug, encoder information loss, critic blindness
to candidate features, GAE-at-microdecision-scale, the floor-vs-win reward
trade-off and operational fixes. The findings here are the ones that critique
did not measure, and they change the priority order.

## TL;DR

0. **A cleaned-up OOC update improved the original in ~40 minutes, where v2
   made it worse.** Frozen greedy combat, OOC-only macro credit and 768-game
   batches gave **+0.71 ± 0.28 floors** paired on 1,007 worlds after 8 updates
   (Act 1 clear +1.8pp, Act 2-boss reach +4.8pp, wins 2 → 6), shifting map
   choices back toward fights. One seed: promising, not established (E8).
1. **PPO is not "forgetting"; it is random-walking downhill.** On 512 fresh
   TRAIN worlds with the *training* (stochastic) policy, mean floor went
   original 26.1 → CP54 22.9 → CP128 24.0. The learner got worse at its own
   objective on its own distribution.
2. **The gradient is ~90% noise.** Policy gradients from two disjoint 64-episode
   halves (≈ ⅔ of a v2 chunk) have cosine 0.07 (combat actor) / 0.17 (OOC actor).
   After 13.5k v2 Adam steps, every weight block moved 0.7–1.4× the
   *pure-random-walk* prediction `lr·√(steps·numel)` and ~100× less than a
   consistent gradient would. Advantage normalisation + Adam turn noise into
   full-size steps.
3. **What signal there is in the OOC advantage is biased toward avoiding
   fights,** and PPO followed it. GAE (bootstrapping across the combat critic)
   scores rest/shop/event map choices +0.02–0.03 above monster fights and favours
   card-skipping, while realised Monte-Carlo outcomes point the other way.
   Under v2, event picks rose 31.8% → 35.9%, monster picks fell 29.2% → 26.8%,
   fights per run fell 11.7 → 10.2–10.7, Act 1 card-skip doubled 8% → 18%, and Act 1
   boss deaths rose 99 → 180 /512. That is exactly the observed regression.
4. **Training data comes from a much weaker policy than the one evaluated.**
   Greedy 29.6 vs sampled 26.2 mean floor on the same worlds; combat sampling
   alone costs ~2 floors. OOC credit is dominated by self-inflicted combat noise.
5. **Deck/HP (strategy), not combat skill, is the main wall past Act 2.** Of the
   Act 2 boss fights the learner lost, privileged native search with 2000
   simulations wins only 12/60; runs that lose arrive with ~56 HP vs ~72 for
   runs that win. Combat is a real but secondary lever (≈ +13pp on Act 1 boss,
   ≈ half of Act 2 hallway deaths recoverable by strong search).
6. **The evaluation protocol cannot see the effects being chased.** At n=192 the
   80%-power minimum detectable effect is ≈ 11pp Act 1 clears, 8.6pp Act 2
   clears, 2.8pp wins. "2/192 → 0/192" (which paused v2) and the overnight
   pilots' ±1–3-clear differences are noise. 1,024 greedy games take ~3–4 min.
7. **Strategic gaps are real but co-adapted.** The original policy smiths 0 times
   in 2,477 campfires (58% of rests at ≥75% HP), buys card removal 0/2,402 times,
   takes an elite 1 time in ~5,400 map choices, and takes Neow's boss-relic
   swap 504/507 times. Crude hand rules: smith-when-healthy +0.40 floors; forced
   removal −0.59; elite-seeking −0.60; key-avoidance +0.07. Fixing strategy needs
   a learning signal that can see multi-floor consequences, not a rule patch.
8. **A cheap, reliable strategic signal exists: short fight replays.** Replaying
   a learner-reached boss fight 4 times gives a state score with split-half
   reliability ≈ 0.8, separating states that will lose (P≈0.05–0.26) from ones
   that win (P≈0.66–0.92), and only partly explained by HP (E9). By contrast a
   single strategic decision moves the whole-game return by ~1% of its noise,
   and same-world branching only cuts that noise ~3.5× because outcomes
   decorrelate chaotically (E10).

**Plan in one line:** stop small-batch PPO; replicate and adopt the E8 recipe
for OOC; make a fight-panel deck-strength evaluator the OOC learning signal
(teacher → distilled value → shaping); then targeted combat expert iteration
on run-ending fights; judge everything on ≥1,024-world evaluations.

## Evidence

Checkpoints: original `joint_act3_0030.pt` (`57f1e961…`, the v2 starting point),
v2 CP54 (`full3_continuous_00000054…`) and CP128 (latest). Native build 0.1.1
(`40ab5056…`), frozen `.venv`. Probes run 8–12 CPU workers; 512 fresh full-run
TRAIN worlds take ~80–150 s.

### E1. PPO made the policy worse at its own objective

Stochastic (training-time) policy, identical 512 TRAIN worlds:

| Checkpoint | Mean floor | Act 1 clear | Act 2 clear | Deaths at Act 1 boss |
| --- | ---: | ---: | ---: | ---: |
| Original | 26.10 | 76.1% | 6.1% | 99 |
| CP54 | 22.94 | 57.6% | 2.3% | 180 |
| CP128 | 24.04 | 63.3% | 2.7% | 157 |

The v2 per-chunk logs show the same drift on the learner's own fresh starts
(chunk 0–18 ≈ 25–26.7, later ≈ 22–25). This is not retention loss on old skills:
on-policy PG with small steps should go *up* in expectation. Something is wrong
with the gradient estimate, not just with what is protected.

### E2. The gradient is mostly noise, and the weights random-walked

`grad_snr.py`: 256 original-policy episodes (82k transitions); policy-gradient
(ratio=1, buffer-normalised GAE advantages, as `update_ppo`) on two disjoint
random episode groups, cosine averaged over 4 resamples:

| Episodes per half | Combat actor | Combat encoder | OOC actor | OOC encoder |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.03 | 0.04 | 0.12 | 0.18 |
| 32 | 0.07 | 0.09 | 0.10 | 0.00 |
| 64 | 0.07 | 0.12 | 0.17 | 0.05 |

For a 128-episode v2 chunk the implied full-buffer signal fraction is ≈ 0.13
(combat) / ≈ 0.3 (OOC), and each PPO *minibatch* (1,024 rows) is far worse.
Entropy and value gradients are small relative to PG (value/PG ≈ 0.08 on the
combat encoder, 0.25 on the OOC encoder), so critic interference is not the
main mechanism.

Weight displacement from the original, against the pure-noise Adam prediction
`lr·√(T·numel)` (each noisy Adam step moves each parameter ≈ lr in a random
direction) and the consistent-gradient prediction `lr·T·√numel`:

| Block (CP128, T=13,514) | ‖ΔW‖ | random-walk prediction | ratio | consistent-drift prediction |
| --- | ---: | ---: | ---: | ---: |
| combat state encoder | 8.67 | 9.42 | 0.92 | 1095 |
| combat scorer | 11.64 | 10.32 | 1.13 | 1199 |
| combat critic | 8.33 | 5.96 | 1.40 | 693 |
| OOC state encoder | 8.85 | 9.42 | 0.94 | 1095 |
| OOC scorer | 9.43 | 10.32 | 0.91 | 1199 |
| OOC critic | 4.93 | 5.96 | 0.83 | 693 |

CP54 (T=6,125) matches equally well (ratios 0.72–1.40). Only the combat critic
sits clearly above the noise line. Small per-update KL (≈0.005) is fully
compatible with this: many tiny uncorrelated steps accumulate to the observed
original→learner KL (0.79 combat / 0.29 OOC), starting from a checkpoint that
was itself *selected* as the best of many on DEV, so a random walk is expected
to go down.

### E3. The OOC advantage is biased toward avoiding fights

From `analyze_probe.py` on 512 original-policy episodes (157,798 transitions;
71% combat, 29% OOC; 31% of OOC and 15% of combat decisions are forced):

- The critic is mostly a floor counter. Value tracks Monte-Carlo return well by
  floor bucket, but within-act explained variance is low (Act 1: 0.36 combat /
  0.28 OOC; Act 2: 0.28 / 0.21; Act 3 OOC: −0.13) and start-of-run value has
  correlation −0.04 with the realised return.
- Raw advantages are tiny (|A| ≈ 0.07 in units where one floor = 0.05), and
  normalisation rescales them to unit size regardless of content.

Non-forced map choices, by chosen room (n≈1.1–1.5k each, stochastic original):

| Room | mean TD δ | mean GAE A | mean (MC − V) |
| --- | ---: | ---: | ---: |
| MONSTER | −0.001 | +0.008 | −0.008 |
| EVENT | +0.019 | +0.022 | −0.023 |
| REST | +0.029 | +0.034 | −0.018 |
| SHOP | +0.018 | +0.025 | −0.027 |

GAE ranks every non-combat room above fights by 0.014–0.026. Realised
outcomes rank fights *best* (each MC difference is only ~1 SE, so treat the MC
column as directional). Similarly, card skips get higher mean GAE than takes
(+0.008 vs +0.005) although their realised MC−V is worse (−0.053 vs −0.023).
The likely mechanism is cross-critic bootstrapping at phase boundaries: an OOC
choice that enters combat is scored by `V_combat(next) − V_ooc(now)` (mean δ for
OOC→combat map moves −0.009, OOC→OOC +0.012, both with SE ≈ 0.001), so any
calibration offset between the two heads becomes an action-type-specific bias.
PPO's drift follows the bias exactly:

| Stochastic, 512 worlds | EVENT | MONSTER | Fights / run | Card skip (all / Act 1) | Card-reward screens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 31.8% | 29.2% | 11.7 | 6.5% / 8.4% | 14,319 |
| CP54 | 35.4% | 27.4% | 10.2 | 15.2% / 18.3% | 12,064 |
| CP128 | 35.9% | 26.8% | 10.7 | 15.7% / 17.9% | 12,540 |

Fewer fights and more skips means fewer cards and a weaker deck at the Act 1
boss, where the extra deaths are. The policy also drifted into dead actions
that the 0.1.1 action space offers in a max-Act-3 game: "take key" went from
1 to 53–88 per 512 runs.

### E4. Training data come from a much weaker policy than the evaluated one

Same 512 worlds, original checkpoint:

| Combat / OOC sampling | Mean floor |
| --- | ---: |
| sampled / sampled (what PPO trains on) | 26.2 |
| sampled / greedy | 27.4 |
| greedy / sampled | 28.1 |
| greedy / greedy (what is evaluated) | 29.6 |

Roughly 2 floors of every training game are lost to combat sampling misplays,
which the OOC learner must treat as noise in its credit signal. Stochastic
outcomes on the same world correlate ρ ≈ 0.42 (≈42% of floor variance is world
difficulty), so per-world baselines (K rollouts per seed) or paired designs
would remove a further ~40% of return variance.

### E5. Where runs die, and whether stronger combat would save them

Original, greedy, 507 valid runs: Act 1 clear 88.6%, Act 2 clear 11.8%, one
win. Stochastic deaths: 47% at bosses (Act 1 boss 99/507; of 162 runs reaching
the Act 2 boss only 19% win), 33% in hallway fights, 19% at elites.

`boss_replay.py` reconstructs the learner's own fight from its action trace and
replays it with each controller (privileged native search sees hidden state, so
it is an optimistic bound on combat play):

| Fights the run lost | Learned greedy | Learned sampled (×3) | Search 200 | Search 2000 |
| --- | ---: | ---: | ---: | ---: |
| Act 1 boss (n=60, entry HP 70) | 16/60 | 12–20/60 | 24/60 | 31/60 |
| Act 2 boss (n=60, entry HP 56) | 3/60 | 1–5/60 | 9/60 | 12/60 |
| Act 2 hallway/elite (n=50, entry HP 32) | 12/50 | 12–14/50 | 19/50 | 26/50 |

Fights the run *won* mostly stay won under every controller (learned greedy
56/59 Act 1 boss, 22/29 Act 2 boss; search-2000 56/59 and 24/29). Reading: most Act 2 boss losses are decided before the fight (deck
strength, entry HP); mid-Act-2 deaths start at 40% HP and about half are
combat-recoverable; combat improvement is worth roughly +10–15pp at bosses.

### E6. Strategic gaps and what crude fixes buy

Original greedy policy, 507 runs:

| Behaviour | Observation |
| --- | --- |
| Campfires | smith offered in ~2,100/2,477; chosen 0 times. 58% of rests at ≥75% HP. "recall" (Ruby key) 62 |
| Shop | card removal offered 2,402 times, bought 0; potions ≈ 70% of purchases |
| Map | ELITE chosen once in ~5,400 non-forced choices |
| Neow | boss-relic swap 504/507 (loses Burning Blood) |
| Rewards | only skips when the remaining item is a key (fine) |

`overlay_probe.py` overrides the learned choice only when a rule fires (paired
512 TRAIN worlds, greedy; SE of a paired difference ≈ 0.4 floors):

| Overlay | Mean floor | Δ | Act 1 / Act 2 clear |
| --- | ---: | ---: | ---: |
| none | 29.57 | — | 88.6% / 11.8% |
| never take Act 4 keys | 29.64 | +0.07 | 88.8% / 11.6% |
| + smith when HP ≥ 55% (≥75% before boss) | 29.97 | +0.40 | 88.6% / 12.0% |
| always buy removal, remove Strike/Defend | 28.98 | −0.59 | 86.1% / 9.8% |
| take elites when HP ≥ 70% (Acts 1–2) | 28.97 | −0.60 | 86.2% / 11.0% |
| smith + removal + no keys | 29.80 | +0.23 | 86.8% / 12.4% |

The OOC policy sits in a local optimum co-adapted to its combat policy and its
floor-progress reward: elite avoidance and no removal are locally rational for
this fighter. Only coordinated learning with a signal that sees multi-floor
consequences will move it.

### E7. Evaluation power

Greedy floor SD ≈ 8.0. Two-sample 80%-power minimum detectable effects:

| n worlds | Act 1 clear (p≈0.8) | Act 2 clear (p≈0.1) | Win (p≈0.01) | Mean floor |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 22.9pp | 17.1pp | 5.7pp | 4.6 |
| 192 | 11.4pp | 8.6pp | 2.8pp | 2.3 |
| 1,024 | 4.9pp | 3.7pp | 1.2pp | 1.0 |
| 4,096 | 2.5pp | 1.9pp | 0.6pp | 0.5 |

1,024 greedy full games take ~3–4 minutes on 12 workers.

### E8. A cleaned-up OOC update (bounded experiment)

Setup (`ooc_macro_ppo.py`): start from the original; combat head frozen
(bit-identical, asserted every update) and run greedy during collection;
OOC-only macro GAE (λ=0.95, bootstrap with `V_ooc` at the next OOC decision);
768 fresh TRAIN games per update (new seeds every update); non-forced OOC
transitions only (~51k per update); PPO 3 epochs × 4,096 minibatch, LR 3e-4,
clip 0.2, no entropy bonus, before-update KL guard 0.05; 8 updates. ~42
minutes wall on 12 CPU workers including evaluations. Evaluation is greedy on
the same 1,024 TRAIN-eval worlds every 2 updates (no checkpoint selection; the
final update is the result).

| Update | 0 | 2 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy eval mean floor (±0.26) | 29.19 | 28.98 | 29.37 | 29.39 | **29.93** |
| Act 1 / Act 2 clear | 86.9 / 11.5% | 83.6 / 11.0% | 86.5 / 11.4% | 87.3 / 10.3% | 88.6 / 12.1% |

Training-time floor (sampled OOC, greedy combat, fresh worlds each update):
28.11, 28.19, 28.34, 28.17, 28.42, 29.20, 28.20, 28.92.

Paired on 1,007 common worlds, update 8 vs original: **+0.71 ± 0.28 floors**
(t = 2.5; bootstrap 95% CI [0.18, 1.24]); Act 1 clear 86.8 → 88.6% (77 worlds
gained, 59 lost); Act 2 clear 11.5 → 12.0%; reaching the Act 2 boss 47.9 →
52.7%; HP on arrival 63.4 → 65.2; wins 2 → 6. Behaviour moved in the
*opposite* direction from v2's biased drift: monster picks +1.5pp, event picks
−1.6pp. Smith stayed at 0% (never sampled, so never learned; item 9), and
Act 4 key-taking rose 121 → 144 (item 4).

For comparison, v2 used ~16.5k games and 13.5k optimizer steps to *lose* ~2
floors (stochastic) from the same start. This is one seed with no component
ablation, so it shows the direction is fixable, not which change fixed it,
nor how far it goes. Replicating it with ablations is the cheapest next step
(X0b).

### E9. Short simulated fights are a reliable, cheap state-strength signal

From the E5 replays, each learner-reached fight start was played 4 times with
the learned combat policy (greedy + 3 sampled). Win probability per state:

| Fight starts | P(win), runs that died there | P(win), runs that survived | Split-half reliability (2 vs 2 replays) | corr with entry HP |
| --- | ---: | ---: | ---: | ---: |
| Act 1 boss (n=120) | 0.26 | 0.92 | 0.82 | 0.27 |
| Act 2 boss (n=89) | 0.05 | 0.66 | 0.77 | 0.39 |
| Act 2 fatal hallway/elite (n=50) | 0.25 | — | 0.36 | 0.55 |

Four short replays give a state score with reliability ≈ 0.8 at bosses, and HP
explains only part of it: the rest is deck, relic and potion strength. Compare
whole-game returns, where one decision's effect is ~1% of the noise (E10). The
C++ already supports starting a battle with a chosen encounter from a
`GameContext` (`BattleContext::init(gc, MonsterEncounter)`), but the Python
binding only exposes `init(gc)`, so this signal is currently only available at
fights the run actually reaches.

### E10. Neither macro credit nor same-world branching fixes the SNR

- `grad_snr.py --macro` (greedy frozen combat, OOC-only macro GAE): OOC-actor
  half-batch cosine ≈ 0.11 at 64 episodes per half (0.02 at 16, −0.11 at 32),
  no better than the 0.17 under v2's micro GAE. Macro credit removes the
  cross-critic bias (E3) but carries each segment's whole outcome variance.
- Size of a real strategic effect: smithing at every eligible campfire (~6 per
  run) is worth +0.4 floors per run (E6), i.e. ~0.07 floors per decision,
  against a per-game SD of ~8 floors.

- `branch_probe.py`: 700 non-forced OOC decisions sampled from 507 greedy
  original-policy runs; each alternative (≤3) forced on the same world and
  continued greedily (1,664 branches, 5 faults). 63% of alternatives end on
  exactly the same floor; the paired difference SD is 6.0 floors, against
  ≈ 11.3 for two independent games. Same-world branching is therefore ≈ 3.5×
  more sample-efficient than independent comparisons, but a 0.5-floor effect
  still needs ~150 branches per decision type to resolve at 1 SE. The policy's
  choice beats a random alternative by only 0.14 floors on average (17% of
  alternatives better, 20% worse). Selected per-type means (alternative − policy,
  floors ± SE): take gold → skip rewards −2.7 ± 0.9; rest → recall −0.9 ± 0.5;
  rest → smith +0.5 ± 0.9; take card → skip card +0.9 ± 0.6. Card-select screens
  showed alternative +4.2 ± 1.6 (n=36), but inspection shows this is chaos, not
  a policy flaw: choosing a *different copy of the same card* (Strike #0 vs
  Strike #4 at Neow) changes the same world's final floor by +17. Card instance
  identity perturbs later shuffles, so once anything differs the futures
  decorrelate. This is a hard floor on how much common-random-number
  branching can reduce variance in full-game returns, and another reason to
  prefer short, replicated fight simulations (E9) as the OOC signal.

## Proposal: what to change, in priority order

### P0 — Stop spending compute where it cannot work (today)

1. **Do not resume v1/v2 or spend the remaining 3.96 h on any 128-game-per-update
   PPO variant**, including the proposed "frozen-original anchor + LR 1e-5"
   pilot. Anchoring and lower LR slow a random walk; they cannot create signal
   (E2), and an anchor to the original also anchors the biased fight-avoidance
   gradient's starting point (E3).
2. **Re-base development evaluation on ≥1,024 fixed TRAIN-eval worlds, greedy,**
   reporting mean floor ± SE, Act 1/Act 2 clear, Act 1/Act 2 boss pass rate,
   entry HP at each boss. Keep 4,096 worlds plus a fresh DEV cohort for
   promotion decisions. Pre-register the MDE (E7) before every pilot; a pilot
   that cannot move the metric by more than its MDE is not a test. Stop reading
   anything into win counts below ~10.
3. **Instrument every update with three cheap learning-health signals:**
   half-batch gradient cosine per phase (target ≥ 0.5), parameter displacement
   against the `lr·√(T·numel)` random-walk line, and greedy-vs-stochastic
   return. These would have flagged v1/v2 within the first hour.
4. **Remove dead actions** for `max_act=3` runs: Ruby key ("recall"), "take key",
   and other Act-4-only choices. They have no possible benefit and PPO drifted
   into them. This is an environment/menu change, so it needs a version bump.

### P1 — Give out-of-combat strategy a usable learning signal (the binding constraint)

The core quantitative fact (E2, E9, E10): a single strategic decision moves the
whole-game return by ~1% of its per-game noise, and neither macro credit nor
paired same-world branching changes that much. Whole-game return is the wrong
*training* signal for deck-building at this scale. Short simulated fights from
the current deck are a cheap, reliable one (split-half reliability ≈ 0.8 with 2
replays). Build on that.

5. **Build a fight-panel "deck strength" evaluator and make it the OOC learning
   signal.** Expose `BattleContext::init(gc, MonsterEncounter)` (already in
   C++; only `init(gc)` is bound) so any OOC state's `GameContext` copy can
   fight a chosen encounter. Define `panel(state)` = win probability and
   expected HP loss of the frozen combat policy (sampled, K≈4–8) against this
   act's boss, its elites and a few hallway encounters, at current HP. Use it
   three ways, in this order:
   - **Afterstate teacher** for decisions with deterministic afterstates (card
     pick/skip, smith target, rest vs smith, removal, purchases, boss relic,
     Neow): apply each option on a state copy, score the panel, label the best.
     Rest-vs-smith and elite-or-not become measurable trade-offs (HP now vs
     deck strength at the boss) instead of folklore.
   - **Distil** labels into a fast neural `V_deck(state)` on typed deck and
     relic features (P3), so the policy does not pay simulation cost at play time.
   - **Potential-based shaping** for end-to-end OOC RL: `r' = r + Φ(s') − Φ(s)`
     with Φ = `V_deck`. This is policy-invariant at γ=1 with Φ(terminal)=0, and
     gives every OOC decision an immediate, low-variance advantage.
   Validate by Act 1 and Act 2 boss pass rate and mean floor on 1,024-world
   greedy evals, not by teacher agreement.
6. **During OOC learning, freeze combat and run it greedy.** It removes ~2
   floors of self-inflicted noise per game (E4) and matches deployment.
7. **When using whole-game RL for OOC, use OOC-only macro credit.** Sum rewards
   from one OOC decision to the next and bootstrap with `V_ooc` at the next OOC
   decision. This removes the cross-critic phase-boundary bias that rewarded
   avoiding fights (E3). It does *not* by itself raise SNR (E10).
8. **Batch for signal, not for chunk cadence.** 768–2,048 games per update,
   1–3 epochs, no entropy bonus (OOC entropy on real choices is already ~0.2
   nats, and the entropy term is a consistent bias when PG is noisy), and a
   per-world group baseline (K=4 rollouts per seed, subtract the world mean)
   removing the ~42% between-world variance (E4). At ~5 games/s, 1,024 games
   is ~3.5 minutes. Monitor half-batch cosine; if it is below ~0.3, the update is
   mostly noise whatever the loss curves say.
9. **Explore at OOC choices on purpose.** BC left near-zero probability on
   whole strategies: smith was sampled 0 times in 2,197 *stochastic* campfire
   decisions. PG cannot learn the value of an action it never samples. The
   afterstate teacher (item 5) sidesteps this; for PG, sample OOC choices from a
   tempered or ε-uniform behaviour policy and store its log-prob.

### P2 — Combat: worth ~10–15pp at bosses (after P1 starts paying off)

10. **Targeted expert iteration on the fights that end runs:** learner-reached
   boss, elite and low-HP (≤50%) Act 2 states, labelled by native search at
   ≥2,000 simulations (E5: search-2000 wins about twice as many of the
   learner's lost fights as the learner does). Use `boss_replay.py`-style
   held-out fight panels as the fast metric, then full games. Keep the
   privileged-teacher caveat: imitating a scum searcher is approximate, so judge
   it by live outcomes.
11. **Never co-train the two heads with one buffer-wide advantage
    normalisation, one gradient-norm cap and one optimizer.** Separate
    optimizers and update phases; train combat on combat-only batches.

### P3 — Representation and critic (agree with the Codex critique; second order)

12. Adopt the action-identity fix and typed encoder from `codex/nonllm-critique`,
    and give the OOC critic the deck and relics as an entity set. This matters
    more for P1.5 (the deck-strength value needs precise deck features) than for PPO,
    whose problem is noise and bias rather than resolution. Do it as part of
    building the value model, not as a blocker before it.

### P4 — Process

13. The current workflow optimises for auditability (hash ledgers, active-second
    accounting to 12 decimals, embargo exceptions, 4k-line progress logs) at the
    expense of iteration speed and statistical power. Keep the heavy protocol
    for final claims. For development, a one-page condition card is enough:
    hypothesis, metric, MDE, n, stop rule. Since 11 October there have been 76
    commits and ~32k lines of new docs. The one decision-relevant quantity,
    whether the gradient carries signal, was never measured.

## Experiments to run next (ordered; each has a cost and a decision rule)

| # | Experiment | Cost | Go / no-go |
| --- | --- | --- | --- |
| X0 | Evaluation and health harness: 1,024-world greedy TRAIN-eval (mean floor ± SE, Act 1/2 clear, boss pass and entry HP), plus half-batch gradient cosine and random-walk displacement logged per update. `scripts/investigations/rl_debug_2026_10_13/` has working versions of all of these. | ~2 h | Required before anything else is called a result |
| X0b | **Replicate E8 and attribute it**: same script, 2 more seeds; plus ablations of (a) sampled instead of greedy combat during collection, (b) micro GAE instead of macro, (c) 128-game batches at matched games. Add the dead-action mask and ε=0.05 OOC exploration as a further arm. ~45 min per arm on 12 workers. | ~½ day | Mean gain ≥ +0.5 floors across seeds → adopt as the OOC fine-tuning recipe. The ablations say which pieces are load-bearing |
| X1 | **Fight-panel evaluator.** Bind `BattleContext::init(gc, encounter)`; `panel(state, encounters, K)` with the frozen combat policy. Validate from mid-act states (floor ~8–14): split-half reliability, and AUC for predicting the real boss outcome. Bundle the Act-4-key menu removal into the same env version bump. | ½–1 day | Reliability ≥ 0.7 and AUC clearly above entry-HP-only AUC → X2 |
| X2 | **Afterstate teacher on the two most frequent strategic choices** (card reward ~28/run, rest vs smith ~5/run). Score each option's afterstate with X1 and overlay the teacher on the original policy, as in `overlay_probe.py`. Paired 1,024-world greedy eval. | ~1 day | ≥ +1 floor or ≥ +4pp Act 1 boss pass → distil into the OOC head (BC on teacher labels), extend to shop, boss relic, Neow and removal. Otherwise inspect which panel encounters mis-rank options before widening |
| X3 | **Shaped OOC RL**: E8's setup (greedy frozen combat, macro credit, ≥768 games per update) plus potential shaping with the distilled `V_deck`, starting from the X2-distilled policy. Matched-compute unshaped arm, 2 seeds each. | ~1 day | Shaped arm beats unshaped by ≥ MDE at 1,024 worlds, with half-batch cosine ≥ 0.5 |
| X4 | **Combat expert iteration on run-ending fights**: learner-reached boss, elite and ≤50%-HP Act 2 starts, search ≥2,000 labels, combat-only optimiser. Held-out fight panels (as `boss_replay.py`) are the fast metric, then full games. | ~1 day, parallel with X2 | ≥ +8pp on held-out boss panels without hallway regression |
| X5 | Only after X2/X4 move boss pass rates: typed encoder and action-identity fix (Codex branch), longer end-to-end runs, a win-focused reward taper. | — | — |

**Do not do (on current evidence):** the proposed anchored LR-1e-5 OOC pilot;
any further 8-chunk/128-game pilots; larger models or a bigger GPU; switching
to SAC/DQN/world models. None of these addresses a gradient whose signal is ~1%
of its noise, or a reward channel that cannot see deck strength within a few
floors.

## Reproducing

From this worktree (`.venv` is a symlink to the frozen
`/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv`;
nothing is installed into it):

```bash
export STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build
ORIG=/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0030.pt
D=scripts/investigations/rl_debug_2026_10_13
PYTHONPATH=src .venv/bin/python $D/rollout_probe.py --checkpoint $ORIG --output /tmp/o.pkl --episodes 512 [--deterministic|--greedy-phase combat]
.venv/bin/python $D/analyze_probe.py /tmp/o.pkl
PYTHONPATH=src .venv/bin/python $D/grad_snr.py --checkpoint $ORIG --episodes 256 [--macro]
PYTHONPATH=src .venv/bin/python $D/boss_replay.py --probe /tmp/o.pkl --checkpoint $ORIG --floors 16 --n 120 --sims 200,2000 --output /tmp/b.json
PYTHONPATH=src .venv/bin/python $D/overlay_probe.py --checkpoint $ORIG --rules smith,nokey
PYTHONPATH=src .venv/bin/python $D/branch_probe.py --probe /tmp/greedy.pkl --checkpoint $ORIG --decisions 700
PYTHONPATH=src .venv/bin/python $D/ooc_macro_ppo.py --checkpoint $ORIG --out /tmp/run --iterations 8 --episodes 768
```

Probes need `rollout_probe.py --deterministic` output (with action traces) as
input for `boss_replay.py` (sampled works too) and `branch_probe.py` (greedy
only). TRAIN-eval worlds start at 5e9; `grad_snr.py` uses 6e9; `ooc_macro_ppo.py`
trains on 7e9+.
