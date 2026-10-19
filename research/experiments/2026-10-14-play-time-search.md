# Play-time search at bosses, elites and hallways

**Status:** finished 14 October 2026. **Verdict:** search at Act 2 and 3
bosses and elites improved full-game wins on fresh worlds; adding ordinary
hallway fights gave a larger gain on a smaller DEV panel, without fresh
confirmation.

## Question

Can the champion play better, without any training, if it lets the search
teacher choose its moves in hard fights? How much does [play-time
search](../strategies.md#play-time-search) help, and should it cover only
bosses, bosses plus elites, or ordinary hallway fights as well?

## Design

The canonical comparison changed where search played, keeping the champion's
combat and out-of-combat (OOC) weights fixed throughout.

- **Baseline:** round-2 OOC PPO update 20, the champion, playing greedily. The
  canonical comparisons used environment 0.1.4 and one observation per network
  inference in both learned-only play and search fallback.
- **Controller:** the search teacher runs 24 samples at each eligible
  decision. Each sample copies the battle, shuffles the copied draw pile and
  resets the battle's random-number generators (RNGs), covering enemy
  behaviour, cards, reshuffles, potions and monster HP. Run 1,000 native
  search simulations per sample. Execute the plurality choice: the action
  proposed by the most samples. Ties keep the first maximal entry in the vote
  list. If search abstains, use the learned network's highest-scoring action.
  Search chooses again at the next decision.
- **Canonicalization:** environment 0.1.4 sorts draw, discard and exhaust
  piles into a stable order before seeded sampling. Previously, a fixed set of
  samples could depend on the true hidden pile order. Returned discard and
  exhaust selections are mapped back to the corresponding live card instance.
  Sampling seeds now come from a hash of the public observation. The public
  observation version and ordinary game execution were unchanged by design.
- **Scopes:** search starts in Act 2. The three scopes cover bosses; bosses
  and elites; or bosses, elites and ordinary hallway fights. Act 1 and event
  fights remain learned-only.
- **Evaluation:** paired full games on 4,096 reused DEV worlds, with a
  512-world prefix for pilots, followed by 4,096 fresh confirmation worlds.
  Win rate was the headline metric and mean floor was secondary.
- **Decision rule and approval:** on 14 October, Sid authorized continued
  local experiments until stopped. The protocol specified paired full-game
  wins for selection, without a recorded numerical promotion threshold.
  Boss-plus- elite search was selected on DEV; its weights, settings and
  simulator were frozen before opening fresh confirmation, with no tuning from
  its outcomes.

## Results

Boss-plus-elite search beat learned-only play on fresh confirmation worlds.
All rows use environment 0.1.4 and completed every requested world without
simulator faults. Differences are paired against the named comparator.

| Worlds and evidence | Search scope | Search wins | Comparator wins | Mean floor difference [95% interval] |
|---|---|---:|---|---:|
| 4,096 fresh, Confirmed | Bosses and elites | 333 (8.13%) | Learned-only: 213 (5.20%) | +1.846 [+1.610, +2.072] |
| 4,096 DEV, Exploratory | Bosses only | 251 (6.13%) | Learned-only: 219 (5.35%) | +0.863 [+0.667, +1.048] |
| 4,096 DEV, Exploratory | Bosses and elites | 344 (8.40%) | Learned-only: 219 (5.35%) | +2.007 [+1.791, +2.229] |
| 512 DEV, Exploratory | Bosses, elites and hallways | 98 (19.14%) | Bosses and elites: 42 (8.20%) | +2.266 [+1.590, +3.002] |

The fresh comparison's exact paired win test gave p = 9.14 × 10⁻¹³. Adding
elites to boss-only search on the 4,096 DEV worlds gained another +1.144 mean
floor [+0.980, +1.320], with nominal paired win p = 8.27 × 10⁻¹². Boss-only's
win comparison against learned-only was weaker: nominal p = 0.0416.

The hallway gain also held against the other matched 512-world references:
learned-only won 29 games and boss-only search won 38. Against
boss-plus-elite, the hallway win comparison gave nominal paired p = 6.12 ×
10⁻¹¹.

Removing Act 3 boss search did not resolve its separate contribution. On 512
DEV worlds, searching only Act 2 bosses won 32 games versus 38 when searching
both acts' bosses: paired mean floor difference −0.012 [−0.029, +0.006],
paired win p = 0.263. Act 3 boss-entry HP matched on every world.

Earlier results pointed in the same direction under different conditions:

- **13 October, Exploratory:** the round-2 starting player, before its later
  PPO updates, used 64 samples of 1,000 simulations at Act 2 and 3 bosses. It
  won 69/2,019 paired worlds (3.42%) versus 53 (2.63%) without search; paired
  win p = 0.085, mean floor gain +0.78 [+0.54, +1.02]. Progression improved,
  but the full-game win result was inconclusive.
- **Environment 0.1.3, Exploratory:** the champion with the old 64-sample boss
  search won 267/4,096 DEV worlds (6.52%) versus 220 (5.37%) learned-only;
  paired mean floor gain +0.829 [+0.645, +1.022], paired win p = 0.002731.
  This is a separate earlier condition, superseded for canonical comparisons.

## Conclusion

Search established a stronger simulator-assisted player without training. The
boss-plus-elite gain survived fresh confirmation; hallways remain promising.

The boss-only gain chiefly came from getting more worlds through Act 2.
Neither the timing ablation nor the full-game comparisons establish a separate
Act 3 boss-search benefit. They also do not establish that the hallway pilot's
much higher win rate generalizes to fresh worlds.

Caveats:

- Search is privileged: it operates on native simulator copies. Canonical
  resampling fixes hidden-pile-order dependence, but other latent native
  fields, history and ties between publicly identical cards remain uncertified
  as functions of public information alone.
- DEV worlds were reused across several budget and scope choices. Their
  evidence is exploratory and their reported p-values do not adjust for
  selection; only the frozen boss-plus-elite candidate was freshly confirmed.
- The old 0.1.3 comparison also mixed batched learned inference with single-
  observation search fallback. Its 220-win baseline must stay separate from
  the canonical 219-win baseline. Releases and protocols are not
  interchangeable.
- The 13 October source does not name its environment release. It reports 1.3%
  simulator faults over 2,048 requested games, with 2,019 paired worlds in the
  comparison; those faults are not evidence of game defeats.
- Sources supply paired mean-floor intervals and paired win tests, but no
  intervals for win-rate differences. The extra compute cost of hallway search
  was not measured under controlled conditions.

## Next

Freeze the broader hallway condition and confirm it on fresh worlds before
adopting that scope. Measure its additional compute cost before extending it
or declaring a cheaper search budget. Keep learned-only and search-assisted
results separate; using search as a training teacher is a [different
study](2026-10-14-overnight-tests.md).

## Log

- 13 Oct: boss-only play-time search improved progression; the win result
  remained inconclusive.
- 14 Oct: fixed-seed dependence on hidden pile order was found; canonical
  sampling adopted in environment 0.1.4.
- 14 Oct: inference batching changed some trajectories; canonical comparisons
  switched to matched single-observation inference.
- 14 Oct: larger DEV comparisons selected boss-plus-elite search; the
  candidate was frozen before fresh confirmation.
- 14 Oct: fresh confirmation established the boss-plus-elite gain; the hallway
  pilot remained exploratory.
- 14 Oct: Sid requested a report before further training; no broader search
  extension followed.

Code: `src/sts_rl/panel.py` and `scripts/rl/eval_search_boss.py` (before the
experiment-folder convention), canonical source snapshot at commit `d5c56e1`.
Data: run `canonical_20261014` (local), with outputs in `evals/`,
`confirmation/` and `exploration/`; earlier run `searcheval_r2_a23_s64`
(local).
