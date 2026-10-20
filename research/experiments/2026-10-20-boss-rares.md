# Boss rares: does taking a rare at boss card rewards help the champion?

**Status:** finished 20 October 2026. **Verdict:** taking Demon Form whenever
it is offered wins more games (179 against 156 of 4,096, p = 0.010); forcing
any rare at boss rewards helps the boss fight and the floor reached but not
clearly the win count.

## Question

The champion skips the card at 72% of Act 1 and 58% of Act 2 boss card
rewards, and takes Demon Form at only 29% of its offers
([power picks](2026-10-20-power-picks.md)). Are those skips costing games?
If forcing a rare, or Demon Form in particular, wins more games, the rule can
be distilled into the policy with the existing overlay, distil, PPO loop.
(Approach: [deck-strength oracle](../strategies.md#deck-strength-oracle),
deck-building.)

## Design

- **Arms,** all the champion checkpoint played greedily on the same 4,096
  second-DEV worlds (environment 0.1.5.1, seeds from 91,001,000,000) under
  the study protocol (`scripts/rl/fast_eval.py`, 8 processes of 32 games):
  1. champion, no overlay (the baseline for the other arms);
  2. `--no-skip-rares`: at any card reward whose cards are all rare (boss
     rewards, and a few elsewhere), the "skip card reward" and "proceed"
     actions are removed, so the champion takes the rare it ranks highest;
  3. `--take-card "Demon Form"`: whenever a card reward offers Demon Form
     (upgraded or not), only that card can be taken;
  4. both rules together.
- **What stays fixed:** the networks, the study's menu filters, the worlds
  and the evaluator. The overlays change only the menus offered to the
  policy.
- **Outcomes:** wins (the Act 3 boss beaten) as the primary outcome; mean
  floor and Act 2 boss pass rate as secondary.
- **Decision rule, fixed in advance:** an arm is *Better* than the champion
  if it wins more of the 4,096 paired worlds and the exact McNemar test on
  the discordant worlds gives two-sided p < 0.05; *Worse* on the same terms
  in the other direction; otherwise *No clear difference* if the discordant
  count could have resolved a 1.5-point change, else *Inconclusive*. Mean
  floor is reported with its paired bootstrap 95% interval but does not decide.
- **Evidence level:** Planned (rule fixed in advance, reused DEV worlds).
- **Compute:** about 5 minutes per arm on the laptop, beside a training run.
- **Approval:** Sid, 20 October 2026 ("please run that evaluation").

## Results

**Demon Form passes the rule; the other two arms win more games but not
clearly.** 4,096 worlds per arm, no faults or cuts; the champion arm
reproduces the study's 156 wins exactly.

| Arm | Wins (rate) | Discordant worlds, champion only / arm only | McNemar p | Mean floor, paired difference | Act 2 boss pass, paired difference | Verdict |
|---|---:|---:|---:|---|---|---|
| Champion | 156 (3.8%) | | | 35.08 | 45 to 46% | |
| Never skip a boss rare | 172 (4.2%) | 91 / 107 | 0.29 | +0.24 [+0.03, +0.47] | +3.1 points [+1.4, +4.8] (2,798 worlds) | No clear difference |
| Take Demon Form | 179 (4.4%) | 25 / 48 | 0.010 | +0.21 [+0.10, +0.32] | +2.4 points [+1.5, +3.2] (2,968 worlds) | Better |
| Both | 181 (4.4%) | 94 / 119 | 0.10 | +0.30 [+0.08, +0.53] | +3.8 points [+2.1, +5.5] (2,779 worlds) | No clear difference |

- The Demon Form rule touches few games (73 discordant worlds) and helps
  two of every three it changes. The boss-rare rule touches many (198
  discordant) with a smaller net gain: the champion's second choice among
  rares is a mixed bag.
- Every arm reaches further and passes the Act 2 boss more often; the boss
  pass gain is the clearest effect of all.
- A first pass of the same four arms through `scripts/rl/eval_policy.py`
  gave the same directions, weaker: wins 70, 74, 79 and 72 of 4,096 (McNemar
  p = 0.74, 0.11 and 0.91), floors +0.15 [−0.04, +0.35], +0.25 [+0.15,
  +0.35] and +0.20 [+0.00, +0.40]. That evaluator lets the policy drink and
  discard potions out of combat, which the study protocol hides, and the
  champion wastes them: on the 3,282 worlds the two cohorts share it reached
  3.3 floors less and won 57 games against 117. Hiding those actions
  reproduces the study's games exactly (128 of 128 worlds checked), with
  fast or slow observations. The protocols are now named in
  [metrics](../metrics.md#players-teachers-and-encoders).

## Conclusion

Declining Demon Form is a real cost: forced to take it, the champion wins
0.6 points more (3.8% to 4.4%) on the same worlds, passes the Act 2 boss
2.4 points more often, and reaches 0.2 floors further. This is the first
single deck-building habit shown to change the win count, and it supports
the reading that the out-of-combat head is locked into choices that a
better-informed policy would not make. Forcing any rare at boss rewards is
not the same lever: it helps the next boss fight but trades away enough
elsewhere that wins rise only within noise.

Caveats:
- DEV worlds reused: a distilled rule must be confirmed on the confirmation
  worlds before promotion.
- An overlay is not a learned behaviour; distillation and later PPO may not
  keep it.
- The batched evaluator at 32 games per process can flip exact ties between
  equivalent actions; all arms share the setting.
- The gain is specific to this combat head; whether Demon Form is worth more
  with a better combat player is being tested.

## Next

- Test the co-adaptation reading: the same overlays with the old-encoder
  combat student in place of the champion's combat head.
- Distil "take Demon Form" into the out-of-combat head
  (`collect_rule_labels.py`, `distill_ooc.py`), confirm on confirmation
  worlds, and continue PPO from it with the potion filter on.
- Extend the rule grid to the other declined powers (Corruption, Fire
  Breathing), card removals and "never skip any card reward".

## Log

- 20 Oct, 12:34: all four study-protocol arms finished; Demon Form passes
  the decision rule.
- 20 Oct, 12:04: the first pass (per-decision evaluator) showed the champion
  at 1.7%, not 3.8%; traced to visible out-of-combat potion actions and the
  evaluator's TRAIN-split world choice; rerun on the study evaluator with
  new overlay support.
- 20 Oct: the first draft of the `--no-skip-rares` flag was silently false
  (argparse reads a flag starting with "no-" as a negative form); fixed and
  checked on a smoke run before launch.

Code: `scripts/rl/fast_eval.py` and `scripts/rl/eval_policy.py` (overlays
in `src/sts_rl/common.py`), comparison in `experiments/2026-10-20-boss-rares/`.
Data: run `boss_rares_20261020` (local; `fast/` holds the study-protocol arms).
