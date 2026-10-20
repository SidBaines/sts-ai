# Boss rares: does taking a rare at boss card rewards help the champion?

**Status:** running. **Verdict:** pending.

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
  second-DEV worlds (environment 0.1.5.1, seeds from 91,001,000,000),
  through `scripts/rl/eval_policy.py` with 12 in-process collectors of 32
  games:
  1. champion, no overlay (the baseline for the other arms);
  2. `--no-skip-rares`: at any card reward whose cards are all rare (boss
     rewards, and a few elsewhere), the "skip card reward" and "proceed"
     actions are removed, so the champion takes the rare it ranks highest;
  3. `--take-card "Demon Form"`: whenever a card reward offers Demon Form
     (upgraded or not), only that card can be taken;
  4. both rules together.
- **What stays fixed:** the networks, the dead-action filter, the worlds and
  the evaluator. The overlays change only the menus offered to the policy.
- **Outcomes:** wins (the Act 3 boss beaten) as the primary outcome; mean
  floor and Act 2 boss pass rate as secondary.
- **Decision rule, fixed in advance:** an arm is *Better* than the champion
  if it wins more of the 4,096 paired worlds and the exact McNemar test on
  the discordant worlds gives two-sided p < 0.05; *Worse* on the same terms
  in the other direction; otherwise *No clear difference* if the discordant
  count could have resolved a 1.5-point change, else *Inconclusive*. Mean
  floor is reported with its paired bootstrap 95% interval but does not decide.
- **Evidence level:** Planned (rule fixed in advance, reused DEV worlds).
- **Compute:** about 3 minutes per arm on the laptop.
- **Approval:** Sid, 20 October 2026 ("please run that evaluation").

## Results

Pending.

## Conclusion

Pending.

## Next

Pending.

## Log

- 20 Oct, 12:04: all four arms finished (about 3 minutes each). Surprise:
  the champion arm wins 70 of 4,096 (1.7%), not the 3.8% of the study's
  recorded games. `eval_policy.py` picks TRAIN-split seeds, so only 3,282
  worlds are shared with the study, and on those it reaches 3.3 floors less
  and wins 57 against 117. The evaluators differ in protocol (the study hides
  out-of-combat potion actions and uses slow observations); diagnosing which
  setting matters before drawing conclusions.
- 20 Oct: the first draft of the `--no-skip-rares` flag was silently false
  (argparse reads a flag starting with "no-" as a negative form); fixed and
  checked on a smoke run before launch.

Code: `scripts/rl/eval_policy.py` (overlays in `src/sts_rl/common.py`).
Data: run `boss_rares_20261020` (local).
