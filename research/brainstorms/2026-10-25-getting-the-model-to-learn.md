# Getting the 0.9B model to learn

**25 October 2026.** The planned pilot is likely to end "inconclusive" for
reasons unrelated to whether RL can teach this model. It starts at
trivial-baseline skill, most of its combat reward is noise between near-equal
actions, it gets about 150 optimizer steps at a low LoRA learning rate, and it
has no early metric that separates learning from noise. We propose a staged
plan instead. Each stage is a quick test that can fail, and the first stages
use only out-of-combat decisions, where the teacher is decisive. The
decisions needed from Sid are listed at the end.

## What we found

All figures come from the 64-state timing panel (40 combat, 24 out of combat,
55 worlds), with one sample per state at temperature 0.6 unless stated.
Treat them as exploratory: the panel is small and the intervals are wide.

The RTX runs at different concurrencies and backends reused the same
per-request seeds, so each effort level there is close to a single sample.
For example, the four RTX low-effort runs choose the same action on 63 of 64
states, but a Mac low-effort run with fresh randomness agrees with them on
only 35. The ranges below therefore include the Mac runs from
[the Mac study](../experiments/2026-10-25-mac-local-speed.md) as independent
samples.

**The model plays at trivial-baseline level.** Mean teacher probability of the chosen action:

| Policy | Combat | Out of combat | All |
|---|---:|---:|---:|
| Random legal action | 0.20 | 0.24 | 0.21 |
| Always the first menu option | 0.27 | 0.34 | 0.30 |
| Model, low effort (most answers have empty reasoning) | — | — | 0.31–0.35 |
| Model, medium effort | 0.24–0.27 (RTX) | 0.33 (RTX) | 0.26–0.33 |
| Model, high effort | 0.23–0.26 (RTX) | 0.23–0.26 (RTX) | 0.22–0.27 |
| The teacher's own top choice | 0.36 | 0.85 | 0.54 |

On the 25 states where the teacher is decisive (top action at least 0.3 ahead
of the next), matching the teacher's top action happens:
- 44–56% of the time at low effort (two independent samples);
- 36–48% at medium;
- 28–36% at high;
- 36% for "always first";
- 29% for random.

**Reasoning does not help, and high effort hurts.** Low and medium effort are
within sampling noise of each other. The reasoning reads as generic task
framing rather than play:
- "The user gave a menu… Probably choose x=4. I'd pick index 0."
- On a rest site at 23/80 HP, the model weighs options without using the HP.

The model picks the first option in 45–62% of valid answers. The teacher's own
top choice is the first option 44% of the time (combat 48%, out of combat 40%),
so position bias is real but only moderate.

**The reward is decisive out of combat and mostly noise in combat.**
- In the 128 prepared states, the teacher's top probability averages 0.88 out
  of combat (margin over the second choice 0.78; 42 of 48 decisive).
- In combat it averages 0.37 (margin 0.13; 10 of 80 decisive).
- Across four medium samples of the same state, the best and worst sampled
  rewards differ by a median of only 0.06–0.08.
- About a quarter of four-sample groups had identical rewards: 9–12 of 24 out
  of combat, 6 of 40 in combat.
- The trainer divides each group's rewards by their standard deviation. A
  group scoring {0.30, 0.30, 0.30, 0.31} therefore gets the same size of
  update as {0, 0, 0, 1}, so in combat the gradient mostly follows near-ties.

**The prompt is long and easy to misread for a 0.9B model.**
- Combat prompts average about 6,000 characters, and a quarter of that is
  rules and caveat text.
- Out-of-combat prompts average about 3,300 characters, half of which is the
  full act map, even at a rest site.
- Hand cards, enemies, potions and menu actions are all labelled `[0]`, `[1]`,
  and so on. Hand `[0]` is often not menu `[0]`.
- An environment bug: the out-of-combat text lists potions by potion count
  rather than slot. It shows the wrong potions whenever an empty slot comes
  before a filled one: 7 of 48 out-of-combat prompts disagree with their own
  menu (`slaythespire.cpp` around line 849).

**Too few updates.**
- At the RTX rates measured on 23 October, the seven-hour pilot gives about
  140–170 optimizer steps covering about 600 states.
- A learning rate of 0.00001 is at the low end for rank-16 LoRA, which
  usually wants several times the full fine-tuning rate.
- The teacher is worth imitating: the non-LLM champion wins 27–29% of games,
  and the model wins none.

## Ideas considered

1. **Prove the loop can learn before asking whether it generalises.**
   - Train on 32 fixed, decisive out-of-combat states and check that reward on
     those same states climbs towards the teacher's top choice within about 50
     steps.
   - If it cannot even overfit, something is wrong with the learning rate, the
     loss or the parsing, and no held-out result would be meaningful.
   - Recommended first.
2. **Make the reward measure decisions, not noise.**
   - Centre advantages on the group mean without dividing by the standard
     deviation, so near-ties give near-zero updates.
   - Start with decisive states and add diffuse ones later.
   - For combat, label states offline with the stronger search teacher rather
     than the champion's diffuse combat network. The non-LLM workstream already
     collects search labels, and offline labelling with privileged search is
     allowed for teachers.
   - Recommended.
3. **Remove shortcuts in the answer format.**
   - Give menu actions their own labels (letters, say) or have the model answer
     with the action text. Sid's earlier semantic-target study raised top-1
     agreement from 0.51 to 0.61 for supervised training.
   - Shuffle the menu order per sample, so the reward can only be earned
     through content.
   - Trim the caveat text and show only the reachable map.
   - Fix the potion line on `main` first.
   - Changing the prompt changes the experiment, so do it before the baseline,
     with an A/B on the unchanged model. Recommended; needs Sid's decision.
4. **Treat reasoning length as an outcome, not a setting.**
   - With teacher agreement as the reward, the model may learn to shorten its
     reasoning, since its near-empty low-effort answers score no worse. That
     is itself a result about reasoning under RL.
   - To tell "reasoning helped" from "reasoning was dropped", train a
     low-effort arm with identical settings.
   - Recommended as a cheap second arm once stage 2 works.
5. **Give the optimiser a chance.**
   - Screen learning rates 0.00002, 0.00005 and 0.0001 on the overfit set for
     about 30 steps each.
   - Use 8–16 states per step rather than 4.
   - Set the KL coefficient low at first, and watch KL rather than fixing it
     blind.
6. **Bridge to full games cheaply.** The environment's `ooc` mode lets the LLM
   make only out-of-combat choices while native combat plays the fights. That
   is about 30–60 model decisions per game instead of hundreds, and it tests
   whether better out-of-combat choices raise floors reached and wins.
7. **Rejected for now:**
   - Full-game terminal-outcome RL: almost every group would be dropped
     today, and HF generation takes about an hour per game.
   - Separate phase adapters: they add a variable before any learning signal
     exists.

## Early-signal metrics

Evaluate a fixed probe set every 10–20 steps: 128 validation states, half
decisive out of combat and half combat, four samples each at the training
temperature.

| Metric | What it shows | Learning looks like |
|---|---|---|
| Top-1 agreement on decisive states | The cleanest signal: binary, and the teacher is reliable there | Clearly above the base model's 0.36–0.56, on held-out states |
| Normalised teacher score: (p(chosen) − mean legal p) / (p(top) − mean legal p) | 0 for random, 1 for the teacher's choice; comparable across phases | Rising on held-out states, not only on training states |
| First-option rate against the teacher's own 44% | Position shortcut | Moves towards 44%, not towards 0 or 100% |
| Same state with a shuffled menu: same action content chosen | Whether choices depend on content | Consistency rising |
| Validity, cap rate, reasoning tokens (median, 95th percentile, share empty) | Format health and the reasoning outcome itself | Validity ≥ 99%; length trend reported, not targeted |
| Training-set versus probe gap | Memorising versus generalising | A gap that stays bounded |
| Per group: zero-variance share, reward range, distinct actions | Whether the batch carries signal | Zero-variance share well below 50% |
| Per step: gradient norm before clipping, KL, clip fraction | Optimiser health | Finite; KL growing slowly; clip fraction near 0 with one step per batch |

How much data this needs: with about 200 decisive held-out states, a 7-point
change in top-1 agreement is about two standard errors, and a paired
comparison on the same states is tighter. The current pilot's +0.02
mean-probability rule on all states is much harder to detect, because combat
near-ties add variance.

## Proposed stages

0. **Prompt A/B on the unchanged model**, run on the Mac in MLX.
   - Arms: the current prompt, and a trimmed and relabelled prompt with a
     shuffled menu.
   - Measure: decisive top-1 agreement, normalised score and first-option rate,
     on the 64-state panel with four samples each.
   - Choose the prompt before any training.
1. **Overfit test and learning-rate screen** on 32 decisive out-of-combat
   training states. Gate: reward on those states rises clearly within about 50
   steps at some learning rate.
2. **Out-of-combat pilot.**
   - Train on decisive out-of-combat states from training worlds, with
     mean-centred advantages.
   - Evaluate on the probe set and on held-out out-of-combat states.
   - Gate: held-out decisive top-1 agreement improves beyond its interval.
3. **Add combat** with search-teacher labels, still with decisive states
   first, then run the low-effort comparison arm.
4. **Out-of-combat-only games** on a fixed paired world panel, comparing the
   base and trained adapters: floors reached, wins and model failures,
   reported separately.

## Decisions for Sid

- May the prompt and answer format change before the pilot baseline (stage 0)?
- Start with out-of-combat decisions only (stages 1–2)?
- Replace standard-deviation scaling with mean-centring, and start with
  decisive states?
- Hardware: the laptop can run stages 0–2 under MLX, at an estimated 95 s per
  iteration (about 38 iterations an hour) with grouped generation and
  checkpointed updates ([Mac study](../experiments/2026-10-25-mac-local-speed.md)).
  A rented GPU with vLLM serving is only needed for full-game panels or much
  larger runs. Is the laptop acceptable for stages 0–2?
