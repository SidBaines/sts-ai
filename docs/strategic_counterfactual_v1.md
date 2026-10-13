# Complete-menu strategic continuation diagnostic

13 October 2026, after the overnight allocation. This bounded follow-up tests
whether a single alternative OOC choice improves actual continuation outcomes.
It creates diagnostic panels only; it does not supply calibrated action values
or training targets. Source worlds are TRAIN, with no DEV/final gradient use.

`scripts/probe_strategic_counterfactual.py` replays a provenance/hash-pinned
strategic start, evaluates the unforced original policy, then tries **every legal
first action** under the same frozen public OOC and legacy combat policy. One
worker/task at a time preserves inference batch shape. Forcing the original
first action must reproduce the whole control trace, reward and endpoint.
Faults, horizon cuts and missing branches invalidate a complete menu; they are
not defeats and cannot be omitted to choose a winner. Interventions are blocked
from producing on-policy PPO rows. The collector is v5; default sampling remains
v1. Its default trajectory semantics are unchanged.

The first version uses one fixed native world. Optional policy sampling seeds
vary continuation choices, **not hidden futures or independent worlds**. It does
not establish which action a public agent should choose under uncertainty.

## First complete panel

The full compatible 49-frontier bank finishes in **43.37 seconds**, one worker:

| Measure | Result |
| --- | ---: |
| Source menus | 49: 40 Act 2, nine Act 3 |
| Evaluated legal alternatives | 140 |
| Valid completed branches | 140/140 |
| Control trace/reward/endpoint matches | 49/49 |
| Faults, cuts, unstarted sources | 0 |
| Reference conditional victories | 3 |
| Menus with an alternative full-win gain | 0 |
| Menus with an alternative floor gain | 3 |
| Menus tied on all recorded paired outcomes | 33 |

These are continuations from native-generated decks, including correlated
different-act starts from some worlds; they are not fresh-game win rates.
Three alternative actions lose an existing conditional victory. Three reward
menus allow greater floor reach, including skipping a card in world 390000051
for +11 floors, but none converts a defeat to a victory. The four-source panel
repeats in a fresh process with **16/16** complete baseline/branch trace, reward
and endpoint identities.

The immediate implication is narrower than "OOC choices do not matter": this
late start bank and frozen continuation provide little new full-win signal from
a single changed decision. A useful next teacher needs earlier/broader strategic
choices and/or a stronger continuation policy. Full-run outcome ties also hide
differences in intermediate HP and combat quality. No new labels are accepted
from these panels.

Evidence is under `data/nonllm_v4/counterfactual/ooc_menu_greedy_all49`, with
per-panel source observations, complete branch outcomes and source/checkpoint
identities in the manifest. The four-source runs are `ooc_menu_greedy_panel4`
and `ooc_menu_greedy_panel4_repeat`. Current execution acceptance: **546
host/native tests**, one optional artifact skip, plus bounded smoke without
error. The new unit cases verify executed interventions, preserved RNG/control,
whole-menu invalidation, forbidden on-policy rows and split/role boundaries.
