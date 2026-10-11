# Native teacher budget screen

This CPU-only development screen compares native search budgets of 50, 200 and
1,000 simulations. Search is privileged: it sees hidden simulator state. Its
labels, seeds, task identities and latency measurements remain diagnostic data,
not learned policy inputs. Only search seed 1 is used for paired frontier queries;
native streams 0 and 1 alias on this build, so they are not independent votes.

All episode panels contain the exact same 48 battle prefixes from 21 development
worlds, selected with shuffle seed 777 and horizon 256. Initial public hashes,
OOC preview/RNG-commit prefixes, native/interface provenance and task bounds
match. Task identities intentionally differ because `battle_simulations` differs;
an independent scenario identity excludes only that budget and matches all 48
starts exactly. Reset validates the expected public hash before any battle action.

| Simulations | Wins | Mean HP cost, equal-world | Mean HP cost, episode | Faults / truncations |
| --- | ---: | ---: | ---: | ---: |
| 50 | 42 / 48 | 10.98 | 9.42 | 0 / 0 |
| 200 | 44 / 48 | 6.73 | 6.73 | 0 / 0 |
| 1,000 | 43 / 48 | 7.09 | 5.90 | 0 / 0 |

The 200-simulation episode panel is the already collected reference. Full-panel
wall time includes process startup, replay and observation/RPC overhead and is
recorded in `data/nonllm_v1/teacher_budget/summary.json`; it is not a pure search
microbenchmark. Differences in these 48 outcomes are exploratory evidence, not a
proof that one search budget is universally best. More simulations need not
monotonically improve the online action trajectory.

The paired query screen retains 64 public decision frontiers from 24 development
worlds. It includes decision offsets 0, 2, 4 and 6 along a fixed 200-simulation
reference path. Each frontier queries all three budgets on the same live state
with explicit search seed 1; native queries search a clone and do not step the
environment. Query order rotates to limit systematic cold-cache effects. Four
short fights ended before the desired offset; reserve starts supplied the 64
retained frontiers. The repeat records total qualified reserve queries explicitly.

| Budgets compared | Different selected labels |
| --- | ---: |
| 50 versus 200 | 15 / 64 (23.44%) |
| 200 versus 1,000 | 19 / 64 (29.69%) |
| 50 versus 1,000 | 24 / 64 (37.50%) |

| Simulations | Mean query + RPC | Median query + RPC | 95th percentile |
| --- | ---: | ---: | ---: |
| 50 | 0.369 ms | 0.334 ms | 0.759 ms |
| 200 | 1.060 ms | 0.952 ms | 1.905 ms |
| 1,000 | 4.813 ms | 4.224 ms | 8.862 ms |

A second run reproduced every retained frontier public hash and all 192 labels
exactly. Repeat mean latencies were 0.367, 1.077 and 4.793 ms. Both runs had zero
native faults. The frontier RPC measurements exclude reset/prefix replay time.
Labels can differ without proving that either label is better or uniquely right.

Keep 200 simulations as the current campaign teacher default: this bounded panel
shows better HP preservation than 50 while 1,000 costs roughly 4.5 times as much
per query and supplies no consistent improvement over 200: its episode-weighted
HP cost improves, but wins and equal-world HP cost worsen. A larger budget
can still be reconsidered on later difficult-state cohorts; final worlds remain
embargoed until policy selection freezes.

Reproduce the paired screen with `scripts/benchmark_teacher_budget.py --tasks
.../battles.jsonl --output REPORT.json --split dev --frontiers 64 --workers 4
--seed 777 --search-seed 1 --seconds 30`. Episode panels use the standard baseline
CLI with `--baseline native --simulations {50,200,1000}` and the matched settings
above. Raw panels, paired labels/latencies, repeat and summary reside in
`data/nonllm_v1/teacher_budget/`. Two synthetic regression tests verify paired
disagreement arithmetic and that scenario matching excludes only search budget.
