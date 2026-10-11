# Expanded development battle references

The expanded later-act panel contains 32 available battle starts from 30
development worlds. `--episodes 256` requests up to 256; it does not make this a
256-battle evaluation. Starts preserve the collected public hash, exact action
prefix and OOC preview/RNG-commit history, ascension and native/interface
provenance. Search uses 200 simulations, task horizon 256, act bound 3 and shuffle
seed 777. All six panels below requested and executed the same exact task ID set.
No final worlds or learner GPU jobs were executed for the reference comparison.

| Policy | Act 1 development wins | Later-act development wins | Later-act world-mean success | World-bootstrap 95% interval | Later-act world-mean HP cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform random | separate panel | 13 / 32 | 38.33% | 21.67–56.67% | 51.63 |
| Public heuristic | separate panel | 28 / 32 | 88.33% | 76.67–98.33% | 21.48 |
| Privileged native | separate panel | 30 / 32 | 93.33% | 83.33–100% | 8.80 |
| PPO pilot, learning rate 0.0003 | 59 / 64 | 24 / 32 | 78.33% | 63.33–91.67% | 24.48 |
| PPO pilot, gamma 0.99 | 58 / 64 | 26 / 32 | 81.67% | 66.67–93.33% | 21.45 |
| PPO pilot, clip 0.1 | 55 / 64 | 29 / 32 | 91.67% | 81.67–100% | 19.82 |

All later-act panels had zero simulator faults and zero truncations. Native
combat search is privileged and is a reference, not an optimal public policy.
These are development comparisons during selection, not embargoed final tests.
The Act 1 column shows the previously executed learned pilot panel; references
in the later-act column are not inferred from the earlier Act 1 reference panel.

Paired comparisons resample entire worlds and average each world equally. For
clip 0.1 versus native, success differs by -1.67 percentage points with 95%
interval [-11.67, +8.33]. HP cost is 11.02 HP higher, interval [+6.52, +16.40].
Versus the public heuristic, success differs by +3.33 points, interval [-6.67,
+13.33]; HP cost differs by -1.67 HP, interval [-7.10, +3.27]. This small panel
supports a close win-rate point estimate to native, while showing a substantial
HP preservation gap. It does not establish an advantage over the heuristic.

The reusable `report_nonllm.paired_world_comparison` matches exact task IDs and
rejects missing/duplicate identities. It excludes every world faulted by either
policy and excludes unresolved/truncated tasks, recording those exclusions.
Faults never become losses. This panel needed no exclusions. The paired delta
uses the same clean tasks for both policies, preserving within-world dependence.

Artifacts are `data/nonllm_v1/baselines/expanded_hard_dev_{native,heuristic,random}.json`
and `expanded_hard_dev_comparison.json`, including world intervals, paired deltas
and the exact requested identity set. Reference commands use
`scripts/evaluate_nonllm_baselines.py --tasks
data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --kind battle --split dev
--episodes 256 --seed 777 --horizon 256 --max-act 3 --simulations 200 --workers 4
--fast-observations --fast-replay --baseline POLICY --seconds 120 --output PATH`.
Fast replay is the already validated prefix execution path used by the learned
panels; it skips intermediate renderings and validates the same public start.

Regression coverage includes exact-task requirements, duplicate rejection,
whole-fault-world exclusions, paired differences, world resampling, censoring and
baseline selection. The 12 report/baseline/teacher-budget tests passed after the
paired helper and optional baseline fast-replay flag were added.
