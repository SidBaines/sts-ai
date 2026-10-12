# Evaluation helpers and safeguards

`scripts/evaluate_nonllm_baselines.py` supplies uniform random, a deliberately
simple public heuristic, and a privileged native reference. Native combat
search sees hidden state; native OOC is scripted. These are different reference
policies, not a claim that native search is optimal.

Run matched development battles against the collected bank:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_nonllm_baselines.py \
  --tasks data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl \
  --kind battle --split dev --episodes 32 --workers 4 --simulations 200 \
  --horizon 256 --seed 0 --seconds 30 --fast-observations \
  --baseline random --output data/nonllm_v1/baselines/battle_random_dev32.json
```

Repeat with `--baseline native` and `--baseline heuristic`. Match the learned
evaluation's bank, split, count, shuffle seed, horizon, simulations and act scope.
Every requested task has a stable `task_id` over its exact TaskSpec. OOC/run
evaluation deduplicates worlds if supplied a bank containing many fights per
world; battle evaluation retains the distinct prefixes. Evaluation CLI task
selection defaults to development; final worlds remain embargoed during model
selection. No final worlds were executed for the smoke evidence below.

`scripts/report_nonllm.py METRICS.json ... --output REPORT.json` reports BC
training/development metrics and live battle/OOC/run results. Faults, true losses,
truncations and unresolved terminal outcomes are counted separately. Reported
success ratios exclude faults and censoring; they are conditional on decisive
completion and can be biased if censoring differs, so always retain the counts.
`act_clear` is distinct from full-game `victory` in the outcome counts.

The uncertainty helper resamples whole worlds and gives each world equal weight.
This preserves the grouping of many fights/states from one world and prevents
worlds with more sampled fights dominating the world-level estimate. It reports
95% percentile bootstrap intervals only with at least two worlds. Episode-level
success percentages are also shown as a separate estimand. Aggregate BC accuracy
alone cannot establish world-cluster uncertainty; optional per-record
`bc_records` must contain `world_seed`, `phase`, `correct`, and `loss`. Training
rounds with different policies receive separate summaries, with an explicit note
that a pooled curve is not a fixed-checkpoint evaluation.

The 32-battle development smoke on the exact same tasks produced:

| Reference | Battle wins | Equal-world success | World-bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| Uniform random | 14 / 32 | 42.65% | 23.53–62.75% |
| Public tactical heuristic | 27 / 32 | 79.41% | 58.82–95.59% |
| Privileged native, 200 simulations | 29 / 32 | 86.27% | 68.63–100% |

There were 17 distinct worlds, zero faults and zero truncations for each policy.
These small development results establish reference execution and reporting,
not final model competence. Artifacts are in `data/nonllm_v1/baselines/`.

The tactical heuristic scores publicly shown damage/block, immediate lethal
targets, draw/energy, and basic OOC health/card/path choices; it has no learned
parameters. Its weakness is useful context when judging learned policies rather
than only comparing them with uniform random.

Synthetic regression tests cover world weighting, repeatable intervals, missing
world identity, fault/censoring separation, unknown terminal outcomes, BC interval
limits, bank task deduplication, and deterministic matching. Actual bounded native
baseline runs verify execution of all three policies. The broader campaign must
still compare learned checkpoints using the same task selections and reserve
final evaluation until after candidate selection freezes.

The campaign's screening panel uses 48 development battles, shuffle seed 777,
horizon 256 and 200 native simulations. Matched reference artifacts are
`data/nonllm_v1/baselines/dev48_{random,heuristic,native}.json`: random won 28/48,
the public heuristic 42/48, and native search 44/48. Each panel covers 21 worlds,
with zero faults or truncations. Equal-world success means are 56.67%, 86.90%
and 93.25%; their clustered 95% intervals are 40.47–72.54%, 75–96.43% and
86.89–98.81%. Task identity sets were independently regenerated from the actual
learned evaluation CLI configurations and matched exactly; actual episode
world/floor multisets matched too. Order can differ because the training CLI
reshuffles the selected panel before collection. Match the set of exact task
identities, not completion order.

### OOC control and HP interpretation audit (2026-10-12)

OOC TaskSpecs use environment mode `ooc`, `combat_control=search`: the policy
never receives combat action menus. CLI simulations 200 means ordinary native
search 200, with the released default boss multiplier 2 (400); a per-action
teacher query of 200 is a different operation. Keep this distinction in reference
comparisons. The immutable backend has not changed.

`hp_cost` is **net starting minus ending HP**, not cumulative incoming damage.
For OOC/full runs, act-boundary healing can erase prior HP losses; successful
Act 1 clears may therefore all show zero net cost. Terminal HP shaping can also
be constant at these healed endpoints. Use actual clears, survival/floor outcomes
and explicit within-battle measurements; do not infer OOC HP preservation from
this field alone.

Evaluation now uses its declared collection budget rather than the 120-second
PPO-round cap. Development selection requires exact requested task coverage and
uses wins/requested tasks for its conservative selection score. Simulator faults
remain separate outcomes, never fabricated defeats or training transitions.
