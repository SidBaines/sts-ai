# Frozen final evaluation — 2026-10-12

The selected public-state non-LLM policies cleared **20/21 Act 1 full-control worlds** and **17/21 Act 1 worlds with native combat support**. They won **145/160 isolated Act 1 battles** and **25/32 later-act battles**. Neither the selected joint policy nor the original specialist composite completed a three-act game. Act 1 competence is supported; whole-game competence is not established.

All **22 declared final panels** completed their requested cohorts and passed the committed report's independent readiness gate: exact journal metrics SHA, completed command status, frozen task identities, current command/input signature, selected checkpoint path/SHA, executed native manifest/module and Python-interface evidence. No panel was missing or stale. The freeze was verified before and after reporting, with unchanged digest `adaf0347546d50e8d1ddb1b58e3cf1663c84f226ad1c4dd0145f22682cf1966c`. No policies were executed during report generation, and no policy/core/script files were modified.

## Matched requested cohorts

Cells are **wins / defeats / faults / truncated**. Faults remain separate from defeats. Every observed truncation count was zero. The random and heuristic references use public observations; native search is privileged, with scripted native out-of-combat choices.

| Frozen task cohort | Requested tasks / worlds | Selected learned policy | Native reference | Public heuristic | Uniform random |
| --- | ---: | --- | --- | --- | --- |
| Isolated Act 1 combat | 160 / 20 | 145 / 15 / 0 / 0 | 149 / 11 / 0 / 0 | 141 / 19 / 0 / 0 | 93 / 67 / 0 / 0 |
| Isolated Acts 2–3 combat | 32 / 25 | 25 / 7 / 0 / 0 | 26 / 6 / 0 / 0 | 20 / 12 / 0 / 0 | 10 / 22 / 0 / 0 |
| Act 1 OOC policy, native combat | 21 / 21 | 17 / 4 / 0 / 0 | 6 / 14 / 1 / 0 | 13 / 8 / 0 / 0 | 1 / 20 / 0 / 0 |
| Act 1 full control | 21 / 21 | 20 / 1 / 0 / 0 | 6 / 14 / 1 / 0 | 4 / 17 / 0 / 0 | 0 / 21 / 0 / 0 |
| Three-act full control | 21 / 21 | 0 / 20 / 1 / 0 | 0 / 20 / 1 / 0 | 0 / 21 / 0 / 0 | 0 / 21 / 0 / 0 |

The specialist composite separately recorded **0 wins / 21 defeats / 0 faults / 0 truncated** on the same three-act requested cohort. Its source heads were the frozen combat and OOC selections. The selected joint policy's mean reached floor was 30.05 across its 20 nonfault episodes, versus 20.38 across the composite's 21 and 18.40 across native's 20. Those descriptive means exclude different faulted worlds; they do not replace the zero-win result or constitute a paired progress estimate.

Later-act combat was evenly split: learned **13/16 Act 2** and **12/16 Act 3** wins, versus native **13/16** and **13/16**. These are replayed isolated fights, not successful runs to those acts. `final_battle_act1` and `final_combat_extended_act1` are exact duplicate policy/cohort panels (145/160 each); the table counts this evidence once.

## World-level descriptive uncertainty

Counts and requested world N are primary. Below, differences are learned minus reference, with equal weight per common valid world and a 2,000-replicate world-cluster bootstrap. A world faulted by either compared policy is excluded from the paired estimate and explicitly retained in the outcome table. Intervals are descriptive, not multiplicity-adjusted significance claims.

| Paired comparison | Common valid worlds | Success difference, percentage points | Bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Act 1 combat versus native | 20 | −2.3 | [−6.2, +0.7] |
| Later-act combat versus native | 25 | −2.0 | [−14.0, +10.0] |
| Act 1 OOC versus native | 20 | +50.0 | [+30.0, +70.0] |
| Act 1 OOC versus heuristic | 21 | +19.0 | [−4.8, +42.9] |
| Act 1 full control versus native | 20 | +65.0 | [+45.0, +85.0] |
| Act 1 full control versus heuristic | 21 | +76.2 | [+57.1, +90.6] |

The isolated-battle raw success fraction and equal-world mean differ because worlds contribute different numbers of fights. The complete JSON includes both counts and world estimates. For three-act comparisons, all observed valid outcomes were failures: their empirical bootstrap success intervals collapse to zero. This is a property of the small sample, not proof of zero uncertainty, equivalence between policies, or competence.

Combat horizons were 256 actions. Generated final OOC and run horizons were 2,048; development OOC used 1,024 and development Act 3 used 4,096. Development/final differences therefore cannot be attributed solely to generalization. Endpoint HP across act transitions includes automatic healing; it is not evidence of run-long HP preservation. Native faults occurred on world 10240 in OOC/Act 1 run/full run; the selected joint three-act policy faulted on world 10239. No fault was counted as an ordinary loss.

## Frozen selected checkpoint lineage

All current checkpoint bytes matched their frozen SHA256 identities:

| Role | Checkpoint | SHA256 |
| --- | --- | --- |
| shared | `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0030.pt` | `57f1e9613a69e31230b591ef0ff2cca27e23ddc81c3a24366cee173bc9909e70` |
| combat | `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt` | `52c557a0c73921c0c39aea2a744054d731794bb5e2a55d93b05421702312d85b` |
| ooc | `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt` | `67230041c216ae74c00250580df1cab74d7d2c6d47a73355080b182e9c3f9ad1` |

The selected shared checkpoint is the final development-selected joint Act 3 policy. Its producing-condition parent inputs are recorded in `selected_checkpoint_lineage`; nested composite source ancestry remains available through the campaign's immutable composition manifests. Legacy ancestry without native provenance is not invented.

Machine-readable evidence: [`campaign_report.json`](../data/nonllm_native_revision/final_report/campaign_report.json), [`campaign_panels.csv`](../data/nonllm_native_revision/final_report/campaign_panels.csv), and the latest-first [`campaign_report.md`](../data/nonllm_native_revision/final_report/campaign_report.md). The development-only curve is [`native011_act1_development.png`](../data/nonllm_native_revision/report/native011_act1_development.png). The final report CLI accepts `--png`, but its current final-stage outputs are JSON/CSV/Markdown; it does not render a final PNG.

Reproduce this file-based report without executing a model:

```sh
PYTHONPATH=src .venv/bin/python scripts/report_nonllm_campaign.py --stage final --allow-final-results --campaign data/nonllm_native_revision/campaign --output data/nonllm_native_revision/final_report
```
