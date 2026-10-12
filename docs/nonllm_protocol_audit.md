# Final-cohort protocol audit

Audit snapshot: 2026-10-12 02:54:58 UTC, before final selection freeze. This was a read-only audit of saved command journals, non-final metrics, diagnostic traces and task definitions. No learned final evaluation was executed and no final policy results were read. Evidence is saved in [nonllm_protocol_audit_evidence.json](nonllm_protocol_audit_evidence.json).

## Diagnostic deviation

`world_split(3)` is `final`. Two extra deployment/reproducibility checks ran the learned BC agent on world 3 before freezing model selection:

- `data/nonllm_v1/verification/agent_plugin_seed3_process1.jsonl`
- `data/nonllm_v1/verification/agent_plugin_seed3_process2.jsonl`

Both headers identify `sts_policy.model_agent:make_agent`, checkpoint `data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt`, seed/policy seed 3, full control, max act 1 and native simulations 50. Each has 459 JSONL rows, 228 external actions and zero native automatic steps, ending in ordinary `GameOutcome.PLAYER_LOSS` / `defeat`. The recorded purpose and interpretation in `docs/nonllm_progress_notes.md` are plugin integration and cross-process determinism; they do not establish winning competence.

These learned seed-3 checks were extra diagnostics. The repository instruction requiring a bounded random-agent seed-3 smoke explains the habitual fixture but did not require a learned-agent run on that world. The preserved corrected-native random smoke, `data/nonllm_curriculum/profiles/native011_validation/native011_seed3_smoke.jsonl`, also uses seed 3 and ends in ordinary defeat; it is a random-agent backend validation, not learned-policy evaluation.

The entire hash-defined final partition therefore was **not untouched**. This exception must remain visible in reporting. Seed 3 is outside the designated quantitative final cohort below. It is excluded from future final inference, model selection, training and competence claims; do not change the hash split retroactively. Future learned-agent deployment fixtures must use an explicitly verified train/dev world. No saved campaign selection field or aggregate metric references either learned seed-3 trace. The recorded use is a qualitative integration check, not hyperparameter selection. The audit cannot reconstruct unlogged commands or private reasoning, so this is an evidence-based statement about the persisted experiment record.

## Designated final cohorts

The existing `Campaign.freeze_final_selection` protocol generates the first 21 hash-final worlds at or above 10000:

`10018, 10021, 10057, 10101, 10105, 10127, 10141, 10153, 10155, 10169, 10196, 10205, 10211, 10213, 10220, 10235, 10236, 10239, 10240, 10245, 10247`.

The three generated panels are OOC Act 1, full-control Act 1 and full-control Act 3. Each uses horizon 2048, 200 simulations, shuffle seed 777 and 21 episodes; max act is respectively 1, 1 and 3. Task identities were regenerated from canonical `TaskSpec.to_dict()` and compared with the actual `scripts/train_nonllm.py:tasks_for` selector, without resetting any environment. All three exact identity sets matched. Their sorted-task-ID digests are:

| Panel | SHA256 of sorted canonical task IDs |
| --- | --- |
| OOC Act 1 | `6f5c4350f64f0f128dc58eb9811e12ac1c8370dda01294f67118fc6487e06f0c` |
| Full Act 1 | `e1cc55bb17d060edff78dffd5c53c3cb071bfd53fff9cc3a17bda86744ac793a` |
| Full Act 3 | `bf341fbd8a9f397e520f4af7127da1a571073fc5a1b55052a33b2ad8baea684c` |

The snapshot also audits task definitions for all original, expanded and corrected-native final battle panels. Across these definitions and the generated cohort, there are 101 distinct designated final worlds; none is seed 3. The original and corrected Act-1 banks each contain 175 available final battle starts from 20 worlds. World 10240 remains in the generated whole-run cohort even though it has no retained battle-bank starts; this is the pre-existing distinction between a generated run panel and available bank data, not an audit-driven cohort change.

Teacher collection produced labels and battle starts for final worlds. This is permitted data generation, not a learned-policy test. The training CLI filters teacher rows by the fixed world split (`train` for learning, `dev` for development accuracy) and forbids non-evaluation commands with `--split final`. The audit did not inspect final teacher decisions/outcomes as policy performance.

## Persisted-command and outcome checks

The audit inspected 308 saved non-final metrics/record files across `data/nonllm_v1` and `data/nonllm_native_revision`, including verification, health and baseline records. It found 276 recorded command entries and 200,493 world-identity rows (rows can repeat across saved aggregations). All observed metric rows were train or dev: 190,781 train and 9,712 dev. There were no hash-final metric rows, no designated-final world overlaps, no recorded `--split final` commands and no unreadable files in this scan.

The three saved campaign journals contained 173 original commands, 69 corrected-campaign commands and 11 health-study commands. None had frozen identities at this snapshot. The scan of preserved trace/smoke/process JSONL artifacts found only the two learned seed-3 exceptions and the random seed-3 backend smoke described above. No other learned diagnostic on a hash-final world was found in the persisted artifacts examined.

This supports the narrower claim: **the designated quantitative final cohorts have no recorded learned training/evaluation exposure before selection freeze at this audit snapshot**, while world 3 has a disclosed diagnostic exception. Preserve both qualifications. The final immutable checkpoint/source/task/backend freeze remains required before actual final evaluation; this audit does not substitute for that gate or prove later events automatically.
