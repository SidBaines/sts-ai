# Reproduce the nonlanguage curriculum investigations

These scripts preserve the bounded investigations supporting the [critique](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/nonllm_training_critique_2026-10-12.md). Their captured JSON outputs are in [the evidence directory](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12). No training algorithm or environment implementation was changed by this review.

The scripts are historical local investigation programs. Some intentionally contain absolute paths to the three existing worktrees, ignored teacher banks, original checkpoint, CP128, and pinned native build. The Git branch does not contain those large ignored artifacts. Preserve those worktrees or copy their artifacts before archiving them. Script defaults write only to `/private/tmp`; they do not alter the campaign journals, models, tasks or paused processes.

## Runtime and environment

Run from `/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI` with the existing optional PyTorch/NumPy runtime:

```sh
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build
```

Use `/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/.venv/bin/python` for the commands below. The native module identity for the recorded rollouts is `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`. A later environment/encoder revision is a different investigation condition; changed probe results under it should receive new output names and provenance.

## Probes

| Script | Purpose | Bounds and dependencies |
| --- | --- | --- |
| `nonllm_encoding_critique_probe.py` | Exact public encoder aliases and optional TRAIN scan | Synthetic fixtures require NumPy; optional scan has row cap |
| `nonllm_action_dedup_native_probe.py` | Replay world 12022 step 14 and execute two different raw Strikes | Disposable child with 15-second response limit; original corrected Act 3 teacher bank |
| `nonllm_map_alias_probe.py` | Compare equal candidate vectors with map path statistics | TRAIN rows only, explicit row cap |
| `sts_critic_menu_probe.py` | Demonstrate actor distinction with identical critic values | CPU forward only; reads `/private/tmp/nonllm_encoding_critique_results.json` and original model |
| `sts_learning_probe.py` | Collect phase/forced counts, realized returns and separate gradients | 32 TRAIN worlds, four CPU workers, one thread, 30-second collector limit; backward only, no optimizer |
| `sts_v2_learning_metrics_audit.py` | Parse 129 completed v2 jobs and source-family weights | Existing journals/metrics only; records input hashes |
| `nonllm_operations_probe.py` | Parse runtime/evidence retention and profile a saved CPU fixture | 512 balanced TRAIN observations; CPU forward, no simulator or learner updates |
| `sts_reward_objective_probe.py` | Reward objective counterexample and descriptive interval arithmetic | Reward function/stdlib only, no simulator |

For example, generate the synthetic encoding fixtures and critic comparison in order:

```sh
/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/.venv/bin/python scripts/investigations/nonllm_critique_2026_10_12/nonllm_encoding_critique_probe.py --output /private/tmp/nonllm_encoding_critique_results.json
/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/.venv/bin/python scripts/investigations/nonllm_critique_2026_10_12/sts_critic_menu_probe.py
```

The learning probe defaults to stochastic original-policy collection and `/private/tmp/sts_learning_probe.json`. `STS_PROBE_GREEDY=1` selects greedy control, `STS_PROBE_CHECKPOINT` chooses a different existing model, and `STS_PROBE_OUTPUT` selects an output path. Always choose distinct output names for materially different conditions. The four recorded runs use the same 32 TRAIN worlds and seed initialization, but different policies can visit different states and consume different RNG draws.

The recorded Act 3 map audit uses:

```sh
/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/.venv/bin/python scripts/investigations/nonllm_critique_2026_10_12/nonllm_map_alias_probe.py --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act3_200_native011_seed12000/teacher.jsonl --max-train-rows 150000 --output /private/tmp/nonllm_map_alias_act3_results.json
```

## Evidence interpretation

Synthetic encoding fixtures were not all verified as reachable pairs of native states. They establish information discarded by the encoder, not live failure rates. The separate native free-once replay is a reached TRAIN-state correctness proof with exact prefix/start hash/native provenance. Most duplicate candidate vectors in artifact scans correspond to equivalent choices.

Monte Carlo diagnostics compare predictions with single completed realized returns. They include outcome noise and correlated observations within worlds. They are not exact expected values, confidence intervals or a causal comparison of critics. The backward probes do not update parameters. Every recorded learning collection completed 32 worlds with zero faults and zero cutoffs; those small TRAIN counts are diagnostics, not held-out competence estimates.

The campaign and performance-profile snapshots can change after future authorized work. Captured evidence includes hashes where available. The evidence manifest records each copied script/result/report digest and the review branch's original source commit. Original output paths remain in JSON for provenance; copied filenames provide the durable review artifact.

The full inherited suite includes process-ownership and local HTTP tests. Restricted execution blocks `ps` and localhost socket binding; the review reran these tests with host access and passed 471 tests with one optional artifact skip. Treat those permission errors as an execution-profile limitation, rather than disabling the corresponding tests.
