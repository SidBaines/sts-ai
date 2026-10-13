# Training records and simulator fault replay

The primary experiment worktree is now the separate `codex/reproducibility-fault-records` checkout. The 13 October round-2 OOC study completed update 32 in its original worktree. Its results and source remain historical evidence; the recording changes apply to subsequent jobs. No new scientific training condition or simulator release has been adopted.

## Preserved historical pipeline

[The pipeline audit](rl_pipeline_audit_2026-10-13.json) traces 69 ancestral checkpoints behind the original composite. All referenced checkpoints exist, all recorded parent hashes match, and the recorded dataset paths exist. The successful later sequence is original composite → OOC PPO update 36 → smith distillation → lambda 1 PPO update 12 → smith and elite distillation → round-2 PPO. Combat tensors match the original throughout these later stages.

Round-2 rule labels were collected with the earlier smith-distilled model, then fitted onto the later lambda 1 checkpoint. The audit records both inputs, the actual saved label counts, configurations, resume events and evaluated checkpoints, including the completed update-32 extension.

The verified local archive is `data/reproducibility/pipeline_20261014.zip`, with receipt `data/reproducibility/pipeline_20261014.receipt.json`. It contains 989 source files represented by 985 unique objects, including original-model ancestry, recorded training inputs and metrics, all RL-debug runs, native binaries, source snapshots and Git history. Each object was decompressed and checked against its SHA256. The archive is 4,530,650,879 bytes, or about 4.2 GiB. `manifest.json` maps original paths to content-addressed members. Preserve this archive when cleaning worktrees; it is a local copy, not an off-machine backup.

Archiving does not recover the early missing launch-source bindings, discarded online PPO trajectories or absent historical RNG states. Historical checkpoints also copied original provenance fields into newer models. The audit preserves those limitations instead of treating inherited fields as current training evidence. Source updates after archive creation are retained separately in the branch and companion source bundle.

## Records for future jobs

Each queue launch and each trainer start or resume has an immutable `records/launch-.../launch.json`. It records effective settings, command, source file hashes and source archive, runtime/package versions, actual native manifest and binary, parent-checkpoint identity and input dataset hashes. Identical source and binary snapshots share immutable hard-linked copies in `record_objects`; live inputs and checkpoints are never hard-linked into that store.

OOC and combat PPO write one compressed `collection_XXXX.json.gz` per iteration. These records retain requested task specifications, successful worlds and their action prefixes, preview markers, outcomes, native provenance and fault-file references. Full raw training tensors are not written. Distillation records bind the fitting checkpoint, label shard hashes, settings and fitting log. New output guards reject accidental replacement of existing results.

`latest.pt` and evaluated named checkpoints retain weights, optimizer state, actual iteration/update counts, launch identity and current Python, NumPy, Torch, MPS and local-sampler RNG states. A resume restores these states; old checkpoints without them explicitly log `rng_restored=false`. New checkpoints place inherited metadata under `historical_parent_metadata`, with accurate current launch and initialization identities kept separately. New stages create a fresh optimizer; an explicit resume carries its moments. Source/build changes and legacy resumes remain separately identified conditions.

The bounded native verification compared an uninterrupted two-iteration OOC run with a stop after iteration one followed by a fresh-process resume. Both final weights and every sampled action-trace hash matched exactly. This establishes the tested continuation path on this build; it does not establish identical historical retraining across platforms or after operational faults.

## Storage and throughput

Successful simulator calls build a compact replay sequence in memory. Public-state hashes are recorded at reset and every 64 calls; a fault also captures the exact state immediately before the failed call. Disk fault records are written only on failures. Timeouts, process exits and contained panel failures remain diagnostic outcomes and do not enter loss labels. A failed evidence write raises an explicit recording error.

A warm-worker ABBA control used 16 Act-1 worlds per pass. Recording-off times were 1.675 and 1.726 seconds; recording-on times were 1.671 and 1.718 seconds, with identical trajectories. The median difference was -0.38%, within timing noise: no measurable recording slowdown in this short test. Launch snapshots and checkpoint I/O were outside this collector-only measurement.

One two-game full-run iteration compressed to 2,001 bytes; this is a small sample, not a storage bound. Its complete resume checkpoint was 36,832,203 bytes. Checkpoint retention dominates routine disk use: latest state plus evaluated checkpoints, rather than a full checkpoint for every simulator action. The captured 195-call fault packet was 40,238 bytes. The large historical archive is a one-time preservation operation.

## Fault capture and replay

`CurriculumWorker` records reset, step and teacher-query failures before disposing of the native process. `PanelWorker` additionally records search failures and contained panel faults. Files retain task/seed, successful RPC arguments, failed RPC, periodic and final public hashes, last observation/info, native identity, source-launch reference, worker bounds, exception/child traceback and available process exit code. These private diagnostics never enter policy observations.

Faults normally live under the launch record's `faults/` directory. Direct adapter use defaults to `data/simulator_faults`; explicit `fault_dir=None` is reserved for controlled recording ablations. PPO collection receipts retain the paths. Label and fight-bank helpers quarantine the whole world if it later faults; unexpected policy or recording errors propagate instead of becoming simulator-fault counts. Existing helper outputs are preserved and can contain partial rows from faulted worlds; in particular, the old combat-search datasets require that qualification before reuse.

Replay the exact recorded calls with finite bounds and a new output path:

```sh
.venv/bin/python scripts/rl/replay_fault.py RECORD.json \
  --timeout 10 --seconds 120 --output data/fault_replays/new_case.json
```

The default rejects a different native or public-interface identity. Results distinguish reproduction, a different failure, an earlier failure, divergent state/history, budget exhaustion and success past the former failing operation. A timeout replay establishes the same timeout condition, not a deterministic native defect; compare normal-load runs before diagnosing it.

Test a separately built clean-environment candidate explicitly:

```sh
.venv/bin/python scripts/rl/replay_fault.py RECORD.json \
  --build-dir /CANDIDATE/external/sts_lightspeed/build \
  --environment-source /CANDIDATE/src --allow-native-change \
  --timeout 10 --seconds 120 --output data/fault_replays/new_candidate.json
```

The candidate core is imported in an isolated package overlay, while the replay adapter remains from this branch. All recorded comparison hashes must still agree before the failed operation. A successful operation after a patch is evidence for that case, not certification that all simulator faults are eliminated. Keep native repairs in the clean environment patch, use regression tests and a new release identity, and preserve historical training backends.

## Captured Discovery case

The committed fixture `tests/fixtures/discovery_fault_6000000009.json` captures an actual previously faulted evaluation world: Act 2, floor 21, playing Discovery at RPC 195. Two fresh-process replays reproduced the failure on the RL study build. It also reproduced on frozen clean environment 0.1.1.

The old simulator flags Discovery as unsupported through its undefined-behaviour flag; this is not evidence of a diagnosed memory error. The existing separate `codex/fix-act1-boss-undefined-behavior` 0.1.2 candidate validates Ironclad Discovery offers. A replay using that candidate matched all 195 preceding public-state hashes from the original full-history record and passed the formerly failing action. The candidate has not replaced the study binary or been promoted to a release by this work.

Detailed verification is preserved under `data/repro_verification/20261014_v3/verification.json`, with clean-base and candidate comparison files in the parent directory. Acceptance ran 491 host/native tests successfully with one optional generated-artifact skip, plus the bounded core smoke (21 steps, no error). Recording regressions include the real fault fixture, exact RNG/optimizer restoration, whole-world quarantine, overwrite refusal and queue exit-status integrity. Maintain these source/build identities alongside later experiments.
