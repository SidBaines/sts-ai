# Periodic full-run development observer

`scripts/monitor_full_run_eval.py` observes the long curriculum independently. It uses CPU inference, four workers, one Torch thread, and nice 15; it never changes learner checkpoints, selection, goals, rewards, or allocation. The fixed panel is exactly the 48 prespecified development worlds from `configs/long_curriculum_protocol.json`: max Act 3, horizon 4096, policy seed 777, native simulations 200, preserved task prefixes. Final worlds remain embargoed.

On startup and every 30 minutes after finishing a batch, the observer captures the selected best and latest **completed** training checkpoint. It copies bytes from an open descriptor into its own SHA-addressed snapshot directory and verifies the completed job's expected SHA before and after copying. Retention can unlink the original after open without invalidating the captured bytes. A missing original before open is retried from a later journal. Identical checkpoint conditions reuse an immutable measured panel and add role aliases; reuse is not counted as a fresh evaluation.

The initial full-three-act anchor is imported read-only from the protocol's raw metrics, checkpoint SHA, exact task IDs and condition signature. Its result is 0/48; its timestamp is explicitly the saved metrics file's modification time. Surface simulator faults separately from ordinary defeats, truncation and unstarted requests. Every panel validates checkpoint path, backend/module SHA, Python interface, CPU configuration and exact requested task IDs before publishing results. These measurements are diagnostic and do not enter learner promotion or rollback.

Run from the long-curriculum worktree:

```sh
PYTHONPATH=src .venv/bin/python scripts/monitor_full_run_eval.py --execute
```

A bounded validation command is `--execute --once --max-panels 1`. Plan-only invocation without `--execute` starts no process. Production launch belongs to the orchestrating agent. SIGTERM stops only the observer and its verified owned CPU child group; it does not signal the learner.

Outputs are confined to `data/long_rl_v1/full_run_eval/`: immutable snapshots under `checkpoints/`, unique command logs/raw metrics under `jobs/`, observer journal/progress, and dashboard `telemetry.json`. The telemetry has `version: 1`, `monitoring_only: true`, `histories`, role-indexed `latest`, and a separate `baseline`. Records expose `roles`, `checkpoint_sha256`, `captured_at`, `captured_global_round`, `evaluated_at`, `task_ids`, `task_ids_sha256`, `complete`, `coverage_complete`, `requested`, `clears`, `defeats`, `faults`, `truncated`, `unstarted`, `floor_mean`, raw metrics path/SHA and training-environment provenance. A reused role alias includes the new capture time while preserving the original measurement timestamp.

The observer checks the learner's running status, controller birth identity, active allocation, optional hard UTC deadline, battery and disk guards. It starts no panel while the learner is paused, out of allocation, or needs attention; it remains waiting and resumes only when that same authorized learner is running again. Child execution is bounded by the panel budget plus 60 seconds and a 150-second absence-of-output stall limit. Frozen learner source/backend/task hashes are checked before evaluation. The learner's active clock and checkpoint publication are untouched.

## Bounded verification

Twelve focused CPU tests pass. The first actual bounded full panel finished in 16.75 seconds with exact 48 requested/classified DEV task IDs: 0 clears, 48 defeats, 0 surfaced faults, 0 truncations and 0 unstarted tasks; mean floor 30.9167. Captured selected checkpoint SHA is `9019cdec9c4c2b92520249919d7c97d6f6e626907511b510a44d88258cede278`. Raw metrics SHA is `d43e839c76d055b81bb6e9abd5f0425f27ceccc54d98afeb52ae4e476b566b8f`. A host process inspection confirms no members of its owned group remain. This is monitoring evidence, not a learner promotion. Latest-candidate evaluation remains pending the orchestrator's production launch.
