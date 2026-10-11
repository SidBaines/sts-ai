# Experiment decision notes and recovery instructions

This is the maintained human log; `nonllm_progress.md` is automatically refreshed
every 30 seconds from the atomic command journal. Keep both: the generated page
records exact commands and measurements, while these notes explain decisions.
Raw datasets/metrics/checkpoints are gitignored local artifacts, so do not assume
they exist on a fresh clone. The code and these logs belong on the experiment
branch; never move the environment release tag or modify the original checkout.

## 2026-10-11 20:08–20:50 UTC — setup, profiling and first policies

- Created attached worktree `codex/nonllm-curriculum` from `env-v0.1.0`, preserving
  the original research work and the clean release checkout.
- Wrote the detailed curriculum plan before substantial training. Budget includes
  setup and final evaluation; original deadline is 2026-10-12 06:08:46 UTC.
- Delegated independent encoder/model, curriculum/data, training, and evaluation
  work. The parent owns exclusive GPU scheduling; agents use CPU tests/collection.
- Verified MPS and MLX using actual matrix operations. The sandbox hides Metal
  devices; GPU commands require the approved host execution path.
- Audited fresh 256-world Act 1 bank: 49,944 legal labels, 2,245 starts, 252 valid
  worlds, four quarantined native-UB faults. No old Gemma/native-build banks reused.
- Found and corrected replay RNG accounting: prefixes must preserve OOC preview
  commits. Twenty-five sampled starts reproduce public state and native labels.
- Found potion description grammar mismatch before training and versioned encoder
  `public_candidate_v2`; old CPU smoke v1 checkpoints are not campaign inputs.
- Measured 1/4/8/12 workers, exact observation caching, CPU/MPS/MLX training and
  inference. Selected eight workers, CPU rollout inference, MPS batch updates.
  Measurements and qualifications live in `nonllm_throughput.md`.
- Ran 228 tests plus required bounded release smoke before GPU training. Added
  further regression tests for masking, PPO old-policy ratios, GAE time limits,
  checkpoint recovery, cache/gradient parity, split/report aggregation and replay.
- First small policies trained roughly 6–7 million examples each in 90-second
  jobs. Development imitation is only diagnostic; forced choices are now reported
  separately so they cannot inflate meaningful action accuracy unnoticed.
- Initial evaluation hit a missing CLI cache flag. Preserved both completed BC
  checkpoints, paused only our dispatcher, added flags, and reran evaluations.
  The original deadline remained fixed. First models each won 42/48 matched dev
  fights; same-task random 28/48, public heuristic 42/48, native search 44/48.
  No final worlds evaluated. Small panels do not establish integrated competence.
- Collected separate Act 3 enrichment: 68,520 labels, 2,983 starts, 246 valid
  worlds/ten flagged faults. Act 3 coverage is thin (123 fights, no final boss).
  Do not claim boss generalization from this bank.
- Collected separate privileged train-only OOC lookahead labels: 256 frontiers,
  four candidates, ≤30-decision continuations, 89 strict proxy improvements.
  Act-clear healing complicates endpoint-HP proxy; compare real held-out outcomes.
- Identified prefix-reset overhead: replay repeatedly renders intermediate states
  that are immediately discarded. A read-only prototype suggests a large gain;
  an optional production path must preserve per-step UB and RNG/terminal checks.
- Identified a released auto-native-combat limitation: its temporary BattleContext
  UB flag is not latched. Zero surfaced OOC/lookahead faults cannot certify zero
  native UB. Audit existing APIs before any change; do not silently change the
  frozen simulator or reinterpret faulted games as ordinary defeats.

## Next gates and remaining work

1. Finish six size/context BC screens, learning-rate comparisons and seed repeats.
2. Verify CPU-inference/MPS-optimizer routing with genuine multi-round updates.
3. Compare combat PPO hyperparameters/reward controls; use easy-to-hard replay
   starts and corrective labels, retaining earlier fights to limit forgetting.
4. Compare scripted versus lookahead OOC warm starts and optimize OOC with a fixed
   native combat controller. Keep its search budget consistent across conditions.
5. Train full-control traces, expand Act 1 to Act 3, compare shared/specialized
   composition, and evaluate actual learner control of both phases.
6. Freeze selection before final embargo access. Report grouped-world uncertainty,
   faults, truncations, Act clears and full victories. Stop training by the deadline.

## Resume safely

Worktree: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI`.
Use `.venv/bin/python`, `PYTHONPATH=src`, and the immutable base build path in the
campaign config. GPU operations need host execution; do not let independent
agents launch GPU jobs. Check process ownership and `campaign.json` heartbeat;
a live PID alone is not evidence of training progress. Preserve final checkpoint
artifacts before interrupting our dispatcher. Its child jobs use their own process
groups so bounded termination can include spawned simulator workers.

The supervisor reuses completed commands only when metrics exist. For an interrupted
command, a metrics file with top-level `elapsed_seconds` means the CLI finished;
recover that journal entry as complete after checking the checkpoint. A partial
PPO metrics file is not a completed job. Resume from its atomic checkpoint with
explicit matching data/reward/config identity, recording any trajectory restart.
Never erase failed commands or silently replay beyond the original deadline.

## 2026-10-11 21:28 UTC — laptop interruption and authorized extension

- Sid closed the laptop and explicitly asked to continue on return. All six
  architecture checkpoints were preserved. The dispatcher had been paused for
  an isolated GPU routing verification; its completed child was recovered from
  the final metrics/checkpoint instead of repeating training.
- Genuine CPU inference/MPS PPO updates passed: two rounds, 258 transitions,
  eight optimizer steps, 2.49s total, no surfaced faults. PPO ratios changed
  from one, KL was finite and the optimizer survived both device transitions.
  Artifact: `data/nonllm_v1/verification/mps_route.{pt,metrics.json}`.
- Sid extended the hard deadline to **2026-10-12 06:30:00 UTC**. The original
  06:08:46 deadline remains recorded for audit. Current phase boundaries retain
  time for OOC, integrated traces and final evaluation.
- The Mac reported battery power at 62%; Sid confirmed he will connect power.
  Continue checking power/progress rather than assuming a live process is enough.
- The detailed generated progress page refreshes every 30 seconds; its writer
  PID is 12028. It has no GPU usage. Raw command records and failed invocations
  remain intact. Each new condition records adapter source hashes.

## 2026-10-11 21:32 UTC — recovery verified and replay optimization

- Power is now AC, charging. Resumed architecture learning-rate jobs show fresh
  supervisor heartbeats and actual checkpoint/metric progress.
- Full experiment suite passed **242 tests** in 20.27s, with simulator tests
  enabled and no skipped native gates.
- Fast prefix replay passed matched RPC tests: twenty later-act starts, 7,852
  prefix actions, 4.118s → 1.579s (**2.61×**). Public observations, teacher labels,
  following rewards, diagnostics and seeded fault rejection match exactly.
- The optional replay path keeps the immutable native/Python base identities.
  Frozen bank collection source hash is `64389785…`; optimized adapter source
  hash is `08e37bd5…`. Replaying these banks is compatible; collection resume
  under a changed adapter is deliberately rejected. New commands log source hashes.
- Independent review caught optimizer reset between PPO chunks, selected reward
  mode being discarded, restart curriculum clock reset, and stale metrics reuse
  after a failed command. Repairs preserve compatible AdamW moments, selected
  reward, persistent phase frontier, and require successful fresh metrics.
  Incompatible BC/reward/task changes reset optimizer with a logged reason.
- Final Act 3 evaluations now include matching Act 3 random/heuristic/native
  baselines on the same embargoed worlds; selection is frozen first.

## 2026-10-11 21:41 UTC — real inference and optimizer checks

- Leading large policy real ABBA inference benchmark: warm CPU collection
  0.962s versus MPS 1.395s for 687 decisions/48 fights; actor+bootstrap CPU
  0.375s versus MPS 0.787s. All four runs had 45 wins and 687 transitions.
  This establishes matching counts, not a separate assertion of bitwise traces.
- First optimizer-carry smoke correctly reset because older checkpoint reward
  metadata differed. A second smoke using current metadata restored real MPS
  AdamW moments: source step 4 → 7, `compatible_ppo_chunk`, 1.377s.
- Two new regression tests prevent a failed command from reusing stale successful
  metrics and prevent the log attributing a later rerun's results to a failed job.
- Reviewed supervisor resumed PID 17369; generated-log writer PID 17370. These
  are recovery handles, not sufficient evidence of progress; check fresh metrics.
- User requested a short reverse-chronological summary in addition to the full
  log. Maintain `docs/nonllm_summary.md` at major phase changes and meaningful
  decisions; put dated updates at the top, include compact outcome/speed tables
  or plots, preserve uncertainty, and link underlying metrics rather than
  copying commands there. Keep this note and the raw timeline detailed.

## 2026-10-11 21:42 UTC — full verification gate

Full suite: **245 tests passed**, simulator gates enabled, in 20.719s.
The campaign is running with fresh heartbeats and continuing learning-rate
comparisons; current source and logs are being committed as a recovery point.
