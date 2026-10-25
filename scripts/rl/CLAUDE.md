# scripts/rl — sts_rl experiment tooling

Start with [research/README.md](../../research/README.md) for the current state
and [research/strategies.md](../../research/strategies.md) for what works and
what failed. Commands and module usage: [src/sts_rl/README.md](../../src/sts_rl/README.md).

- **Run jobs through `queue_runner.py`** (detached, pauses on battery, one job
  per `--lane`). Run outputs and logs are in `data/rl_runs/<name>/`.
- **Disk trap:** record-search outputs and fit manifests are large; collectors and fits stop below 6 GiB free, and
  a busy day writes about 9 GiB of run data. Run `disk_report.py` before launching big batches.
- **Check disk periodically** (at least daily during long sessions, and whenever free space is under about
  30 GiB): run `disk_report.py <each data/rl_runs root> --out <nonllm-large-batch>/data/disk_reports`. Sid
  approved applying `--apply compress-fit-manifests --apply interrupted-fit-outputs --apply
  trainer-latest-duplicates` whenever the report finds candidates; every action is logged to
  `cleanup_log.tsv` there. Anything beyond those three rules (deleting checkpoints, corpora, evaluation games
  or benchmark details) needs Sid's approval. There is deliberately no scheduled job.
- **Native build:** `build_native_rl.sh` applies the current environment patch
  plus `patches/sts_lightspeed_rl_panel.patch`
  (`GameContext.copy/enter_battle/upcoming_*_encounters`). The fight panel
  (`src/sts_rl/panel.py`) needs this build. It uses the separate
  `.venv-native` build tools, takes the target interpreter from `PYTHON`, and
  requires a new, empty `BUILD_DIR`. Never build over a module a running job
  has loaded, and never install into a frozen runtime.
- **Memory:** `collect_rollouts` keeps every transition, with its full
  observation dict, until it returns. A 768-game batch plus a 1,024-game eval
  in one process reached 20+ GB and drove the 48 GB Mac into swap exhaustion
  (macOS then force-quit Chrome). `sts_rl` collects and evaluates in chunks
  (`--collect-chunk`, `evaluate(chunk=256)`), keeps only OOC choice
  transitions, and slims their observations in place. Peak is ~6 GB per
  trainer. Check `top -o mem` and `sysctl vm.swapusage` before adding jobs.
- **Evaluation must be an exact argmax** (`evaluate(mode="greedy")`). Logit-scaled
  "greedy" sampling breaks exact ties (duplicate card copies) randomly, and the
  run diverges chaotically, so results become irreproducible. Argmax results
  of the per-decision evaluator are identical across worker counts. The
  batched evaluator (`fast_eval.py`) is reproducible only for a fixed
  `--games-per-process` and `--processes`: batching can flip exact ties
  between equivalent actions. With one game per process it reproduces the
  per-decision evaluator exactly. Compare arms only at identical settings.
- `dashboard.py` (rebuilt 2026-10-13; `dashboard.py restart` after edits) shows the win set first, then active jobs and key runs (`RUN_INFO`; others archived). It updates charts in place, polls
  `/api/state?since=<runs_sig>&max_points=N`, and pauses while the tab is hidden.
  Don't go back to destroying and recreating charts on every poll. Chrome
  dying on this machine is usually system-wide memory exhaustion from
  trainers, not the dashboard.
- Panel fights must never run the game's post-battle code on scratch copies
  (`exit_battle` → `afterBattle` can call an empty native callback:
  `std::bad_function_call`). `panel.py` stubs `exit_battle` during panel fights.
- **Throughput:** single-process collection is bound by the trainer's main
  thread (Python encoding plus forward), with game workers mostly idle. Use
  `--collectors K` (`src/sts_rl/parallel_collect.py`). It gives ~2× throughput
  at K=4, and greedy eval records are identical. Use `--micro-batch`
  (default 1024) for the PPO update; a 4,096-row padded minibatch alone spiked
  to ~12 GB.
- **Queue names vs `--out`:** the stall watchdog reads the log under the run's
  `--out` directory (fixed 2026-10-13). Before the fix, a resumed or branched
  run whose queue name differed from its out dir was killed 40 minutes after
  launch. Restart the runner process after editing `queue_runner.py`; running
  jobs survive, since they are in their own sessions.
  The watchdog also counts `<--out>/progress.jsonl`, which `fresh_collect.py`
  writes after every world (added 2026-10-25, after it killed eight healthy
  labelling jobs). Any other job that prints nothing for 40 minutes needs a
  progress file or a keep-alive.
- **PPO update on the GPU:** `train_ooc.py --update-device mps` runs the PPO
  update on Apple's GPU (25 October benchmark, one 3,072-game iteration: update
  233 s on CPU, 109 to 113 s on MPS; whole iteration 433 s to about 315 s).
  MPS runs repeat bit for bit, but MPS and CPU runs diverge (largest weight
  difference after one update 0.016, against 0.047 for the update itself). So
  use one device for every arm of a matched comparison and for resuming a run.
  `update_seconds` includes the CPU gradient-cosine health check
  (`--health-every`).
- **Branching a run:** copy its `latest.pt` into a new `--out` and pass
  `--resume`. Same `--seed` means the same training worlds, so comparisons are
  paired. `--lr` now overrides the optimizer's saved lr on resume; it used to
  be silently ignored.
- **Strategic decisions:** the OOC head's logits are saturated. PPO never
  samples smith, elite paths or removals, and per-decision credit noise
  (~0.15) dwarfs their effects (~0.02). The working loop is:
  1. Measure a behaviour with a menu overlay
     (`eval_policy.py --smith-above/--elite-above/--prefer-remove/--skip-cards-after/--dominated-filter`),
     paired over 1,024 worlds.
  2. Distil any winning rule (`collect_rule_labels.py` → `distill_ooc.py`).
  3. Continue PPO from the distilled checkpoint.

  `--strategic-temperature` explores those menus, but on its own it learned
  the wrong lesson: low-HP smiths are very costly.
- **Measure on the study protocol: pass `--hide-ooc-potions`.** Since
  environment 0.1.5 the menu offers out-of-combat potion drinks and
  discards. Our policies discard almost every potion they hold when offered
  (588 discards in 64 games on 20 October), and the habit does not unlearn
  under PPO. `eval_policy.py` and `train_ooc.py` show those actions by
  default, which roughly halves the measured win rate (the 19 October
  champion: 1.7% against 3.8%) and hid a confirmed +2-point gain in a
  trainer's own evaluation. Pass `--hide-ooc-potions` to both so they match
  `fast_eval.py`, and never compare numbers across the two protocols.
- **Cut topic branches from `origin/ws/nonllm`.** The local `ws/nonllm`
  branch (checked out in an old worktree under `/private/tmp`) lags the
  remote, so a branch cut from it silently misses recent work. Run
  `git fetch` first.
- **Selection bias:** every comparison reuses the same DEV worlds. After
  choosing among variants, confirm the winner on fresh confirmation worlds.
  The world ranges depend on the environment version: see
  `research/metrics.md` (on 0.1.5, DEV starts at 91,000,000,000; the seeds
  quoted below are from before 0.1.3).
  `scripts/rl/compare_evals.py A.json B.json ...` gives the paired difference.
- **Timestamps** in `research/status.md` and Log entries: take the time from
  `date`. Never estimate it.
- **OOC credit must not be critic-bootstrapped at λ = 0.95** (2026-10-13). The
  OOC critic cannot see the value of delayed-payoff choices, such as a single
  card upgrade, but it sees HP directly. So λ = 0.95 macro advantages rank
  "rest" above "smith" at high HP: the sign is flipped relative to paired
  evaluation. Plain PPO from a distilled smithing policy unlearned it: smith
  rate 45% → 14.5% by update 16, −0.54 floors. With `--gae-lambda 1.0`
  (Monte-Carlo, critic only as a baseline) it kept the behaviour and improved,
  +0.88 floors by update 12. Prefer λ = 1.0 for OOC runs; it is noisier,
  but it is not biased.
- **Combat:** fight-bank PPO did not move full-game outcomes, either with
  sparse win/loss (v1/v2) or with dense potential shaping (v3, 20 updates).
  Don't re-run either without a new idea.
- **Win-rate claims** need the 4,096-world win set (`--world-start 6000000000
  --n 4096`, `data/rl_runs/win_eval/`) and the McNemar line from
  `compare_evals.py`. At ~2% wins, 1,024 worlds cannot resolve 1-point changes.
- **Run every heavy job through `queue_runner.py`, one-off evals and search
  tests included.** The runner pauses its jobs on battery and resumes them on
  AC. An ad-hoc `nohup` job keeps draining the battery, and a hand-rolled
  resume watcher dies with the agent session. Trainers survive a long SIGSTOP
  (worker step timeouts fire as ordinary faults on resume). For evaluations
  those faults leave the cohort incomplete: re-run each affected shard whole
  into a new folder, or use `fast_eval.py`, whose stall check ignores time the
  whole job spent paused. Panel and search helpers with long RPC timeouts are
  better restarted than resumed.
- **Search helpers:** `PanelWorker.search_votes` is privileged: it keeps the
  true battle RNG streams. `search_votes_public` re-seeds them and needs
  `build_pub`. Use only the public one for anything presented as agent
  capability. Distilling either one's per-decision votes into the combat head
  hurt the Act 2 boss (−6 points), whereas playing with public search helped
  (+8 points).
- **Select the native build explicitly.** The queue's built-in
  `STS_LIGHTSPEED_BUILD` overrides the shell environment. Use
  `-- env STS_LIGHTSPEED_BUILD=BUILD ...` in the queued command so both its
  launch record and child use the intended build.
- **Plan a migration before editing source-bound checkpoints.** Entity loading
  compares a SHA-256 of all bytes in `src/sts_policy/entity_encoding.py` and
  checks vocabulary identity. Controlled loading additionally compares source
  bytes for `controlled_encoding.py`, `controlled_entity.py`, `entity_model.py`
  and legacy `encoding.py`. Even a comment edit changes these identities and
  makes affected checkpoints refuse to load. Never edit those files without a
  planned migration; ordinary entity loading does not itself hash the model file.
- **Preserve saved frontiers explicitly.** In `train_nonllm.py`, use
  `--preserve-task-prefixes` for saved out-of-combat/run continuations; without it,
  the task file supplies worlds restarted from the beginning. The flag also
  preserves each task's horizon, act limit and search settings.
- **Check the replay worker class.** The generic fault replayer accepts only
  `CurriculumWorker` and `PanelWorker`; fresh-panel and lookahead subclass
  records are not supported. Do not promise replay merely because a packet exists.
- **Keep faults out of outcomes and targets.** Simulator failures, reported
  undefined behavior, timeouts and censored worlds are not defeats. Quarantine
  whole-world labels where supported; older helper outputs can contain partial
  faulted-world rows. Programming and recording errors must propagate.
- **Preserve fresh-start filtering and selection.** Recover public potion slots
  from the raw menu before hiding out-of-combat drink/discard choices; preserve
  original indices and keep combat potions. Call `policy_observation` with the
  raw unfiltered observation; it creates policy context without mutating that input.
  Freeze panel grids before fitting or scoring students, defer test scoring until validation-only selection freezes,
  and compare exact cohorts with frozen expectations. Never use cohort intersections
  to repair incomplete fresh-start results.
- **Teacher benchmarks are not champion games.** Set boss-only teacher
  probabilities; never fit or compare these cohorts as the champion. See the
  [collector options](../../src/sts_rl/README.md#fight-panels-and-fresh-start-tools).
- **Final-boss scope:** identify final-boss fights from the public act and
  room fields, and verify the encounter when investigating anomalies. Do not
  assume the Act 3 final boss is on floor 50: Secret Portal can legally reach
  it earlier, so fixed-floor counters undercount entrants. (A floor-50 counter
  caused a false alarm about invalid wins on 14 October.)
- **Keep evaluated checkpoints:** never let checkpoint retention delete a
  checkpoint whose results are reported or used for a decision. Copy it
  somewhere retention won't touch before evaluating it. (Retention deleted an
  evaluated long-curriculum v2 checkpoint, which blocked later diagnosis.)
