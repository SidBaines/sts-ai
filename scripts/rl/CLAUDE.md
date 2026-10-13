# scripts/rl — sts_rl experiment tooling

Start with `docs/rl_findings_2026-10-13.md` (what works, what failed and why, next
directions). Chronological detail and the dashboard link: `docs/rl_progress_log.md`.
Code: `src/sts_rl/`.

- **Run jobs through `queue_runner.py`** (detached, pauses on battery, one job
  per `--lane`). Run outputs and logs are in `data/rl_runs/<name>/`.
- **Native build:** `build_native_rl.sh` → `external/sts_lightspeed_rl/build`.
  It applies the unchanged 0.1.1 study patch plus `patches/sts_lightspeed_rl_panel.patch`
  (`GameContext.copy/enter_battle/upcoming_*_encounters`) and plays identically
  to the frozen binary. The fight panel (`src/sts_rl/panel.py`) needs this build.
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
  are identical across worker counts.
- `dashboard.py` updates its charts in place, polls
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
- **Selection bias:** every comparison uses the same eval worlds (seeds from
  5e9). After choosing among variants, confirm the winner on fresh TRAIN worlds
  (`--world-start 5600000000`, `data/rl_runs/fresh_eval/`).
  `scripts/rl/compare_evals.py A.json B.json ...` gives the paired difference.
- **Progress-log headers:** take the time from `date` when writing the entry.
  Never estimate it.
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
  (worker step timeouts fire as ordinary faults on resume). Panel and search
  helpers with long RPC timeouts are better restarted than resumed.
- **Search helpers:** `PanelWorker.search_votes` is privileged: it keeps the
  true battle RNG streams. `search_votes_public` re-seeds them and needs
  `build_pub`. Use only the public one for anything presented as agent
  capability. Distilling either one's per-decision votes into the combat head
  hurt the Act 2 boss (−6 points), whereas playing with public search helped
  (+8 points).
