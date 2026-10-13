# scripts/rl — sts_rl experiment tooling

Progress log and live-dashboard link: `docs/rl_progress_log.md`. Code: `src/sts_rl/`.

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
