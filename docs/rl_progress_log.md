# RL progress log (most recent first)

**Live dashboard: <http://127.0.0.1:8791>** (all runs, eval curves, paired gain vs original, learning health)
· queue: `.venv/bin/python scripts/rl/queue_runner.py status`
· runs and logs: `data/rl_runs/<name>/log.jsonl`
· background: [rl_investigation_2026-10-13.md](rl_investigation_2026-10-13.md)

If the dashboard is down: `.venv/bin/python scripts/rl/dashboard.py start`. If the
queue runner is down: `.venv/bin/python scripts/rl/queue_runner.py start` (it pauses
runs on battery and resumes on AC).

Reference point for everything below: the original composite (`joint_act3_0030.pt`),
greedy, dead-action filter, 1,014 valid TRAIN-eval worlds (seeds from 5e9):
**29.27 ± 0.26 mean floor**, Act 1 boss pass 88.1%, Act 2 boss reach 48.9% / pass
23.4%, 1 win.

---

## 2026-10-13 14:35 — Fight-panel binding built and verified; evaluation determinism bug fixed

- **Native binding for the fight panel is built.**
  - `patches/sts_lightspeed_rl_panel.patch` (25 lines, applied on top of the
    unchanged 0.1.1 study patch) exposes `GameContext.enter_battle(encounter)`
    and the run's upcoming hallway and elite encounter lists.
  - Build: `scripts/rl/build_native_rl.sh` → `external/sts_lightspeed_rl/build`,
    using an isolated `.venv-native` for cmake. The frozen venv and binary are
    untouched.
  - The study patch reproduces byte-for-byte (`77fddbd…`). **The new binary
    plays identically to the frozen one on 125/125 worlds** (same final floor
    and step count), so it is a safe drop-in.
- **Bug (mine), now fixed:** the first version of `evaluate()` implemented
  "greedy" as logits×10⁴ and sampled, rather than an exact argmax. Exact ties
  (e.g. two identical Strike copies on a card-select screen) then broke
  randomly, and runs diverged chaotically: ~20% of worlds changed between
  repeats. With exact argmax, evaluation is fully deterministic at a fixed
  worker count.
  - The E8 numbers in the investigation used a true argmax and stand.
  - `ooc_main_s1` started with the old evaluator. Its paired numbers are
    unbiased but noisier; I will re-score its saved checkpoints with argmax.
- Recomputing the baseline with exact argmax
  (`data/rl_runs/baselines/orig_argmax_1024.json`); queued runs will use it.

## 2026-10-13 13:30 — Took over as primary agent; infrastructure up, first queue running

- New package `src/sts_rl/` (composes the frozen `sts_policy`/`sts_env`, edits none of it):
  - `evaluate.py`: deployment-style greedy eval on fixed worlds with per-world
    records, boss reach/pass/entry HP, and paired comparison.
  - `ooc_trainer.py`: the E8 recipe generalised. Every load-bearing choice is a
    flag. It logs half-batch gradient cosine, displacement vs the Adam random-walk
    line, and behaviour stats (smith rate, card skip, map choices, fights/run)
    every update.
  - `common.py`: worker/task helpers, phase-greedy forward, and a dead-action
    filter (Act 4 keys removed from the menu the policy sees).
- `scripts/rl/queue_runner.py`: detached queue and power-aware babysitter.
  `scripts/rl/dashboard.py`: live dashboard. `scripts/rl/eval_policy.py`: eval CLI.
- Baseline recomputed with the dead-action filter (above). Without the filter it
  was 29.19; the filter itself is worth ~+0.07, as in the overlay probe.
- Queue (in order):
  1. `ooc_main_s1`: E8 recipe, seed 1, 24 updates. Replicates E8 at update 8
     and shows whether gains continue.
  2. `abl_sample_combat`: sampled combat during collection.
  3. `abl_micro_credit`: GAE through the combat critic.
  4. `ooc_main_s2`: second seed.
  5. `abl_small_batch`: 128 games/update, matched total games.
- Next while these run: the fight-panel deck-strength evaluator (X1).
