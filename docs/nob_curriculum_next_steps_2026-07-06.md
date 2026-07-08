# Fight-focused training on Gremlin Nob — capability assessment & next steps

**Date:** 2026-07-06 · **Status:** parked (candidate work; a side quest may reshuffle the ordering below)
**Goal:** improve gemma-4-E4B (native thinking) at a *single* fight — Gremlin Nob — as a possible first rung of a
fight-by-fight "learning schedule" (curriculum RL). This doc is the writeup of the codebase capability
investigation so we can pick it up without re-deriving.

> **Implementation update (2026-07-07):** this is now implemented as a detachable **local curriculum task** lane,
> not as Nob-specific logic in the main full-run training path. See `src/sts_ai/local_tasks/` and
> `scripts/local_task_{prepare,build_sft,eval,compare,grpo}.py`. Gremlin Nob is the first task (`--task
> gremlin_nob`): task reward and labels live in task manifests / `meta.extra["local_task"]`, adapters are standalone
> skill patches, and the main harness only consumes them through the existing `--adapter-path` surface. The default
> source split over the 43 Nob fights is 32 train / 11 holdout (`world_seed % 4 == 0` holdout).

## Context / why the Nob

- We are **not** in the framing-experiment phase yet — the near-term goal is competence (get E4B to play well).
- From `scratch/nob_fight_analysis.py` over the 100 base E4B rollouts: **43 Nob fights — 23 won, 20 lost,
  9 "convincing" (≤20 HP lost), incl. seed 74 flawless (0 HP, 3 turns).**
- **The winning policy is attack-forward:** won fights attack 31% / block 23%; lost fights attack 23% / block 28%,
  and in wins the attack share *rises* over turns. The model loses by turtling behind Block, which feeds Enrage.
  (See the plots in `scratch/plots/` and the artifact built by `.../scratchpad/build_nob_artifact.py`.)
- Implication: a reward keyed on "convincingly won the Nob fight" (survived, low HP lost) naturally selects
  attack-forward play, and is **trait-neutral** (an outcome, not a risk/strategy word), so it won't contaminate
  the later framing research (`docs/rl_and_framing_design.md` neutrality invariant).

> **Update (2026-07-07):** the "GRPO is CUDA-only / NVIDIA-only" statements below are now **outdated**. A local
> **MLX GRPO path** was built — `src/sts_ai/train/train_pg_mlx.py` (in-process MLX LoRA policy-gradient trainer) +
> `src/sts_ai/train/mlx_grpo.py`, run via `run_grpo.py --backend mlx`: the same `grpo_loop` in one unified-memory
> process, no vLLM/CUDA, validated end-to-end on Qwen3-1.7B and gemma-4-e4b. My earlier reasons #2/#3 (LoRA
> hot-swap into a separate vLLM engine; the CUDA sleep/wake co-residency dance) were **architecture, not
> fundamentals**, and #1 (no MLX PG trainer) is resolved. It's the **cheap-local-iteration** path (lockstep
> generation, mlx LoRA ≠ PEFT bitwise, no LoRA/sleep on Metal); `--backend cuda` stays the defensible pod path.
> See `docs/grpo_dryrun_checklist.md` §5b and `src/sts_ai/CLAUDE.md`.

## Bottom line

The training *machinery* already exists (SFT/RWR/PG/GRPO). The original missing pieces were **fight-scoped
reward** and a **fight-scoped episode**; these now live in the detachable local-task lane for `gremlin_nob`.
The single enabling fact:

> Every rollout record stores the model-facing prompt (`state_text`) **and** the verbatim completion
> (`agent.raw_response`, incl. E4B's `thought` channel). So offline training examples reconstruct with **zero
> train/inference skew** (`sts_ai.train.sft_format.build_example`) — **no re-sampling required** to start.

Three rungs, increasing cost (details below):
1. **RWR / filtered-SFT on won-Nob combat decisions** — ~a day of glue, **runs on MPS**, uses existing data.
2. **Hinted-rollout SFT** — the existing "teacher" analogue; breaks the thin-positives ceiling; mostly config.
3. **Nob-scoped GRPO** — the real "learning schedule" rung; now available locally through
   `scripts/local_task_grpo.py --backend mlx`; CUDA remains a later scale-up path, not required for the first pass.

---

## Axis A — Do we have the rollouts / deterministic seeds? **Yes.**

- **Data:** `data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192/` — 100 full E4B rollouts
  (native thinking, full-LLM combat, Acts 1–3), per-decision JSONL + `.meta.json`.
- **Nob fights already extracted & HP-labelled:** `scratch/nob_fight_analysis.py` (43 fights; seeds of the 9
  convincing wins: `47,69,74,79,85,88,111,116,128`).
- Each combat record has everything a reward fn *and* a skew-free SFT target need: `state_text`,
  `agent.raw_response`, `agent.valid`/`retries`, `selected_action`, `legal_actions`, `affordances`
  (lethal-available / full-block-possible), and `state`/`after_state` HP for outcome.
- **Deterministic seeds** → we can regenerate/extend cheaply if the positive set is too thin.

**Caveat:** 9 convincing wins is thin for SFT. Cheap fixes: (a) relax the label to "won" (23), or
(b) generate more via *partial rollouts from the Nob entry* (Axis B, replay primitive) — ~5–10× cheaper than
full games because we skip the pre-Nob game.

---

## Axis B — SFT / RWR / RL / logit distillation? + code deltas

| Method | Exists today? | Delta to focus on the Nob |
|---|---|---|
| **Filtered-SFT / RWR** | ✅ full pipeline (`scripts/build_sft_dataset.py` → `scripts/train_policy.py`); RWR = deterministic resampling→SFT (`train/reward.py::rwr_multiplicities`) | **~100–150 LOC:** a "Nob-fight dataset builder" — reuse the fight-window extraction from `scratch/nob_fight_analysis.py`, label won/convincing by HP, emit `sft_format.build_example()` for those combat records (replicate by HP-margin for RWR). **No new trainer.** |
| **Offline signed-advantage PG** | ✅ whole-run path exists (`scripts/build_pg_dataset.py` + `scripts/train_pg.py`); local-task reward path now exists in `local_tasks/pg_dataset.py` | For Nob, use task-local reward/meta rather than `final_floor`; local MLX PG/GRPO is available for the first pass. |
| **Online GRPO** | ✅ whole-run loop/loss exists; local-task MLX entrypoint now exists (`scripts/local_task_grpo.py`) | Fight-scoped episode + Nob reward + replay-to-Nob env are wired for `--task gremlin_nob`; CUDA/vLLM remains a later scale-up path. |
| **Logit distillation** | ❌ not supported | We store completion *text + token counts* but **no teacher logprobs**. Needs a teacher emitting logits + a KL-distillation loss — neither exists. **Skip** — hinting (rung 2) is a cheaper teacher. |

### Two real gates for the RL path
1. **`train/pg_dataset.py` refuses native-thinking traces** — `require_no_thinking=True` default raises on
   `reasoning_mode="native"` (~line 50). E4B *is* native-thinking, so PG/GRPO on it needs this plumbed off +
   validated. The thinking reconstruction machinery already exists (SFT handles `gemma_thought` passthrough,
   `enable_thinking=True`), so this is plumbing + a test, **not new algorithm**.
2. **Reward is whole-run only** — `train/reward.py` = VICTORY / `final_act>min_act` / `final_floor`. **No
   per-fight reward.** Everything to compute "survived the Nob with ≤N HP lost" is in the traces → new pure fn,
   ~30 lines (effectively already written in `nob_fight_analysis.py`).

### The mid-game-start primitive already exists
`src/sts_ai/interactive/replay.py::replay_actions` is a **pure, reusable** helper: the C++ `GameContext`/
`BattleContext` are opaque (no binary snapshot), so replay re-applies a recorded action sequence into a *fresh*
`LightspeedHybridEnv` to rebuild any position. So **"start a rollout at the Nob entry" = replay that seed's
recorded pre-Nob OOC actions, then let the policy play the combat.** It is **not** wired into any training/CLI
runner yet (only the interactive server) — hooking it into a `make_env` is small glue. Curriculum shape:
*pin the pre-Nob path (from the base trace), sample only the combat.*

---

## Axis C — MPS vs NVIDIA?

- **SFT / RWR → MPS ✅.** `train/train_mlx.py` shells to `mlx_lm lora --train --mask-prompt`. Confirmed the
  installed **mlx-lm 0.31.3 ships `gemma4_text` + `linear_to_lora_layers`**, so LoRA can target the E4B arch
  locally (worth one smoke iter to confirm end-to-end). This is the fast local loop.
- **Advantage-weighted gradient has two paths now.** `--backend mlx` is the cheap local iteration route
  (`train_pg_mlx.py` + lockstep generation, no vLLM/CUDA); CUDA/vLLM remains the higher-throughput pod route.
  For Nob specifically, `scripts/local_task_grpo.py --backend mlx` keeps the reward and episode task-local.
- **Generating more Nob data:** MPS works for E4B but slow; vLLM (NVIDIA) is the throughput path. Partial
  rollouts from the Nob entry cut generation cost sharply either way.

---

## Recommended rungs (cheapest signal first)

1. **Rung 1 — RWR-SFT on won-Nob combat decisions, on MPS.** Answers "does concentrating supervision on this
   fight move the needle?" for ~a day and no GPU rental. Clean, trait-neutral reward (attack-forward falls out).
   Deliverables: Nob-fight dataset builder (~100–150 LOC) → `train_policy --backend mlx` → eval the adapter on
   the Nob (paired vs base, HP-lost + win-rate).
2. **Rung 1.5 — Hinted-rollout SFT.** `src/sts_ai/hinting.py` (tactical-truth hints → improved play → launder
   the hint out → SFT) is the existing substitute for logit distillation; an "attack-over-block vs Enraging
   enemies" hint fits its tactical-truth constraint. Breaks the thin-positives ceiling without a teacher model.
3. **Rung 2 — Nob-scoped GRPO** (the RL "learning schedule"). New code: replay-to-Nob `make_env`, combat-scoped
   episode termination + Nob reward, un-gate thinking. ~1–2 days; dry-run per `docs/grpo_dryrun_checklist.md`;
   NVIDIA pod.

## Open questions the side quest may inform
- **Positive-set size:** relax "convincing"→"won", or generate more via partial rollouts? (affects rung 1 data)
- **Thinking vs no-thinking for training:** un-gate native thinking (more faithful to how E4B plays) vs train
  no-thinking (simpler, already supported by PG/GRPO). Affects rungs 2/3.
- **Eval metric for "improved":** win-rate on held-out Nob seeds, mean HP lost, or the attack/block action share
  shifting toward the winning profile. Paired base-vs-adapter via `scripts/compare_paired.py`.
- **Curriculum vs single-fight:** is the Nob a one-off competence fix, or genuinely rung 1 of a per-fight
  schedule (Lagavulin, Sentries, act bosses…)? The replay primitive generalizes to any fight.

## Key file references
- Data: `data/iter2_rwr_hinted/eval/base/vllm_gemma_4_E4B_it_thinking_8192/`
- Analysis/labels/plots: `scratch/nob_fight_analysis.py`, `scratch/plots/`
- SFT: `src/sts_ai/train/sft_format.py` (`build_example`), `scripts/build_sft_dataset.py`, `scripts/train_policy.py`,
  `src/sts_ai/train/train_mlx.py` (MPS), `src/sts_ai/train/train_trl.py` (CUDA)
- RWR: `src/sts_ai/train/reward.py` (`rwr_multiplicities`)
- PG/GRPO: `src/sts_ai/train/pg_dataset.py` (no-thinking gate ~L50), `scripts/build_pg_dataset.py`,
  `scripts/train_pg.py`, `src/sts_ai/train/train_pg_trl.py`, `src/sts_ai/train/grpo_loop.py`,
  `scripts/run_grpo.py`, `scripts/runpod/run_grpo.sh`
- Reward (whole-run only): `src/sts_ai/train/reward.py`
- Mid-game start: `src/sts_ai/interactive/replay.py` (`replay_actions`)
- Teacher analogue: `src/sts_ai/hinting.py`
- Dry-run discipline: `docs/grpo_dryrun_checklist.md`

---

## Execution report (2026-07-07)

### What was implemented

This workstream moved the Nob curriculum from a capability assessment into a detachable local-task lane:

- Added task-local infrastructure under `src/sts_ai/local_tasks/` for replayable fight windows, task reward/labels,
  task-local SFT/RWR data, task-local PG data, and replay-to-fight evaluation.
- Added CLI entrypoints:
  - `scripts/local_task_prepare.py`
  - `scripts/local_task_build_sft.py`
  - `scripts/local_task_eval.py`
  - `scripts/local_task_compare.py`
  - `scripts/local_task_grpo.py`
- Kept Nob-specific logic out of the main full-run training/eval path. Main runs only see a produced adapter via the
  existing adapter surfaces.
- Added local MLX GRPO support for this lane, so Nob-scoped GRPO is no longer CUDA-only for a first local pass.
- Added tests for local-task extraction/replay/reward/dataset behavior.

The first SFT/RWR dataset used only the Gremlin Nob train split:

- Source fights: 43 base E4B Nob fights.
- Split: 32 train / 11 holdout.
- Positive train windows: 18 won or convincing windows.
- Training decisions: 343 unique combat decisions, RWR-resampled to 461 examples.
- Excluded: losses, non-Nob decisions, out-of-combat decisions, and all holdout windows.
- Training type: supervised behavior cloning with RWR replication, not policy-gradient RL.

### Extra issues found and fixed

Two training/eval issues fell out during the first run:

- `vllm-metal` cannot serve a live LoRA adapter on Metal. The local continuous-batching eval path is therefore:
  train MLX LoRA -> `mlx_lm fuse` -> evaluate the fused model with `--backend vllm`.
- The first MLX SFT run was invalid as a native-thinking run. `mlx_lm`'s stock chat dataset calls the Gemma 4 chat
  template without `enable_thinking=True`, which strips `<|channel>thought...<channel|>` from assistant content.
  The adapter therefore learned compact JSON completions despite the source data containing thought-channel
  completions.

Fixes added:

- `train_mlx.prepare_mlx_data` now refuses Gemma thought-channel completions so the unsafe path fails loudly.
- `train_mlx.prepare_native_mlx_data` pre-tokenizes `prompt + completion + "<turn|>\n"` and feeds token ids directly
  to MLX's LoRA trainer, preserving native thought.
- `scripts/train_policy.py --max-seq-len` now defaults to 8192 for MLX SFT.
- Native data prep drops and loudly counts samples that are missing fields, already have truncated thought, would cut
  off inside thought under the sequence cap, or are too long after thought closes.
- The assistant turn terminator `"<turn|>\n"` is included in the supervised target. Without it, the first corrected
  native adapter emitted valid JSON but kept generating until the 8192-token cap.

### Training artifacts

Bad first run, kept for comparison:

- Adapter: `data/local_curricula/gremlin_nob/adapters/rwr_sft_won`
- Fused model: `data/local_curricula/gremlin_nob/models/rwr_sft_won_fused`
- Eval: `data/local_curricula/gremlin_nob/eval/rwr_sft_won_fused_vllm`

Corrected native-thinking run:

- Adapter: `data/local_curricula/gremlin_nob/adapters/rwr_sft_won_native8k_stop`
- Fused model: `data/local_curricula/gremlin_nob/models/rwr_sft_won_native8k_stop_fused`
- Eval: `data/local_curricula/gremlin_nob/eval/rwr_sft_won_native8k_stop_fused_vllm`
- Data prep report: kept 461/461 examples, skipped 0; max total length 3409 tokens, max completion length 2789.
- Training: 200 iterations, batch size 1, learning rate 1e-4; final train loss 0.320, val loss 0.475; 186,663
  trained tokens; peak MLX memory 30.084 GB.
- Smoke check: the fused model emitted Gemma native thought under vLLM (`thinking_tokens=887`,
  `completion_tokens=976`), produced valid JSON, and did not hit the generation cap.

### Holdout eval results

Base vLLM holdout arm:

- Eval dir: `data/local_curricula/gremlin_nob/eval/base_vllm`
- N: 11 windows
- Mean reward: -0.116
- Win rate: 7/11
- Convincing wins: 4/11
- Mean HP loss: 41.4
- Invalid rate: 0

Bad first adapter:

- Mean reward: -0.359
- Win rate: 6/11
- Convincing wins: 1/11
- Mean HP loss: 48.0
- Invalid rate: 0
- Paired reward delta vs base: -0.243 (2 positive / 5 negative / 4 tied, sign-test p=0.453)
- Interpretation: negative result caused by thought stripping in training, not by adapter fusion.

Corrected native-thinking adapter:

- Mean reward: -0.086
- Win rate: 7/11
- Convincing wins: 4/11
- Mean HP loss: 36.5
- Invalid rate: 0
- Paired reward delta vs base: +0.030 (3 positive / 2 negative / 6 tied, sign-test p=1.0)
- Paired HP-loss delta vs base: -4.91 HP (3 improved / 2 worse / 6 tied, sign-test p=1.0)
- Generation health: 215 decisions, 0 invalid turns, 0 cap/truncation hits; mean completion 972 tokens, mean
  thought 880 tokens.

Terminal holdout outcomes for the corrected adapter:

| seed | player HP | Nob state | label |
|---:|---:|---|---|
| 56 | 70/88 | dead | convincing |
| 68 | 24/80 | dead | won |
| 76 | 0/85 | 22/83 alive | loss |
| 88 | 14/80 | dead | won |
| 104 | 0/80 | 29/85 alive | loss |
| 108 | 45/80 | dead | convincing |
| 116 | 0/80 | 1/84 alive | loss |
| 120 | 52/85 | dead | won |
| 128 | 40/80 | dead | convincing |
| 136 | 0/80 | 30/86 alive | loss |
| 144 | 47/80 | dead | convincing |

### Interpretation

The corrected run proves the local SFT/RWR path can train on Gemma native-thinking completions without destroying
thought-channel behavior. The adapter is no longer worse than base on the small holdout and is directionally better on
mean reward and HP loss, but the result is not statistically meaningful: N=11 is too small and six paired windows tied
on reward.

The most important outcome of this workstream is infrastructure and a fixed native-thinking training path, not a
claimed robust Nob competence gain.

### What remains

- Run a lower-variance Nob eval before making a competence claim: more holdout windows, multiple rollouts per window,
  or both.
- Decide whether the next data rung is more partial Nob rollouts, hinted-rollout SFT, or Nob-scoped MLX GRPO.
- If using SFT again, consider adding high-quality corrected/hinted positives rather than only filtering existing
  wins; the current positive set is thin.
- Test whether a Nob adapter composes cleanly with future full-run or other local-task adapters. This work only
  produced a standalone skill patch.
- Do not merge Nob-specific reward into the main training path unless we deliberately promote local curricula into a
  broader schedule. For now, keep this detachable.
- Native MLX SFT logging is minimal because the in-process trainer path bypasses the subprocess CLI and does not wire
  wandb yet.

---

## Retrain + sampled-eval follow-up (2026-07-08)

> Full standalone report: [`nob_rwr_retrain_sampled_eval_2026-07-08.md`](nob_rwr_retrain_sampled_eval_2026-07-08.md).

Follow-up to the two "what remains" items above: retrained at the val-loss optimum and replaced the underpowered
greedy eval. **Conclusion: the RWR/filtered-BC rung is at its ceiling — a properly-trained adapter is
indistinguishable from base on the Nob holdout. The next rung must inject new signal (hinted SFT or shaped-reward
GRPO), not more of the model's own wins.**

### Retrain (undertraining hypothesis: confirmed on loss, not on play)

The first corrected run (200 iters, bs=1) had seen <0.5 epochs of its 461 examples. Retrained identically at
`--iters 1400` (~3.4 epochs): val loss falls 0.450 → **0.396 at iter 1000** (~2.4 epochs), then overfits (0.448 @
1200 while train loss → 0.15). Fused the **iter-1000** checkpoint (plot: `scratch/plots/nob_sft_train_val_loss.png`,
script `scratch/plot_sft_loss.py`; the old 200-iter run ended at val 0.475). Artifacts:
`adapters/rwr_sft_won_native8k_stop_3ep{,_it1000}`, `models/rwr_sft_won_native8k_stop_3ep_it1000_fused`.
Greedy holdout eval of the retrained model: still a wash (−0.141 vs base −0.116, 6/11 wins, 6 exact ties) —
better val loss did **not** translate into better greedy play.

### Eval protocol fix (the "lower-variance eval" item)

- The 2026-07-07 greedy evals ran at **temperature 0.0** (undocumented at the time): 6/11 windows tied because both
  arms play the identical deterministic game. K repeats at temp 0 are a no-op — sampled eval requires temp > 0.
- `local_task_eval.py --rollouts-per-window K`: K sampled episodes per window (`rollout_index = ordinal*K + k`,
  byte-compatible with old dirs at K=1). Both backends now **resume**: completed episodes (meta present) are
  skipped, partial episodes are deleted and re-run.
- `local_task_compare.py` had a real K>1 bug: `_by_window` kept only the lexically-last rollout per window. It now
  averages K rollouts per window before pairing and adds a paired bootstrap 95% CI (`sts_ai.eval_stats`).
- Task metrics (`extra.local_task`) were only injected into episode metas by a post-loop over the *current
  process's* results — a crashed/resumed eval left completed episodes with no reward (compare would silently read
  0.0). Now disk-based (`runner.inject_local_task_meta`) and run over **all** expected episodes. Regression tests
  in `tests/unit/test_local_tasks.py`, `tests/unit/test_local_task_compare.py`.

### Sampled holdout result (temp 0.7, K=4, 44 episodes/arm, paired per-window means)

| arm | mean reward | win rate | convincing | mean HP loss | invalid | attack share |
|---|---|---|---|---|---|---|
| base | −0.252 | 24/44 (55%) | 25% | 43.1 | 0 | 0.245 |
| retrained (it1000) | −0.249 | 23/44 (52%) | 23% | 43.0 | 0 | 0.247 |

Paired (11 windows): reward delta **+0.003, 95% CI [−0.16, +0.18]**, sign +5/−3/=3, p=0.73; HP-loss delta −0.09,
CI [−6.6, +5.0]. Per-window deltas are symmetric shuffling (56: +0.64, 116: +0.31 vs 68: −0.50, 128: −0.39), not
systematic gain. Window structure: 3 windows hopeless for both arms (76/104/136: 0/4 everywhere), 4 near-safe
(88/108/120/128), 4 swing (56/68/116/144) — the swing windows carry essentially all paired sensitivity.
Dirs: `eval/base_vllm_t07_k4`, `eval/rwr_sft_won_native8k_stop_3ep_it1000_fused_vllm_t07_k4`,
`eval/compare_t07_k4_base_vs_3ep_it1000_{reward,hp_loss}.json`.

### Implications for the next rung

- Self-cloning 18 winning windows re-weights the existing policy but does not add capability; more epochs move val
  loss, not play. Rung 1 is **exhausted** — go to hinted-rollout SFT (rung 1.5) or Nob-scoped GRPO (rung 2).
- The 3 hopeless windows are also a GRPO warning: with the cliff reward (all losses = −1.0), K-sample groups there
  have zero advantage variance → no gradient exactly where competence is missing. Add a within-loss tiebreaker
  (e.g. fraction of Nob HP removed — still outcome-based/trait-neutral) before running GRPO.
- Hinting is not yet wired into the local-task lane (the serial runner raises on `hint_cfg`; the vLLM eval path
  doesn't thread it), but the underlying streaming orchestrator supports it — small glue.
