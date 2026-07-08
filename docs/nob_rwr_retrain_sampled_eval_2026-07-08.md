# Nob RWR-SFT retrain + sampled holdout eval — the BC rung closes as a null

**Date:** 2026-07-08 · **Status:** complete · **Branch:** `feat/rwr-and-hinted-rollouts`
**Context:** follow-up to [`nob_curriculum_next_steps_2026-07-06.md`](nob_curriculum_next_steps_2026-07-06.md)
(the local-curriculum lane and the first, inconclusive RWR-SFT result). This session answered its two open
eval/training questions: *was the first adapter undertrained?* and *what does a properly-powered holdout eval say?*

## TL;DR

**The RWR/filtered-BC rung is exhausted.** Trained to its validation-loss optimum and evaluated with a sampled,
paired, K-per-window protocol, the Gremlin Nob adapter is statistically indistinguishable from base
(paired reward delta **+0.003, 95% CI [−0.16, +0.18]**, p=0.73). Better val loss (0.475 → 0.396) did not translate
into better play: self-cloning the model's own 18 winning fight windows re-weights the existing policy but injects
no new capability. The next rung must add new signal — hinted-rollout SFT, or Nob-scoped GRPO with a reward that
still discriminates on all-loss windows.

Along the way the eval path gained the machinery a defensible claim needed (K sampled rollouts per window, resume,
per-window-mean paired compare with bootstrap CI), and three real bugs were found and regression-tested.

## 1. Retrain: undertraining confirmed on loss, refuted as the explanation for the null

The 2026-07-07 adapter (`rwr_sft_won_native8k_stop`) had trained 200 iters at bs=1 over 461 examples — under half
an epoch. Retrained with the identical config at `--iters 1400` (~3.4 epochs), val every 200 iters, checkpoints
every 100 (`adapters/rwr_sft_won_native8k_stop_3ep`):

| iter | 1 | 200 | 400 | 600 | 800 | **1000** | 1200 | 1400 |
|---|---|---|---|---|---|---|---|---|
| val loss | 0.450 | 0.492 | 0.430 | 0.402 | 0.398 | **0.396** | 0.448 | 0.415 |

Clear elbow at ~2.4 epochs (train loss keeps falling to 0.15–0.18 → memorization). Fused the **iter-1000**
checkpoint → `models/rwr_sft_won_native8k_stop_3ep_it1000_fused`. Plot:
`scratch/plots/nob_sft_train_val_loss.png` (script `scratch/plot_sft_loss.py`; the 200-iter run ended at val 0.475).

Greedy holdout check of the retrained model (temp 0.0, N=11, protocol identical to the 2026-07-07 arms): still a
wash — mean reward −0.141 vs base −0.116, wins 6/11 vs 7/11, **6/11 windows exactly tied**. Fitting the winning
trajectories better does not change greedy play.

## 2. Eval protocol: from underpowered greedy to sampled + paired

Three findings, all fixed and regression-tested:

1. **The 2026-07-07 greedy evals ran at temperature 0.0** (unstated in the original writeup). Ties are structural:
   both arms play the same deterministic game per window unless the adapter changes the argmax path. K repeats at
   temp 0 are a no-op, so "multiple rollouts per window" required temp > 0 first.
2. **`local_task_compare.py` had a K>1 bug** — `_by_window` kept only the lexically-last rollout per window,
   silently discarding the rest. Now averages K rollouts per window before pairing and reports a paired bootstrap
   95% CI (`sts_ai.eval_stats`). Reproduces the old K=1 numbers exactly.
3. **Episode task metrics were injected fragilely.** `extra.local_task` (reward/label) was added to episode metas
   by a post-loop over the current process's in-memory results only — a crashed-and-resumed eval left completed
   episodes without task reward, which the compare read as 0.0 **silently**. Injection is now disk-based
   (`local_tasks/runner.inject_local_task_meta`, reconstructs metrics from the episode's own JSONL) and covers
   every expected `(window, rollout_index)` pair.

New capability: `local_task_eval.py --rollouts-per-window K` (indices `ordinal*K + k`; byte-compatible naming at
K=1) with resume on both backends (completed episodes skipped, partial ones deleted and re-run). Tests:
`tests/unit/test_local_task_compare.py`, `tests/unit/test_local_tasks.py::InjectLocalTaskMetaTest`.

## 3. Sampled holdout result (temp 0.7, K=4 → 44 episodes/arm)

| arm | mean reward | win rate | convincing | mean HP loss | invalid | attack share |
|---|---|---|---|---|---|---|
| base | −0.252 | 24/44 (55%) | 25% | 43.1 | 0 | 0.245 |
| retrained (it1000) | −0.249 | 23/44 (52%) | 23% | 43.0 | 0 | 0.247 |

Paired per-window means over 11 windows:

- **Reward: +0.003, 95% CI [−0.16, +0.18]**, sign test +5/−3/=3, p = 0.73.
- **HP loss: −0.09, 95% CI [−6.6, +5.0]**, sign test +4/−4/=3, p = 1.0.

Per-window deltas are symmetric shuffling, not improvement: gains on seed 56 (+0.64) and 116 (+0.31) offset by
losses on 68 (−0.50) and 128 (−0.39). The action profile is unmoved.

**Window structure** (base wins out of 4): three windows are hopeless (76, 104, 136 — 0/4 for *both* arms), four
near-safe (88, 108, 120, 128), four genuine swing windows (56, 68, 116, 144 — 1–3 wins). Greedy eval had
mislabelled some of these (144 looked like a solid convincing win at temp 0; it is a coin flip).

Artifacts (local, `data/` is gitignored): `data/local_curricula/gremlin_nob/eval/base_vllm_t07_k4/`,
`.../rwr_sft_won_native8k_stop_3ep_it1000_fused_vllm_t07_k4/`,
`.../compare_t07_k4_base_vs_3ep_it1000_{reward,hp_loss}.json`, driver log
`.../logs/sampled_eval_driver_20260707.log`.

## 4. Interpretation

This is a **method null, not an execution null**: the two standard rescues (train longer, evaluate with more
power) were both applied and the effect stayed at zero with a CI tight enough to exclude anything usable
(|Δreward| ≥ 0.18 ≈ ≥7 HP saved per fight). Filtered-BC/RWR on the model's own wins teaches the model what it
already does when it wins — it cannot teach the *policy change* (attack-forward against Enrage) that separates its
wins from its losses, because the losses are exactly what gets filtered out.

## 5. Consequences for the next rung

1. **Hinted-rollout SFT (rung 1.5) is the natural next step.** The corrective signal is known and specific
   (turtling feeds Enrage; attack-forward wins) — `hinting.py` injects it directly instead of waiting for lucky
   wins to clone. Gap: the local-task lane doesn't thread `hint_cfg` yet (the serial runner raises on it; the vLLM
   eval path doesn't pass it) — small glue over the existing streaming hint machinery.
2. **GRPO (rung 2) needs a reward change first.** The three hopeless windows produce all-loss K-sample groups
   under the cliff reward (every sample −1.0) → zero group-advantage variance → no gradient precisely where
   competence is missing. Add a within-loss tiebreaker (e.g. fraction of Nob HP removed — outcome-based, so still
   trait-neutral) to `GremlinNobTask.reward_from_metrics` before any GRPO run.
3. **Eval discipline to keep:** sampled paired K-per-window is now the standard for any local-task competence
   claim; greedy K=1 is a smoke check only. The four swing windows carry most of the sensitivity — more source
   windows (new seeds) would broaden the base, at the cost of generating fresh full rollouts.

## 6. Session log (for reproducibility)

1. Retrain: `PYTHONPATH=src .venv/bin/python scripts/train_policy.py --backend mlx --base-model
   mlx-community/gemma-4-e4b-it-bf16 --dataset data/local_curricula/gremlin_nob/datasets/train_won_rwr.jsonl
   --out data/local_curricula/gremlin_nob/adapters/rwr_sft_won_native8k_stop_3ep --iters 1400 --batch-size 1
   --learning-rate 1e-4 --max-seq-len 8192`
2. Fuse iter-1000: copy `0001000_adapters.safetensors` + `adapter_config.json` to a checkpoint dir → `mlx_lm fuse`.
3. Evals (vllm-metal venv): `PYTHONPATH=src ~/.venv-vllm-metal/bin/python scripts/local_task_eval.py --task
   gremlin_nob --manifest data/local_curricula/gremlin_nob/manifests/source.json --split holdout --backend vllm
   --max-tokens 8192 --top-p 0.95 --top-k 64 --thinking` + per-arm `--model/--output-dir`, greedy (`--temperature
   0.0`) or sampled (`--temperature 0.7 --rollouts-per-window 4`).
4. Compare: `PYTHONPATH=src .venv/bin/python scripts/local_task_compare.py --base <base_dir> --trained
   <trained_dir> --metric {reward,hp_loss}`.
