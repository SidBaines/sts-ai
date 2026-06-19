# GRPO-Readiness Handover

**For:** a planning/orchestration agent that will scope and drive the implementation.
**Date:** 2026-06-19. **Branch:** `feat/rwr-and-hinted-rollouts`.

This lists **what needs to be achieved and why** to make the harness ready to run
**GRPO** (on-policy, group-relative policy optimization). It is not a full plan —
scope the *how*, sequencing, and hyperparameters yourself. Read
[`docs/research_plan.md`](research_plan.md) (near-term step 4 + Stage 9) and
[`docs/rl_and_framing_design.md`](rl_and_framing_design.md) first.

## Goal

The repo studies **how training-time framing redirects what an LLM generalizes from
the same reward**. The RL ladder is: **offline filtered-BC → RWR → on-policy GRPO/RLOO**
(trait-neutral reward). We are mid-way through the first rung. The immediate goal of
this handover: **get everything in place so GRPO can start as soon as the current
offline run finishes** — i.e. close the offline loop, build the missing measurement +
algorithm pieces, and validate them with the discipline that this codebase demands.

Net assessment going in: we are **infra-ready** for GRPO (generation/seeding/streaming
are built and now GPU-proven) but **not result-ready or implementation-ready** (no
offline result yet; no GRPO loss/loop; eval too noisy to steer an optimizer).

## Current state (as of this handover)

- **Offline RWR + hinted run is training right now** on RunPod H100 `8qllohk2hiado9`
  (pod `20260619_SidsH100_iter2b`), step ~1165/2000, ~1h15m left, then it auto-runs
  eval (base vs trained on the frozen eval split, seeds 47–151) and a compare, then a
  Mac-side watchdog stops the pod. Artifacts sync to `data/iter2_rwr_hinted/`
  (`train_rollouts/`, `sft/`, `adapter/`, `eval/`, `logs/`). **No eval result yet.**
- **What's validated this session:** vLLM full-game generation at scale (native Gemma-4
  thinking + `--hints on` both fired correctly on GPU); the full TRL LoRA training path;
  the seeding + streaming orchestrator. See the bug list under "Gotchas."
- **Uncommitted repo fixes** (verify + commit): `src/sts_ai/train/train_trl.py`,
  `scripts/train_policy.py`, `pyproject.toml`, `scripts/runpod/{run_iter2_pipeline,
  run_iter2_resume,iter2_status,run_with_autostop}.sh`.

## What needs to be achieved (deliverables — order is roughly dependency order)

1. **Land & interpret the offline result (the gating signal).**
   *Goal:* decide whether our reward (final floor) + pipeline + eval can detect a real
   improvement at all. *Done when:* base-vs-trained on the frozen eval split is reported
   as a **paired per-seed floor delta + sign test, beside the `agent_invalid` rate**, and
   a go/no-go is recorded. If trained doesn't beat base, fix the offline arm before any
   on-policy work — GRPO cannot rescue a reward/eval that can't detect signal.

2. **Lower-variance eval harness (prerequisite for steering any optimizer).**
   *Goal:* K rollouts/seed on the frozen eval split with a paired significance test, not
   single-sample (the offline signal was ~1 SE at N≈60 — too noisy to optimize against).
   *Done when:* a reusable eval reports a CI / significance on the floor delta and basic
   hack-detection metrics (e.g. decisions/run, stall rate). Infra exists: `run_until.py`
   already keys on `(world_seed, rollout_index)`; needs K-per-seed + aggregation.

3. **(Stepping stone) signed-advantage offline PG on the *same* data.**
   *Goal:* a low-risk test that an advantage-shaped objective beats RWR *before* paying
   for the on-policy loop. Replace RWR's `exp(floor/β)` resampling with
   `advantage = floor − baseline` (push below-baseline trajectories *down*). Start from
   the **raw rollouts + per-trajectory `final_floor`**, NOT the built SFT dataset (RWR
   already dropped/replicated it). *Done when:* trains end-to-end and is eval'd vs RWR
   and base; decision recorded.

4. **GRPO implementation (the core build).**
   *Goal:* on-policy group-relative PG. Components to design/build:
   (a) **regenerate K rollouts per world seed** at temp>0 from the *current* policy via
   the **streaming orchestrator** (`run_streaming_rollouts` / `run_until.py`) — non-blocking
   is a durable commitment (rl_and_framing_design §2b); lockstep MLX is forbidden here;
   (b) **group-relative advantage** from trajectory floors (group = K games of one seed);
   (c) **clipped PG loss** over the decision/completion tokens with **KL-to-frozen-reference**;
   (d) **iterate** — regenerate groups every N updates to stay near-on-policy.
   *Done when:* a dry-run-validated loop runs several iterations stably and eval (deliverable 2)
   shows it ≥ the offline baseline. Expect on-policy failure modes (KL collapse, reward
   hacking, length blowup, advantage-norm instability).

5. **Reward sanity / anti-hacking pass.**
   *Goal:* decide whether final-floor suffices or needs shaping, and enumerate exploit
   modes an on-policy optimizer will find (stalling to delay death, sim quirks, degenerate
   loops) + guardrails. Reward must stay **trait-neutral** (the framing invariant — see
   constraints). *Done when:* a reward spec + failure-mode list is documented and the eval
   surfaces the relevant hack metrics.

6. **Validation discipline (cross-cutting, non-negotiable).**
   Every new GPU code path gets a **dry-run that mirrors the EXACT production flags** before
   the real run, and a regression test for every bug fixed. This session lost a full
   training launch because a dry-run omitted `--wandb-project`/`--manifest` and a latent
   crash slipped through. Cheap dry-runs >> multi-hour failed GPU cycles.

## Hard constraints / invariants (must respect)

- **Prompt-neutrality is the independent variable.** The base prompt is comprehension-only;
  risk/strategy/objective framing is the experimental variable and must never be baked in.
  Hints (and any reward shaping) must stay **tactical-truth / trait-neutral** so the data and
  reward remain reusable across framing conditions. (`prompt-neutrality` memory; CLAUDE.md.)
- **On-policy generation uses the streaming orchestrator, never the MLX lockstep path**
  (durable commitment; lockstep idles the GPU on each group's slowest straggler).
- **Reproducibility + schema stability are a contract once data is kept** — `schemas.py`
  is the on-disk format; frozen seeds must not silently change; flag any intentional change.
- **Local training is a benchmark, not an assumption** — the CUDA H100 path is primary.
- **Read outcomes beside the invalid/`stopped_reason` rate** — a rollout *stops* on an
  unrecoverable invalid decision, so floor/win-rate partly reflect format compliance
  (rl_and_framing_design §gotcha 1).

## Gotchas earned this session (will save the next agent hours)

- **Working GPU stack (verified):** CUDA-13 H100 + `vllm==0.23.0` (pulls cu130 torch
  2.11 + transformers 5.12.1) + `trl 1.6.0` + `peft 0.19` + `wandb`. Install `.[train-cuda]`
  with a constraints file pinning `transformers==5.12.1 torch==2.11.0 vllm==0.23.0` so the
  resolver can't disturb the vLLM stack. Pod needs a **CUDA-13-filtered** host (`create-pod-cuda.sh … 13.0`);
  default H100 hosts came up CUDA-12.8 and broke vLLM. Set `VLLM_USE_FLASHINFER_SAMPLER=0`
  (flashinfer JIT needs nvcc the pod lacks).
- **Gemma-4 E4B is a VLM architecture** → TRL **blocks `assistant_only_loss`** for VLMs.
  Use **`completion_only_loss=True` over the `{prompt, completion}` pair** (sft_format emits
  both `messages` and prompt/completion; the prompt is pre-chat-templated, completion is the
  verbatim `raw_response`). Same prompt-masked loss, VLM-safe. Drop the `messages` column so
  TRL doesn't re-detect conversational format.
- **`dataset_builder.discover_rollouts` globs `seed_*_r*.jsonl` non-recursively** → point
  `--rollout-dir` at the agent-label leaf dir (`…/vllm_gemma_4_E4B_it_thinking_8192`), not
  the parent.
- **TRL renamed `max_seq_length` → `max_length`** (trl ≥ ~0.20).
- **The built SFT dataset is the wrong input for PG/GRPO** — RWR already filtered/replicated.
  Start from raw rollouts + per-trajectory `final_floor` (in each `*.meta.json`).
- **RunPod teardown:** the pod-injected `RUNPOD_API_KEY` is restricted (can't `podStop`), and
  the safety layer (correctly) blocks persisting the master account key on a rented pod.
  Drive stop/teardown from the controller (`runpodctl pod stop/delete`) — e.g. a Mac-side
  watchdog that stops the pod when the pipeline process exits. (`run_with_autostop.sh` exists
  but is unused for that reason.)
- **Generation already supports GRPO groups:** `--rollouts-per-seed K`, per-`(world_seed,
  rollout_index)` seeding (`seeding.py`), and the streaming orchestrator. The current run
  used K=1; GRPO needs K>1 regenerated from the current policy.

## Pointers

- Plan/ordering: [`research_plan.md`](research_plan.md) (near-term step 4, Stage 9),
  [`rl_and_framing_design.md`](rl_and_framing_design.md) (§2b generation path, §2c seeding/groups, gotcha 1).
- Prior offline run writeup: [`progress_report_2026-06-18.md`](progress_report_2026-06-18.md).
- Training pipeline: `src/sts_ai/train/` (`reward.py` — RWR + labels; `dataset_builder.py`;
  `sft_format.py` — skew-free prompt/completion; `train_trl.py`). Drivers:
  `scripts/{build_sft_dataset,train_policy}.py`.
- Generation/seeding: `src/sts_ai/{streaming_rollout,parallel_rollout,seeding,hinting}.py`,
  `scripts/run_until.py`. RunPod: `scripts/runpod/` (`setup_pod.sh`, `run_iter2_resume.sh`,
  `iter2_status.sh`).

## Out of scope for now / decisions left to the planning agent

GRPO hyperparameters (group size K, KL coefficient, clip range, regenerate cadence, lr);
GRPO vs RLOO choice; reward-shaping specifics; and the downstream **framing-variant**
experiments (a later stage — keep the data/reward neutral so they stay possible).
