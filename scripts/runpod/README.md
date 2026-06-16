# RunPod A40 Multi-Model Sweep Runbook

This runbook launches the vLLM rollout sweep across two NVIDIA A40 RunPod pods. Pod A and Pod B each run a different model subset, and the local `sync_back.sh` loop keeps copying partial results home so pod crashes or termination do not lose completed rollouts.

## Cost Note

Two A40 pods are typically around `$0.40-$0.50/hr` each on RunPod, so an overnight run should be a few dollars. Confirm live pricing before creating pods:

```bash
runpodctl gpu list
```

## 1. Create Pods

Use the runpod-spinup skill's `create-pod.sh` from your Claude skills directory:

```bash
create-pod.sh podA "NVIDIA A40" SECURE <CUDA-12.x torch template> 1 200
create-pod.sh podB "NVIDIA A40" SECURE <CUDA-12.x torch template> 1 200
```

Before running those commands, confirm the exact A40 GPU id and a CUDA >= 12.1 PyTorch template:

```bash
runpodctl gpu list
runpodctl template search torch
```

Do not use the default `runpod-torch-v21` template for this sweep; it is CUDA 11.8, which is too old for vLLM. Pick a CUDA 12.x torch image/template and verify it with the runpod-spinup skill's `pod-preflight.sh` after creation.

Use about `200GB` of disk. The listed fp16 model weights total roughly `60-80GB`, and vLLM/HuggingFace caches need headroom.

## 2. Clone The Repo

Clone this repo on each pod into:

```bash
/workspace/SlayTheSpireAI
```

For private repos, use the runpod-spinup skill's token-over-stdin workflow. For public repos, normal `git clone` is fine.

## 3. Set Up Each Pod

On each pod:

```bash
ssh runpod-podA
cd /workspace/SlayTheSpireAI
bash scripts/runpod/setup_pod.sh
huggingface-cli login
```

You can export `HF_TOKEN` instead of using `huggingface-cli login`. Gemma and Llama weights are gated, so the token must have accepted licenses for `google/gemma-3-*` and `meta-llama/Llama-3.*`.

## 4. Start The Sweeps

Pod A:

```bash
cd /workspace/SlayTheSpireAI
nohup bash scripts/runpod/run_sweep_on_pod.sh scripts/runpod/models_pod_a.txt data/rollouts/a40_sweep > sweep.out 2>&1 &
```

Pod B:

```bash
cd /workspace/SlayTheSpireAI
nohup bash scripts/runpod/run_sweep_on_pod.sh scripts/runpod/models_pod_b.txt data/rollouts/a40_sweep > sweep.out 2>&1 &
```

The wrapper runs one model per `scripts/run_sweep.py` process. That is deliberate: vLLM does not reliably free GPU memory between in-process model loads. Each process still runs both thinking modes with `--thinking both`, so one model load is reused across the reasoning-off and reasoning-on arms for that model.

The vLLM path uses streaming continuous batching. `CONCURRENCY` defaults to `48`
and caps in-flight rollouts; effective concurrency is
`min(CONCURRENCY, number of seeds per arm)`, so use enough seeds to keep the GPU
busy. Prefix caching is enabled by default.

## 5. Sync Results Locally

Run one sync loop per pod from your dev machine, in separate terminals:

```bash
scripts/runpod/sync_back.sh runpod-podA /workspace/SlayTheSpireAI/data/rollouts/a40_sweep data/rollouts/a40_sweep
scripts/runpod/sync_back.sh runpod-podB /workspace/SlayTheSpireAI/data/rollouts/a40_sweep data/rollouts/a40_sweep
```

If you do not have SSH aliases, quote the raw target:

```bash
scripts/runpod/sync_back.sh "root@203.0.113.10 -p 22022" /workspace/SlayTheSpireAI/data/rollouts/a40_sweep data/rollouts/a40_sweep
```

The sync loop runs every 10 minutes by default and pulls everything under the output directory (per-decision JSONL and per-rollout `.meta.json`) plus `logs/`. Note: in the batched `run_sweep` path a rollout that fails mid-run is recorded inside its own `.meta.json` (`error` / `stopped_reason` fields) — there is no separate `.error.json` sidecar (those are written only by the serial `run_batch.py` path). Rollouts are resumable because `scripts/run_sweep.py` skips any rollout whose `.meta.json` already exists under the output directory.

## 6. Teardown

Only terminate pods after `sync_back.sh` confirms the data is local. Terminating a RunPod pod deletes its disk, so do not clean up before the rollout directories and logs have been copied back.

Then use the runpod-spinup cleanup helper:

```bash
cleanup-pod.sh <pod-id>
```

## Sweep Matrix

The sweep is `9 models x {reasoning-off, reasoning-on} = 18 arms`.

For Qwen3, reasoning-on uses the native `<think>` toggle. For Gemma and Llama, reasoning-on uses prompted reasoning. Each rollout records the active mode in its `.meta.json` under `extra.agent_config.reasoning_mode`.

## Gotchas (learned bringing up the first A40 pod, 2026-06-16)

- **vLLM version must be new enough for the models.** vLLM 0.6.6 (the original pin) only knows `Qwen2`/`Gemma2`; it errors `architectures ['Qwen3ForCausalLM'] are not supported` on Qwen3 and Gemma3. Use **vLLM ≥ 0.23** (the pinned `[vllm]` extra) — it pulls a CUDA-13 torch (2.11) that needs a recent driver (the A40 hosts ship CUDA 13.0, fine).
- **flashinfer JIT-compiles kernels at runtime and needs `ninja`.** Without it, engine init dies with `FileNotFoundError: 'ninja'`. `setup_pod.sh` now installs `ninja-build`; the first model load also pays a one-time kernel-compile cost (cached afterwards).
- **The simulator build needs a Python-3.11-compatible pybind11.** Upstream pins pybind11 v2.7.1 which fails to compile against CPython 3.11 (`PyFrameObject` is opaque). `build_lightspeed.sh` now bumps the submodule to v2.13.6 after `submodule update`.
- The RunPod image has **no `rsync`** by default (needed for repo transfer and `sync_back.sh`); `setup_pod.sh` installs it.
