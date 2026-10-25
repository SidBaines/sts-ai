# Running K2-Horizon-0.9B on Sid's laptop

**Status:** finished 25 October 2026. **Verdict:** an MLX conversion runs the
same model on the laptop. Grouped generation and checkpointed LoRA updates are
fast enough for the teacher-feedback pilot. Mixed-length batching is about 4×
slower than RTX vLLM serving.

## Question

Can the planned training run on Sid's MacBook (Apple M5 Pro, 48 GB), and how
do its generation and update speeds compare with the RTX 2000 Ada
measurements of [23 October](2026-10-23-llm-throughput.md)? See the approach in
[strategies](../strategies.md#running-on-the-laptop-with-mlx).

## Design

- **Model:** the same model at the same revision. The Hugging Face config
  turned out to be a plain Llama layout with YaRN position scaling, so mlx-lm
  loads the unchanged BF16 weights with a Llama config. The check behind this
  is in `mlx_model.py`.
- **Correctness check:** saved completions are scored with the original model
  in FP32 on the CPU, and the MLX model must match it.
- **Benchmarks** repeat the RTX ones where possible:
  - Generation: the 64-state panel at temperature 0.6, top-p 0.95 and an
    8,192-token cap, using mlx-lm's continuous batch generator, with 1, 8 or
    32 sequences at once.
  - A pilot-shaped case: 8 states × 8 samples, top-p 1, 4,096-token cap.
  - The equal-work probe: 64 requests of 2,048 input and 256 output tokens.
  - Updates: the same eight saved completions, effective batch four, one
    warmup and three timed AdamW steps, LoRA rank 16 and alpha 32 on every
    projection. This gives 13,991,936 trainable parameters, identical to the
    PEFT setup.
  - The existing PyTorch trainer, run unchanged on Apple's GPU (MPS).
- **Budget and approval:** Sid asked for these benchmarks on 25 October, with
  about one hour of laptop time. All timings are single cohorts.

## Results

**The conversion is exact up to precision.** Teacher-forced scoring of three
saved completions, compared with the FP32 original:

| Condition | Mean absolute token log-probability difference | Greedy next-token agreement |
|---|---:|---:|
| MLX, FP32 | 0.0013 | 100% |
| MLX, BF16 | 0.017 | 98.1% |
| Original model, BF16 on MPS | 0.019 | 98.5% |

**Generation is fast when each batch is one state's group, and slow when
prompt lengths are mixed.** Medium effort unless noted:

| Case | Mac valid decisions/s | Mac tokens/s | Mac peak GB | RTX vLLM valid decisions/s |
|---|---:|---:|---:|---:|
| 64 states, 1 at a time | 0.22 (61/64; one 8,192-token cap) | 97 | 3.4 | 0.29 |
| 64 states, 8 at a time | 0.34 | 113 | 12.4 | 1.53 |
| 64 states, 32 at a time | 0.42 | 147 | 33.0 | 1.83 (FlashInfer 2.08) |
| Low effort, 32 at a time | 1.72 | 20 | 30.0 | 5.59 |
| High effort, 32 at a time | 0.43 | 170 | 33.4 | 0.50 (FlashInfer 1.69) |
| Pilot shape, 8 × 8, mixed batch of 32 | 0.50 | 157 | 29.1 | — |
| Pilot shape, 8 × 8, one group of 8 at a time | **1.07** | **326** | 4.3 | — |
| Hugging Face path on MPS, 2 × 8, microbatch 8 | 0.24 | 80 | — | 0.33 (RTX Hugging Face path) |

**On equal work, the hardware gap is modest.** In the equal-work probe at 32
at once, the Mac reached 451 output tokens/s against 654 for RTX FlashAttention
(0.69×). At 8 at once it reached 337. Run alone, the Mac reads prompts at
about 7,600 tokens/s and decodes at 113 tokens/s.

The slowdown on natural prompts comes from mixing prompt lengths in one batch:
- A single same-prompt group decoded at 309–341 tokens/s, against 113 tokens/s
  for a mixed batch of 8, and used under a third of the memory.
- Greedy sampling and larger prompt batches changed the low-effort case only
  by 4–10%.
- Training groups share a prompt, so they avoid the problem.

**Checkpointed MLX updates match RTX speed.** Eight saved completions, with
prompts of 1,284–2,591 tokens and completions of 80–743 tokens:

| Update condition | Mac completions/s | Mac peak GB | RTX completions/s (peak GiB) |
|---|---:|---:|---:|
| MLX, checkpointing, microbatch 1 | **0.74** | 4.4 | 0.76 (2.9), PyTorch SDPA |
| MLX, no checkpointing, microbatch 1 | 0.50; repeat 0.60 | 13.6–13.9 | 1.03 (11.7) |
| MLX, no checkpointing, microbatch 2 | 0.52 | 24.8 | 0.57 with checkpointing (3.7) |
| Existing PyTorch trainer on MPS, checkpointing, microbatch 1 | 0.20 | — | — |

Scoring the old and reference policies costs 0.31–0.44 s per completion per
pass on MLX, against 0.40 s on RTX and 0.50 s on MPS.

**Estimated pilot iteration (4 states × 8 samples, medium):** about 30 s of
grouped generation, 20 s of scoring and 43 s of updates, so about 95 s, or
roughly 38 iterations an hour. The RTX Hugging Face path works out at about
165 s per iteration. These are sums of component rates, not a measured
end-to-end run.

Teacher scores in the Mac runs (0.26–0.33 at medium, 0.31 at low and 0.27 at
high) are within the RTX ranges. They are independent samples, because the RTX
runs shared per-request seeds.

## Conclusion

The laptop can run the teacher-feedback pilot under MLX. Its sound settings
are one state's group per generation batch and checkpointed updates with
microbatch one. Its estimated iteration rate beats the RTX Hugging Face
training path, though not RTX vLLM serving. The existing PyTorch trainer runs
on MPS but is 3.7× slower than MLX at updates, so the laptop path needs an
MLX trainer.

Caveats:
- Each timing is a single cohort.
- Peak memory under mixed batching (29–35 GB) is high for a 48 GB machine
  shared with other work.
- The fix agents' unit tests ran during the first update case.
- The parity check used three completions.
- The end-to-end rate is an estimate.

## Next

Write a small MLX version of the teacher-feedback loop:
- grouped generation;
- old-policy scores taken from the sampling pass, where they match;
- checkpointed updates;
- adapter checkpoints readable by PEFT.

Then use it for stages 0–1 of [the proposed plan](../brainstorms/2026-10-25-getting-the-model-to-learn.md).

## Log

- 25 October: the conversion check passed first time; mixed-length batching
  was identified as the generation bottleneck, and grouped generation was
  added and measured.

Code: `experiments/2026-10-25-mac-local/`, plus the MPS timing fix in
`src/sts_llm/training.py`. Data: run `mac_local_20261025` (local). The
converted model directory is `data/models/k2-horizon-0.9b-mlx`.
