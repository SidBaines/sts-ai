# Choosing settings for a small reasoning game player

**Status:** finished 23 October 2026. **Verdict:** batching and repeated-state caching help; effort and numerical settings need a learning study.

## Question

Which generation and update settings make the first teacher-feedback RL study practical? See [the approach](../strategies.md#small-model-serving-before-training).

## Design

Pin K2-Horizon-0.9B in BF16 on RTX 2000 Ada and use environment release 0.1.5.1. The primary panel has 64 states from 55 worlds, 40 combat and 24 out-of-combat, across Acts 1–3. All three efforts use the same states, temperature 0.6, top-p 0.95 and an 8,192-token cap. Keep raw responses and all invalid/capped outputs.

Screen concurrency 1/8/32, attention backend, scheduler budget and graph execution. Test prefix caching on/off with one or four responses per state, with fresh servers and disjoint warmups. Add equal-token synthetic probes to separate execution speed from differing output lengths. The screen is adaptive: every comparison is exploratory. Prefer usable completed decisions, then check length, legality, memory and repeat variability.

Benchmark the actual Hugging Face and PEFT training path separately: two mixed-phase states, eight samples each, top-p 1.0, and generation microbatches 1/4/8. Real update tests use eight saved completions, effective batch four, one warmup and three timed AdamW steps, with synthetic advantages. A bounded real-teacher smoke and checkpoint continuation test the plumbing; no reasoning SFT or main learning study is run.

Budget: at most three RTX hours at the allocated $0.24/hour, plus storage. Sid approved speed checks and code preparation on 23 October. The earlier A40 stopped before any comparable timing completed.

## Results

All timings below are descriptive single cohorts; no timing confidence interval is justified. Full warmup, calibration, failure and mechanics rows are in [the CSV ledger](../figures/llm-throughput/serving.csv).

| Serving condition | Effort | Concurrency | Valid / attempted | Valid decisions/s | Output tokens/s | Capped |
|---|---|---:|---:|---:|---:|---:|
| FlashAttention, graphs, 2,048 | medium | 32 | 64 / 64 | 1.48 | 520.3 | 0 |
| FlashAttention, graphs, 8,192 | high | 1 | 62 / 64 | 0.22 | 91.1 | 0 |
| FlashAttention, graphs, 8,192 | high | 32 | 62 / 64 | 0.50 | 266.9 | 1 |
| FlashAttention, graphs, 8,192 | high | 8 | 62 / 64 | 0.43 | 237.6 | 1 |
| FlashAttention, graphs, 8,192 | low | 1 | 64 / 64 | 3.55 | 54.0 | 0 |
| FlashAttention, graphs, 8,192 | low | 32 | 64 / 64 | 5.59 | 80.2 | 0 |
| FlashAttention, graphs, 8,192 | low | 8 | 64 / 64 | 5.61 | 82.7 | 0 |
| FlashAttention, graphs, 8,192 | medium | 1 | 64 / 64 | 0.29 | 90.9 | 0 |
| FlashAttention, graphs, 8,192 | medium | 32 | 63 / 64 | 1.83 | 599.8 | 0 |
| FlashAttention, graphs, 8,192 | medium | 8 | 64 / 64 | 1.53 | 408.6 | 0 |
| FlashAttention, graphs, 8,192 (repeat) | medium | 32 | 63 / 64 | 1.59 | 521.8 | 0 |
| FlashAttention, no graphs, 8,192 | medium | 32 | 64 / 64 | 1.54 | 491.4 | 0 |
| FlashInfer, graphs, 8,192 | high | 32 | 63 / 64 | 1.69 | 662.4 | 0 |
| FlashInfer, graphs, 8,192 | low | 32 | 64 / 64 | 4.64 | 64.0 | 0 |
| FlashInfer, graphs, 8,192 | medium | 32 | 64 / 64 | 2.08 | 680.7 | 0 |


The repeated FlashAttention medium cohort reproduced all 64 token sequences, but took 15% longer. Backend changes also changed sampled outputs: FlashInfer high generated 25% fewer tokens and avoided an 8,192-token tail. Its apparent 3.36-fold decision-rate advantage is therefore not a pure hardware gain.

| Equal-token condition, 64 × (2,048 input + 256 output), concurrency 32 | Output tokens/s | Ratio to FlashAttention graphs 8,192 |
|---|---:|---:|
| FlashAttention, graphs, 2,048 scheduler tokens | 626.0 | 0.957 |
| FlashAttention, graphs, 8,192 scheduler tokens | 654.5 | 1.000 |
| FlashAttention, no graphs, 8,192 scheduler tokens | 652.6 | 0.997 |
| FlashInfer, graphs, 8,192 scheduler tokens | 692.1 | 1.058 |


All four fixed-work probes completed 64/64 requests. FlashInfer was 5.8% faster in this screen; graph removal and a smaller scheduler budget showed no gain.

| Medium prefix caching, concurrency 32 | Off valid decisions/s | On valid decisions/s | On prompt-token hit fraction | Valid off / on |
|---|---:|---:|---:|---:|
| One response per state (64 total) | 2.08 | 1.83 | 2.4% | 64 / 63 |
| Four responses per state (256 total) | 2.17 | 3.27 | 75.2% | 254 / 254 |


Four-response caching raised decision throughput 50.6% and token throughput 41.4%, while generating 6.1% fewer tokens. Distinct states showed no gain. Only 39 initial tokens were common to all prompts (mean length 1,728); repeated rules later in the text do not form a reusable prefix. Cache changes also changed sampled outputs.

| Actual HF generation | Microbatch | Responses | Seconds | Valid decisions/s | Output tokens/s | Valid fraction |
|---|---:|---:|---:|---:|---:|---:|
| medium (float32) | 4 | 8 | 98.19 | 0.081 | 26.2 | 1.000 |
| high (bfloat16) | 1 | 16 | 278.29 | 0.057 | 18.3 | 1.000 |
| high (bfloat16) | 4 | 16 | 545.03 | 0.028 | 24.9 | 0.938 |
| high (bfloat16) | 8 | 16 | 343.72 | 0.047 | 29.7 | 1.000 |
| low (bfloat16) | 1 | 16 | 23.33 | 0.686 | 15.8 | 1.000 |
| low (bfloat16) | 4 | 16 | 11.29 | 1.418 | 22.5 | 1.000 |
| low (bfloat16) | 8 | 16 | 5.82 | 2.748 | 28.5 | 1.000 |
| medium (bfloat16) | 1 | 16 | 327.98 | 0.049 | 18.3 | 1.000 |
| medium (bfloat16) | 4 | 16 | 104.26 | 0.153 | 52.6 | 1.000 |
| medium (bfloat16) | 8 | 16 | 49.14 | 0.326 | 90.9 | 1.000 |
| Medium BF16, static cache and compilation | 8 | — | 240-second deadline | — | — | — |


The FP32 feasibility probe uses eight responses, a 1,024-token cap and no warmup; it is not a matched speed ratio against the 16-response screen. High effort produced a 5,520-token tail at microbatch eight and an 8,192-token capped response at microbatch four. The compiled static-cache case took 56 seconds for cold setup and 143 seconds for warmup, then exceeded its 240-second deadline without a complete timed cohort. It is not a demonstrated speedup.

| Real update condition | Responses/s | Output training tokens/s | Peak allocated GiB | Result |
|---|---:|---:|---:|---|
| Eager attention, checkpointing on, microbatch 1 | 0.25 | 86.0 | 7.08 | completed |
| SDPA, checkpointing on, microbatch 1 | 0.76 | 255.7 | 2.91 | completed |
| SDPA, checkpointing on, microbatch 2 | 0.57 | 192.5 | 3.68 | completed |
| SDPA, checkpointing on, microbatch 4 | 0.46 | 156.5 | 5.85 | completed |
| SDPA, checkpointing off, microbatch 1 | 1.03 | 346.4 | 11.68 | completed |
| SDPA, checkpointing off, microbatch 2 | — | — | — | Out of memory |


Every successful update case changed adapter weights during its three timed steps. Larger checkpointed microbatches were slower. Removing checkpointing improved speed but used 11.68 GiB; microbatch two then ran out of memory. These completions reached about 3,062 total tokens, so the test does not certify long high-effort updates.

Teacher preference did not improve with reasoning effort. At concurrency 32, the paired differences below use all 64 states and bootstrap whole source worlds (55 clusters). They are diagnostic preferences, not win probability or reasoning quality. All other paired diagnostics, including null results, are in [the complete comparison CSV](../figures/llm-throughput/teacher_comparisons.csv).

| Backend | Effort comparison | Difference in mean teacher probability | 95% interval |
|---|---|---:|---|
| FlashAttention | high minus low | -0.098 | [-0.185, -0.012] |
| FlashAttention | high minus medium | -0.038 | [-0.113, 0.033] |
| FlashAttention | medium minus low | -0.060 | [-0.134, 0.011] |
| FlashInfer | high minus low | -0.129 | [-0.204, -0.064] |
| FlashInfer | high minus medium | -0.067 | [-0.134, -0.005] |
| FlashInfer | medium minus low | -0.061 | [-0.138, 0.008] |


Mechanics checks: 52 library tests pass with real Torch dependencies. The environment suite ran 266 tests: 247 passed and 19 optional tests skipped, and its random and base-model CLI traces replayed. The trained adapter also passed a three-decision clean-CLI smoke and fresh-process replay across both phases. Two teacher-RL iterations (the second resumed from checkpoint) produced 16/16 valid responses and four nonzero optimizer updates. Each iteration had one zero-variance group out of two; this is a plumbing result, not a learning comparison.

Strict BF16 prompt-cache equivalence failed on both tested states; a greedy combat action changed. FP32 controls matched next-token logits within 0.0001 and all completed greedy actions. Broader batch/split diagnostics support numerical rounding as the cause. The HF reuse option remains off. Recomputed full-sequence old scores need not exactly equal cached sampling scores, which matters for interpreting on-policy claims.

## Conclusion

Batched generation and repeated-state serving caches are useful candidates. HF rollout is substantially slower than vLLM serving. Checkpointed SDPA with update microbatch one offers the safer measured memory margin. No playing-strength or learning gain is established, and low effort mostly produces very short answers.

Caveats: timing noise and changing output lengths; only two HF timing states; BF16 numerical dependence on backend/batch/cache; short update-cohort memory coverage; teacher preferences are not game values. Whole-game outcome RL is implemented and unit-tested, but has not been learned here; excluding invalid/incomplete game groups can bias its updates.

## Next

Agree on effort, numerical/backend settings, output cap and response-loss normalization before the first teacher-feedback pilot. Consider a tighter rollout/update engine integration if HF generation dominates the budget. Later full-game studies must report completion/group eligibility alongside wins. No main study is queued.

## Log

- 23 October: speed checks approved before learning; use RTX 2000 Ada as requested.
- 23 October: add explicit cache on/off comparisons after Sid raised shared prefixes.
- 23 October: retain failed strict BF16 cache checks, verify FP32 controls, and keep HF reuse off.

Code: experiments/2026-10-23-llm-throughput/ and src/sts_llm/ at commit 8eb669a (final collation corrections recorded with these notes). Data: run k2_throughput_20261023 (local).
