# Choosing settings for a small reasoning game player

**Status:** finished 23 October 2026, corrected 25 October. **Verdict:** batching and repeated-state caching clearly speed up serving, but on the teacher-preference proxy no reasoning effort beat always choosing the first menu option, and game quality is untested.

## Question

Which generation and update settings make the first teacher-feedback RL study practical? See [the approach](../strategies.md#small-model-serving-before-training).

## Design

Pin K2-Horizon-0.9B in BF16 on RTX 2000 Ada and use environment release 0.1.5.1. The primary panel has 64 states from 55 worlds, 40 combat and 24 out-of-combat, across Acts 1–3. All three efforts use the same states, temperature 0.6, top-p 0.95 and an 8,192-token cap. Keep raw responses and all invalid/capped outputs.

Screen concurrency 1/8/32, attention backend, scheduler budget and graph execution. Test prefix caching on/off with one or four responses per state, with fresh servers and disjoint warmups. Add equal-token synthetic probes to separate execution speed from differing output lengths. The screen is adaptive: every comparison is exploratory. Prefer usable completed decisions, then check length, legality, memory and repeat variability.

Benchmark the actual Hugging Face and PEFT training path separately: two mixed-phase states, eight samples each, top-p 1.0, and generation microbatches 1/4/8. Real update tests use eight saved completions, effective batch four, one warmup and three timed AdamW steps, with synthetic advantages. A bounded real-teacher smoke and checkpoint continuation test the plumbing; no reasoning SFT or main learning study is run.

Budget: at most three RTX hours at the allocated $0.24/hour, plus storage. Sid approved speed checks and code preparation on 23 October. The earlier A40 stopped before any comparable timing completed.

## Results

All timings are descriptive single cohorts, so no timing interval is justified. A repeat of identical work took 15% longer (34.4 versus 39.6 seconds), so only differences well beyond 15% are read as real: batching and four-response caching qualify; backend, scheduler-budget and graph changes do not. Warmup, calibration, failure and mechanics rows are in [the CSV ledger](../figures/llm-throughput/serving.csv).

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

All four fixed-work probes completed 64/64 requests. FlashInfer was 5.8% faster; graph removal and a smaller scheduler budget showed no gain. All three differences are inside the 15% repeat variation.

| Medium prefix caching, concurrency 32 | Off valid decisions/s | On valid decisions/s | On prompt-token hit fraction | Valid off / on |
|---|---:|---:|---:|---:|
| One response per state (64 total) | 2.08 | 1.83 | 2.4% | 64 / 63 |
| Four responses per state (256 total) | 2.17 | 3.27 | 75.2% | 254 / 254 |

Four-response caching raised decision throughput 50.6% and token throughput 41.4%, while generating 6.1% fewer tokens. Distinct states showed no gain beyond the repeat variation. Only 39 initial tokens were common to all prompts (mean length 1,728); repeated rules later in the text do not form a reusable prefix. Cache changes also changed sampled outputs.

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
| Medium BF16, static cache and compilation, 4,096-token cap | 8 | — | 240-second deadline | — | — | — |

The FP32 feasibility probe uses eight responses, a 1,024-token cap and no warmup; it is not a matched speed ratio against the 16-response screen. High effort produced a 5,520-token tail (5,510 of them reasoning) at microbatch eight and an unclosed 8,192-token capped response at microbatch four. The compiled static-cache case used a 4,096-token cap, not the 8,192 used elsewhere; it took 56 seconds for cold setup and 143 seconds for warmup, then exceeded its 240-second deadline without a complete timed cohort. It is not a demonstrated speedup.

Reasoning tokens in the matched FlashInfer runs (64 states per effort, including invalid final answers; final JSON and control tokens excluded; [counts](../figures/llm-throughput/reasoning_lengths.csv)): low mean 3.6, median 0, 95th percentile 27, maximum 73, with empty reasoning in 58/64; medium mean 318, median 255, 95th percentile 721, maximum 1,099; high mean 377, median 254, 95th percentile 969, maximum 1,447. Medium and high had similar medians; high had a heavier tail.

| Real update condition | Responses/s | Output training tokens/s | Peak allocated GiB | Result |
|---|---:|---:|---:|---|
| Eager attention, checkpointing on, microbatch 1 | 0.25 | 86.0 | 7.08 | completed |
| SDPA, checkpointing on, microbatch 1 | 0.76 | 255.7 | 2.91 | completed |
| SDPA, checkpointing on, microbatch 2 | 0.57 | 192.5 | 3.68 | completed |
| SDPA, checkpointing on, microbatch 4 | 0.46 | 156.5 | 5.85 | completed |
| SDPA, checkpointing off, microbatch 1 | 1.03 | 346.4 | 11.68 | completed |
| SDPA, checkpointing off, microbatch 2 | — | — | — | Out of memory |

Every successful update case changed adapter weights during its three timed steps. Larger checkpointed microbatches were slower. Removing checkpointing improved speed but used 11.68 GiB; microbatch two then ran out of memory. These completions reached about 3,062 total tokens, so the test does not certify long high-effort updates.

**Teacher preference: no effort beat always choosing the first menu option.** The score is the teacher's probability for the chosen action, with invalid or capped answers scored zero, on the 64 primary states (55 worlds). It is a preference, not win probability or reasoning quality. Ranges cover distinct runs; two cohorts that copied earlier ones token for token are left out.

| Policy on the 64 states | All | Combat (40) | Out of combat (24) | Picks the first option |
|---|---:|---:|---:|---:|
| Random legal action | 0.212 | 0.197 | 0.239 | 21% |
| Always the first option | 0.295 | 0.265 | 0.344 | 100% |
| Teacher's own top choice | 0.543 | 0.361 | 0.846 | 45% |
| Low effort, 4 runs | 0.351–0.352 | 0.273–0.274 | 0.480–0.481 | 45–48% of valid |
| Medium effort, 9 runs | 0.264–0.324 | 0.231–0.266 | 0.318–0.432 | 48–62% of valid |
| High effort, 4 runs | 0.219–0.252 | 0.225–0.253 | 0.198–0.272 | 52–57% of valid |

Paired against always-first on the same states, bootstrapping whole worlds: low effort +0.056 [-0.012, +0.127] (FlashAttention, concurrency 32; the other low runs agree within 0.001); medium runs -0.031 to +0.030, every interval spanning zero, and four samples per state give +0.004 [-0.042, +0.051]; high runs -0.043 to -0.076, with three of four intervals below zero (FlashInfer -0.073 [-0.136, -0.020]), and two of four when only valid answers count. Out of combat, medium picks the first option in 54–79% of valid answers, where the teacher's top choice is first in 38% of states. In combat even the teacher's top choice scores only 0.361, so this proxy has little room there. Full rows: [baselines](../figures/llm-throughput/teacher_baselines.csv).

Reasoning effort did not raise the score either. Paired differences on the same 64 states, intervals from bootstrapping whole worlds; the last column keeps only states where both arms answered validly:

| Backend | Effort comparison | Difference | 95% interval | Valid pairs only |
|---|---|---:|---|---|
| FlashAttention | high minus low | -0.098 | [-0.185, -0.012] | -0.083 [-0.166, +0.000] |
| FlashAttention | high minus medium | -0.038 | [-0.113, +0.033] | -0.024 [-0.092, +0.046] |
| FlashAttention | medium minus low | -0.060 | [-0.134, +0.011] | -0.059 [-0.137, +0.012] |
| FlashInfer | high minus low | -0.129 | [-0.204, -0.064] | -0.115 [-0.187, -0.054] |
| FlashInfer | high minus medium | -0.067 | [-0.134, -0.005] | -0.053 [-0.115, +0.004] |
| FlashInfer | medium minus low | -0.061 | [-0.138, +0.008] | -0.061 [-0.138, +0.008] |
| FlashInfer, four medium samples per state | high minus medium | -0.077 | [-0.143, -0.015] | -0.070 [-0.134, -0.011] |
| FlashInfer, four medium samples per state | medium minus low | -0.052 | [-0.115, +0.011] | -0.048 [-0.111, +0.014] |

These rows use concurrency 32. In all four high-effort runs, one out-of-combat state with options 0 and 1 got the illegal answer 3, scored zero; without such pairs, the first and fifth rows no longer exclude zero. Top-choice agreement mostly points the same way but is noisier; one comparison reads differently: FlashAttention high minus medium agreement is -0.141 [-0.270, -0.015] while probability shows no clear difference. The repeat medium cohort and the one-response cache-off cohort reproduced earlier cohorts' tokens exactly, so their comparisons are duplicates, not replications. Warmup (2 states) and calibration (4 states) cohorts have too few worlds for an interval (at least 10 are needed). Every paired comparison, with these flags, is in [the comparison CSV](../figures/llm-throughput/teacher_comparisons.csv).

Mechanics checks: 52 library tests pass with real Torch dependencies. The environment suite ran 266 tests: 247 passed and 19 optional tests skipped, and its random and base-model CLI traces replayed. The trained adapter also passed a three-decision clean-CLI smoke and fresh-process replay across both phases. Two teacher-RL iterations (the second resumed from checkpoint) produced 16/16 valid responses and four nonzero optimizer updates. Each iteration had one zero-variance group out of two; this is a plumbing result, not a learning comparison.

Strict BF16 prompt-cache equivalence failed at both efforts tested. At low effort (128-token greedy cap) the greedy action on the combat state changed; at medium (1,024-token cap) the combat trace hit the cap on one path but ended in 653 tokens on the other, and the out-of-combat actions matched while tokens differed. The FP32 control ran at low effort only, with a 128-token cap and 10-token outputs that contain no reasoning; it matched next-token logits within 0.0001 and all greedy actions. There is no medium-effort FP32 control, so cache equivalence for reasoning-length outputs is untested. Batch and split diagnostics, also at low effort, support numerical rounding as the cause. The HF reuse option remains off. Recomputed full-sequence old scores need not exactly equal cached sampling scores, which matters for interpreting on-policy claims.

## Conclusion

Batched generation and four-response serving caches give speedups far beyond the 15% repeat variation; backend, scheduler and graph settings stay inside it. HF rollout is substantially slower than vLLM serving. Checkpointed SDPA with update microbatch one offers the safer measured memory margin. On the teacher-preference proxy the untrained model does not beat always choosing the first option: medium is indistinguishable from it, high scores below it in most runs, and high scored below low in all four matched runs. No playing-strength or learning gain is established, and low effort mostly produces very short answers.

Caveats:
- Timing noise and changing output lengths; one repeat sets the noise estimate; only two HF timing states.
- BF16 outputs depend on backend, batch and cache; the FP32 cache control covers only 10-token low-effort outputs.
- Short update cohorts limit the memory coverage.
- Teacher preferences are not game values; the 64-state panel is small, and the effort rows share states and seeds (low-effort outputs were 60/64 identical between concurrency 1 and 32), so they are not independent replications. The baselines were added after the results were seen.
- Whole-game outcome RL is implemented and unit-tested, but has not been learned here; excluding invalid/incomplete game groups can bias its updates.

## Next

Sid selected medium effort on 23 October. The [medium-effort pilot plan](2026-10-23-medium-teacher-pilot.md) now defines the proposed next steps: matched rollout/scoring checks, a broader held-out corpus, baseline evaluation, a bounded teacher-feedback pilot, and a transfer check. Backend/precision, output cap and response-loss normalization must be frozen before updates. Pilot evaluations should report the random-legal and always-first baselines beside the model, since the untrained model does not yet beat always-first. No pilot is running.

## Log

- 23 October: speed checks approved before learning; use RTX 2000 Ada as requested.
- 23 October: add explicit cache on/off comparisons after Sid raised shared prefixes.
- 23 October: retain failed strict BF16 cache checks, verify FP32 controls, and keep HF reuse off.
- 23 October: Sid selected medium effort; next work is the planned teacher-feedback pilot.
- 25 October: audit corrections: added fixed baselines and a valid-pairs sensitivity, withheld intervals below 10 worlds, flagged duplicate cohorts, and tied timing verdicts to the 15% repeat variation.

Code: `experiments/2026-10-23-llm-throughput/` and `src/sts_llm/`. The serving, HF, update, fixed-work and cache CSVs come from `collate.py` at commit 4d30a8b; the teacher-preference CSVs from `analyze.py`, and reasoning lengths from `reasoning_lengths.py`, at commit 0a2cec6. Data: run `k2_throughput_20261023` (local); teacher analysis in its `analysis_teacher_20261025` folder.
