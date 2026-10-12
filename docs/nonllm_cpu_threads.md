# Bounded CPU actor thread screen

Before the final source/freeze boundary, two CPU-only screens compared Torch threads 1/2/4 at batches 8 and 32. Each used the same 2,048 actual development observations from the corrected-native Act-1 teacher bank (1,401 combat, 647 OOC; maximum 21 legal candidates), with shuffled batches held fixed. No native policy, final-world policy, GPU inference or optimizer was executed. The live campaign continued separately, so contention is an explicit limitation. The two full screens took 12.09 and 12.07 seconds; condition order was reversed in the second.

The captured selected composite checkpoint was `joint_act3_0002.pt`, SHA256 `47166f7d3c7548646cd1c77f56c73c2376f71630a26ade2ce184c38250060e6b`, with two large/no-context candidate heads. The fixture public-observation digest is `1befb1e694c917f729784601cab906bda9e104d231194068a665373a2192c50e`. Source identities before/after import and each screen matched; encoder SHA was `6fe756c7bfa58e7423c58d18bfe11b142d656b04c1e1cf4557a5605cbb6880d1`. These measurements precede the separately proposed bag-encoding cache optimization and must not be silently presented as its timings.

Each condition warmed eight batches, then ran two complete passes. Model-only input tensors were prepared ahead of timing; end-to-end timings included fresh encoding, collation and forward. Batch 8 supplied 512 latency samples per path/condition; batch 32 supplied 128. Both screens produced bit-identical logits, scalar values and argmax actions across all thread choices (maximum absolute logit/value differences zero; checked tolerance rtol/atol 1e-5).

| Pass | Batch | Threads | Model p50 / p95, ms | Encoder/collate/forward p50 / p95, ms | End-to-end examples/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 8 | 1 | 0.711 / 0.938 | 2.404 / 2.977 | 3275 |
| A | 8 | 2 | 0.726 / 1.253 | 2.429 / 3.297 | 3171 |
| A | 8 | 4 | 0.703 / 1.502 | 2.390 / 3.561 | 3188 |
| A | 32 | 1 | 1.764 / 2.459 | 8.168 / 9.608 | 3894 |
| A | 32 | 2 | 2.838 / 4.108 | 9.517 / 11.021 | 3363 |
| A | 32 | 4 | 2.664 / 4.015 | 9.050 / 10.725 | 3505 |
| B reversed | 8 | 1 | 0.713 / 0.936 | 2.397 / 2.967 | 3285 |
| B reversed | 8 | 2 | 0.690 / 0.955 | 2.281 / 2.804 | 3440 |
| B reversed | 8 | 4 | 0.720 / 1.588 | 2.421 / 3.600 | 3106 |
| B reversed | 32 | 1 | 1.754 / 2.510 | 8.241 / 9.753 | 3862 |
| B reversed | 32 | 2 | 2.802 / 4.212 | 9.547 / 11.264 | 3355 |
| B reversed | 32 | 4 | 1.895 / 2.943 | 8.507 / 10.299 | 3719 |

**Decision: retain the current two-thread default.** At the current eight-worker/batch-eight actor configuration, one thread improved mean end-to-end throughput by about 3.3% in pass A but reduced it by about 4.5% in pass B. Four threads produced worse/noisier tails. This does not justify changing the live configuration near final freeze.

At batch 32, one thread reproducibly improved end-to-end throughput by about 15–16% relative to two threads. That is evidence for a future larger-batch CPU deployment/collection route, not permission to change the current worker count, asynchronous batching, updates or scientific settings. A later experiment would need actual integrated collection evidence and menu/phase coverage beyond this Act-1 fixture.

No pipeline settings, source files or supervisor state were changed by this screen. Full input/source identities and measurements are retained in [nonllm_cpu_thread_evidence.json](nonllm_cpu_thread_evidence.json).
