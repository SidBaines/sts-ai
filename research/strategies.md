# Strategies

## Teacher-feedback RL followed by full-game outcomes

Implemented, with no learning-strength result. The model generates its own
reasoning and action; a frozen non-LLM policy supplies action-probability rewards,
without target reasoning. Two bounded iterations and a checkpoint continuation
verify nonzero updates. One of two reward groups had zero variance in each
iteration. Teacher preferences are not calibrated game values. The later
whole-game outcome path is unit-tested; excluding incomplete groups can bias
updates. Start with a shared adapter and evaluate both phases. See the
[speed and mechanics study](experiments/2026-10-23-llm-throughput.md).

## Small-model serving before training

Finished. Batching improves practical throughput; repeated-state serving prefix
caching reached 3.27 versus 2.17 valid decisions/second in one medium-effort
cohort. Distinct states showed no gain. Fixed-token FlashInfer throughput was
5.8% higher, while repeated timings varied 15%. HF rollout is slower than vLLM.
Low effort mostly emits short answers; high effort has long tails. No effort
is established as better at the game. BF16 cache/batch changes alter numerical
outputs; FP32 controls matched. Checkpointed SDPA updates with microbatch one
had the safer measured memory margin. [All evidence and caveats](experiments/2026-10-23-llm-throughput.md).
