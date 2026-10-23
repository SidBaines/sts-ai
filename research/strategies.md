# Strategies

## Teacher-feedback RL followed by full-game outcomes

Planned pilot, with medium effort selected by Sid on 23 October. The model
produces its own reasoning and action; a frozen non-LLM policy supplies
single-decision probability rewards without target reasoning. The
[pilot plan](experiments/2026-10-23-medium-teacher-pilot.md) proposes one shared
adapter, held-out worlds and a bounded compute allocation. Existing mechanics
checks verify nonzero updates and checkpoint continuation, with no strength
claim. Teacher preferences are not calibrated game values. Full-game transfer
is untested; terminal-outcome RL follows completion and reward-signal checks.
Dropping incomplete game groups can bias updates and must be addressed in that
stage's design. Separate phase adapters and prompt-history changes remain
future experiments after the initial learning signal is measured.

## Small-model serving before training

Finished. Batching improves practical throughput; repeated-state serving prefix
caching reached 3.27 versus 2.17 valid decisions/second in one medium-effort
cohort. Distinct states showed no gain. Fixed-token FlashInfer throughput was
5.8% higher, while repeated timings varied 15%. HF rollout is slower than vLLM.
Medium and high had similar median reasoning lengths, with a heavier high-effort
tail. No effort is established as better at the game. BF16 cache/batch changes
alter numerical outputs; FP32 controls matched. Checkpointed SDPA updates with
microbatch one had the safer measured memory margin.
[All evidence and caveats](experiments/2026-10-23-llm-throughput.md).
