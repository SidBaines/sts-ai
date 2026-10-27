# Strategies

## Teacher-feedback RL followed by full-game outcomes

Promising for combat preferences, stalled on reliability. The model generates
its own reasoning and action; a frozen non-LLM policy supplies rewards.
[Stage 1](experiments/2026-10-25-overfit-test.md) learned 22/32 trained choices
from 10/32. The [broader run](experiments/2026-10-26-broader-training.md) collapsed.
[Four local trials](experiments/2026-10-27-learning-trials.md) delayed that failure:
full held-out matches 56 → 61/96 (+5.2 points [−4.1, +14.9], 73 worlds,
Exploratory), but valid answers fell 96 → 90/96 and reasoning doubled.
Combat teacher probability improved; out-of-combat and the additional validation
states did not. The short length penalty did not help. Retain the original
passing adapter. Next: independently scale answer-token learning and full-output
length control, with a frozen-adapter KL anchor and medium effort. No new run
is active; full-game learning and reliable transfer are untested.

## Running on the laptop with MLX

Promising. K2-Horizon-0.9B converts exactly to an MLX Llama model (FP32 token
log-probabilities agree within 0.0013).
- Checkpointed LoRA updates run at the RTX speed (0.74 against 0.76
  completions/s).
- Grouped generation reaches 1.07 valid decisions/s on the pilot-shaped
  workload.
- Batches that mix prompt lengths are 2–3× slower.
- The estimated pilot iteration is about 95 s, against about 165 s for the RTX
  Hugging Face path.
- The existing PyTorch trainer works on MPS but is 3.7× slower; the MLX
  trainer now runs stage-1 and the broader continuation.

[Evidence](experiments/2026-10-25-mac-local-speed.md).

## Small-model serving before training

Finished.
- Batching improves serving throughput (medium effort, 0.29 to 1.83 valid
  decisions/s).
- Repeated-state prefix caching helps (2.17 to 3.27); distinct states showed
  no gain.
- Equal-work backend differences sit within the 15% repeat noise.
- Hugging Face rollout is slower than vLLM.
- BF16 cache and batch changes alter numerical outputs. The FP32 control
  matched only at low effort with 10-token outputs.
- Checkpointed SDPA updates with microbatch one had the safest memory margin.

[All evidence and caveats](experiments/2026-10-23-llm-throughput.md).
