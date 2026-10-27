# Strategies

## Teacher-feedback RL followed by full-game outcomes

Stalled at broader learning. The model produces its own reasoning and action,
a frozen non-LLM policy rewards each decision, and full-game outcomes come later.
[Stage 1](experiments/2026-10-25-overfit-test.md) worked on 32 selected states:
learning rate 0.00003 raised teacher matches from 10 to 22. The
[broader continuation](experiments/2026-10-26-broader-training.md) failed after
21 updates: held-out matches 56 → 6/96 (−52.1 points, 95% interval −62.1 to
−41.8; 73 worlds, Exploratory), with 86/96 final answers reaching the cap.
The observed failure is repetitive output; its cause is not isolated.
Mean-centred rewards and per-sample shuffled menus permit small-set learning,
but do not ensure stable continuation. The original passing adapter is retained.
Next: a targeted lower-rate/KL comparison with measured loss terms, not a seed
repeat. No further experiment is running. Full-game RL is not yet tested.

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
