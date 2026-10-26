# Strategies

## Teacher-feedback RL followed by full-game outcomes

Promising. The idea: the model produces its own reasoning and action, a frozen
non-LLM policy rewards each single decision (no target reasoning), and
full-game outcomes come later.

Stage 1 passed on 26 October
([page](experiments/2026-10-25-overfit-test.md)). On 32 decisive states, half
combat and half out of combat, learning rate 0.00003 raised greedy
top-choice matches from 10 to 22. Agreement rose mainly through content. The
setup that worked: mean-centred advantages, decisive states, letter-labelled
menus, and a fresh menu order for every sample.

Known hazards:
- sharing a menu order within a group teaches "answer A";
- even with orders per sample, the label distribution drifted for two passes
  before recovering (no KL anchor);
- 0.0001 breaks the answer format.

The untrained model's trivial-baseline play is only the "before" reference.
Full-game RL scores model-invalid games as losses; it is unit-tested but has
no learning run and is too slow per game.

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
- The existing PyTorch trainer works on MPS but is 3.7× slower, so the next
  step is a small MLX trainer.

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
