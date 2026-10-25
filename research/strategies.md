# Strategies

## Teacher-feedback RL followed by full-game outcomes

On hold since 25 October, pending Sid's decisions. The idea:
- The model produces its own reasoning and action.
- A frozen non-LLM policy supplies a reward for each single decision, with no
  target reasoning.
- Full-game terminal outcomes come later.

As expected before any training, the unchanged model plays at
trivial-baseline level. No effort level beats always choosing the first menu
option on the teacher measure, and high effort scores below it. These numbers
are the "before" reference, not a finding.

Most combat rewards are near-ties: the teacher's top combat action averages
0.37 probability. The group standard-deviation scaling amplifies those ties.
The [23 October pilot design](experiments/2026-10-23-medium-teacher-pilot.md)
would probably end inconclusive. The
[25 October review](brainstorms/2026-10-25-getting-the-model-to-learn.md)
proposes a staged replacement:
- an overfit test on decisive out-of-combat states;
- mean-centred advantages;
- a prompt fixed on design grounds before training;
- held-out top-1 agreement on decisive states as the early signal.

Full-game RL now scores model-invalid games as losses. It is unit-tested, with
no learning run, and is far too slow per game at current generation rates.

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
