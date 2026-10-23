# Strategies

## Teacher-feedback RL followed by full-game outcomes

Planned. The model generates its own reasoning and action. A frozen non-LLM
policy scores actions during bootstrapping; no target reasoning is supplied.
The eventual objective is winning full games. Teacher probabilities are
preferences, not calibrated action values. Whether this bootstrap transfers
to full games is untested. Start with a shared adapter and evaluate both phases.

## Small-model serving before training

Running. [The serving study](experiments/2026-10-23-llm-throughput.md)
measures low, medium and high reasoning effort on matched public observations.
Throughput must be read beside response length, legal-action validity and
truncation. Serving measurements do not establish learning or game strength.
