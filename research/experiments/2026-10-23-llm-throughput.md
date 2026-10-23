# Choosing serving settings for a small reasoning game player

**Status:** running 23 October 2026. **Verdict:** pending.

## Question

Which serving settings and reasoning effort deliver useful game decisions fast
enough for the first RL study? Approach: [small-model serving](../strategies.md#small-model-serving-before-training).

## Design

Use IFM/K2-Horizon-0.9B in BF16 on one A40. Compare low, medium and high
reasoning effort on the same fixed observations from environment release
0.1.5.1, with combat and out-of-combat menus represented across all three acts.
No model weights are changed and no reasoning examples are supervised.

Screen request concurrency at 1, 8 and 32 with a common server configuration.
Then compare supported attention backends, scheduler token budgets and eager
versus graph execution at useful concurrency. Keep sampling settings, model
revision, prompts and output limits matched within each comparison. Calibration
may increase output limits before freezing a comparison; preserve earlier runs.
Use warmups and repeated batches for finalists, and separate cached from fresh
prefix conditions. Hardware-incompatible cases are recorded as unavailable.

Measure valid decisions per second, generated tokens per second, latency,
completion length, legal-action validity, truncation, transport errors and GPU
memory. Preserve all responses. Prompt-level teacher agreement is diagnostic
only. This is a serving study, not an estimate of game strength or learning.

The decision rule is to prefer settings that improve usable decision throughput
without a material increase in incomplete or illegal outputs. Reasoning effort
is chosen using this measured tradeoff, with conclusions about playing strength
reserved for a later game evaluation. All configuration comparisons are
exploratory because the screening is adaptive.

Budget: at most two A40 hours at $0.59/hour ($1.18 GPU cost), plus storage.
Sid approved a short serving-settings and three-effort test on 23 October.
Machine settings, source identities and exact schedules live in the run record.

## Results

## Conclusion

## Next

## Log

- 23 October: benchmark approved before the RL study; isolated A40 started with a two-hour limit.

Code: experiments/2026-10-23-llm-throughput/. Data: run k2_throughput_20261023 (local).
