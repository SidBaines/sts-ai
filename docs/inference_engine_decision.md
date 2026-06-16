# Inference engine: synchronous LLMEngine.step() vs AsyncLLMEngine

**Date:** 2026-06-16
**Status:** Chosen for the current rollout harness

## Decision

Use vLLM's synchronous `LLMEngine.step()` surface for the current
continuous-batching rollout orchestrator (`src/sts_ai/streaming_rollout.py`),
through the vLLM wrapper in `src/sts_ai/agents.py`.

This is not a throughput decision. A synchronous engine loop and
`AsyncLLMEngine` both provide full continuous batching for this workload, so
expected throughput is identical. The choice is about fit and future ownership.

## Why sync now

- The environment API is blocking C++ (`env.step()` / simulator advancement), so
  a synchronous orchestration loop matches the existing control flow.
- It avoids an asyncio layer around a path whose durable value is not the
  orchestrator itself.
- Online RL should be handed to a framework such as veRL, OpenRLHF, or TRL. Those
  systems own inference actors, batching, rollout scheduling, and weight sync.
  If/when we go there, this local orchestrator is replaced rather than extended.

The durable assets are the simulator/environment abstraction, prompt rendering,
JSON parsing, schemas, and eval traces. Keeping the orchestration simple protects
that boundary.

## Seams that keep this reversible

`src/sts_ai/agents.py` defines the narrow `GenerationBackend` surface:

- `stream_submit(request_id, state_text, legal_actions, seed)`
- `stream_poll()`
- `stream_has_unfinished()`
- `build_decision_from_text(...)`

`src/sts_ai/streaming_rollout.py` depends only on that surface. It does not know
whether completions came from synchronous `LLMEngine.step()`, an async engine, or
a scripted in-process test backend.

The version-sensitive low-level vLLM calls (`add_request` / `step`) are isolated
inside `VllmJsonAgent`.

## Async migration sketch

If async becomes the better fit:

1. Implement `GenerationBackend` over `AsyncLLMEngine`.
2. Run one coroutine per rollout slot, each awaiting generation for its current
   decision and then submitting the next decision.
3. Offload blocking simulator advancement (`advance_to_decision`, `env.step`) to
   a thread executor so the event loop can keep accepting generation completions.
4. Keep `run_streaming_rollouts`' slot lifecycle and record-writing contract
   unchanged, or replace only the thin scheduling loop around the same helpers.

That keeps the migration local to the backend/scheduler layer, not the rollout
record schema or prompt/parser code.

## Caveats

`LLMEngine.step()` is a lower-level and less-stable vLLM API than the async
engine surface used by vLLM's OpenAI server. Treat vLLM upgrades as
compatibility-sensitive and keep the call sites small.

Per-request sampling seeds are derived from
`(world_seed, rollout_index, decision_index)`, not from the current batch. That
makes stochastic sampling independent of batch composition and completion order,
which matters for comparing rollout sets across different `--concurrency`
settings.
