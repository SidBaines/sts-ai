# src/sts_ai/interactive

The **Interactive Rollout Studio**: a FastAPI backend + offline browser UI for
interactively driving agents — load a position, sample N decisions from a chosen
method (`user`/`first`/`random`/`heuristic`/`model`), edit the LLM framing/prompt,
branch to explore alternatives, and cache everything as canonical rollout JSONL.
See the top-level [`CLAUDE.md`](../../CLAUDE.md) for project rules and
[`scripts/interactive_app.py`](../../scripts/interactive_app.py) to launch it.

## Module map

- `replay.py` — `resolve_action_index` / `replay_actions`. Branch/load have **no
  binary state snapshot** (the C++ contexts are opaque), so they rebuild a position
  by replaying the recorded action sequence into a fresh env, matching each action
  by `(bits, description)` against the freshly built display list. No agent/model
  call happens during replay — pure simulator stepping.
- `templates.py` — framing + prompt-template manager. `NEUTRAL_FRAME` default
  framing; `DEFAULT_PROMPT_TEMPLATE` reproduces `prompting.render_action_prompt`
  **byte-for-byte** (locked by a parity test). `render_template` does single-pass
  `{placeholder}` substitution. `TemplateStore` persists named user templates and
  always surfaces the built-ins.
- `store.py` — `SessionStore` + `StoredSession`. One dir per session under the
  cache root: `decisions.jsonl` (**canonical `DecisionRecord` schema, UNCHANGED**),
  `meta.json` (`build_rollout_meta`), `session.json` (Studio reconstruction data:
  lineage, config, per-decision method, current framing/template). Pure dict I/O —
  unit-tested without the simulator.
- `session.py` — `RolloutSession` (live env at the path frontier) + `SessionRegistry`
  (live sessions keyed by id, backed by the store; the only simulator-touching part).
  Step/sample/branch/stream + framing/config editing. Env + agent construction are
  **injected** (`env_factory` / `agent_builder`) so the core is unit-testable with
  fakes; the real defaults lazily import the simulator and the MLX/vLLM agents.
- `server.py` — `create_app(...)`: thin FastAPI/SSE shell over the registry.
  **FastAPI is imported inside `create_app`** so this package imports without the
  `[app]` extra. Static SPA served via `FileResponse` (no extra deps).
- `static/` — vanilla-JS offline SPA (`index.html`/`app.js`/`styles.css`): branch
  tree, server-rendered board (reuses `rollout_view_html`), control panel, prompt
  editor (Framing + Advanced tabs), live token stream. No bundler, no CDN.

The board HTML is rendered server-side by the shared, pure
[`sts_ai.rollout_view_html`](../rollout_view_html.py) (builders lifted from the
Streamlit viewer), so the JS just injects `board_html`.

## Area gotchas

- **A session is one linear path; "go back" = branch.** The live env is forward-only
  (no snapshot), so to act from an earlier decision you `branch_at(k)` — a fresh env
  replays `history[:k]`. The original session is untouched and stays **warm in
  memory**, so replay cost is paid only on load/branch, never per-step.
- **Replay determinism is verified for BOTH combat modes** (`tests/integration/
  test_interactive_replay.py`): `llm` re-applies the exact recorded player actions;
  `search` re-runs the C++ search agent per combat and still reproduces the frontier
  exactly. If `search` ever regresses, `llm`-mode branching stays exact.
- **`search`-mode branch/load replay re-runs each combat** (~`battle_simulations`
  MCTS sims), so a late-game branch can take seconds. The UI shows a "replaying…"
  status; envs stay warm afterwards.
- **`llm`-combat hits unsupported player-choice input states** for some cards/potions
  (per [`../CLAUDE.md`](../CLAUDE.md) milestone-1 scope) — these raise a
  `simulator_error` that `RolloutSession` surfaces (status `error`, `stopped_reason`)
  rather than hanging. `search` mode is the robust fallback.
- **`schemas.py` is NOT changed.** `decisions.jsonl` is the canonical contract so
  `summarize_rollouts`/`compute_risk_proxies`/`compare_models`/`visualize_rollout`
  all work on Studio output; `session.json` is a **separate** sidecar for the
  Studio's own reconstruction/lineage data.
- **The session-summary lifecycle field is `session_status`, not `status`.**
  `_session_summary()` is spread into every view response dict, which already carries
  a top-level `status` (`ok`/`terminal`/`agent_invalid`/…). A lifecycle key named
  `status` would clobber it (bitten during development). Any new view field must not
  collide with a response key.
- **SSE: the client MUST close the EventSource on the `done` event.** `stream_step`
  is a *sync* generator; Starlette iterates it in a threadpool so the blocking model
  decode never stalls the event loop, and a client disconnect closes the generator
  (cancel). But when the generator ends the server drops the connection, and an open
  `EventSource` would **auto-reconnect and re-run a whole generation** — `app.js`
  calls `es.close()` the instant it sees `type:"done"`.
- **`prompt_override` on the agents is additive** (`MlxQwenJsonAgent`/`VllmJsonAgent`
  `choose_action`, default `None`): set, it bypasses `render_action_prompt`/`_base_prompt`
  with a fully-rendered prompt (the *advanced template* path). The framing-only path
  leaves it `None`, so those prompts are byte-identical to the harness. `stream_choose_action`
  (MLX) wraps `mlx_lm.stream_generate`, yields text segments, and **returns** the parsed
  `AgentDecision` via `StopIteration.value`.
- **Offline:** with a cached MLX model the whole thing runs without internet
  (verified with `HF_HUB_OFFLINE=1`). `vllm` backend is CUDA-only.
- **Tests:** core logic is unit-tested with a fake env + fake agent
  (`tests/unit/test_interactive_*`); API tests use `TestClient` + injected fakes,
  gated on `@requires_fastapi` (need neither the simulator nor MLX). The
  simulator-touching path is covered by `tests/integration/test_interactive_*`.
