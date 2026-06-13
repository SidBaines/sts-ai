# src/sts_ai

Core harness package. See the top-level [`CLAUDE.md`](../../CLAUDE.md) for project goals and repo-wide rules.

## Module map

- `lightspeed.py` — `LightspeedHybridEnv`: wraps the simulator, advances through battles, lists/executes out-of-combat actions, summarizes state.
- `agents.py` — policies (`first`, `random`, `heuristic`, optional `mlx`/Qwen) and the JSON action parser.
- `rollout.py` — drives the env+agent loop and records `DecisionRecord`s as JSONL.
- `schemas.py` — dataclasses defining the **on-disk JSONL format**. This is the data contract consumed by future training/eval.
- `prompting.py` — prompt assembly and the framing strings.
- `lightspeed_import.py` — locates and imports the locally built `slaythespire` module.

## Area gotchas

- **The agent sees only the current state.** `choose_action(state_text, legal_actions)` is Markovian — it receives no after-state, reward, or record of combats resolved between decisions. Any "full-history" mode has to be built around this, not assumed.
- **`step()` re-derives the chosen action by index** from a freshly built `getAllActionsInState` list (`lightspeed.py`). This is only safe while that list is deterministic for a given state. If actions ever become stateful/nondeterministic, store and execute the chosen action object instead of re-fetching by index.
- **Legal actions are recomputed several times per decision** (loop `advance_to_decision` → `legal_actions` → `step`→`raw_actions`). Correct today, but watch for cost and the index-alignment assumption above.
- **`schemas.py` is load-bearing.** Once we start keeping data, changing these dataclasses is a breaking change to every stored trace — see the schema-stability guideline in the top-level CLAUDE.md.
