# src/sts_ai

Core harness package. See the top-level [`CLAUDE.md`](../../CLAUDE.md) for project goals and repo-wide rules.

## Module map

- `lightspeed.py` — `LightspeedHybridEnv`: wraps the simulator, advances through battles, lists/executes actions, summarizes state. Supports two combat modes via `combat_control`: `"search"` (default, hybrid — battles auto-resolved by the built-in C++ search agent) and `"llm"` (full control — each in-combat micro-decision is surfaced to the agent).
- `agents.py` — policies (`first`, `random`, `heuristic`, optional `mlx`/Qwen) and the JSON action parser.
- `rollout.py` — drives the env+agent loop and records `DecisionRecord`s as JSONL.
- `schemas.py` — dataclasses defining the **on-disk JSONL format**. This is the data contract consumed by future training/eval.
- `prompting.py` — prompt assembly and the framing strings.
- `risk_proxies.py` — deterministic risk-proxy classification/aggregation over recorded decision dicts (pure Python; no simulator). Keys off the semantic action description; tolerant of the legacy `bits=` prefix in pre-2026-06-14 traces.
- `lightspeed_import.py` — locates and imports the locally built `slaythespire` module.

## Area gotchas

- **The agent sees only the current state.** `choose_action(state_text, legal_actions)` is Markovian — it receives no after-state, reward, or record of combats resolved between decisions. Any "full-history" mode has to be built around this, not assumed.
- **`step()` re-derives the chosen action by index** from a freshly built `getAllActionsInState` list (`lightspeed.py`). This is only safe while that list is deterministic for a given state. If actions ever become stateful/nondeterministic, store and execute the chosen action object instead of re-fetching by index.
- **Legal actions are recomputed several times per decision** (loop `advance_to_decision` → `legal_actions` → `step`→`raw_actions`). Correct today, but watch for cost and the index-alignment assumption above.
- **`schemas.py` is load-bearing.** Once we start keeping data, changing these dataclasses is a breaking change to every stored trace — see the schema-stability guideline in the top-level CLAUDE.md.

## In-combat (full-control) mode

- **`self.bc` is the phase signal.** In `combat_control="llm"`, `advance_to_decision` constructs a `BattleContext` and holds it on `self.bc` while an in-combat decision is pending; `phase()` returns `"combat"` iff `self.bc is not None`. The four decision methods (`raw_actions`/`legal_actions`/`describe_state`/`step`) branch on `self.bc` and use `self.bc` (not `self.gc`) as the action context. `step` captures the context **before** executing, since a battle-ending action clears `self.bc` inside the follow-up `advance_to_decision`.
- **The engine drains itself.** `BattleContext.init` and `BattleAction.execute` both call the C++ `executeActions()` internally, so Python never calls `execute_actions` directly — after either, the state is already at the next player decision or a decided `outcome`.
- **Only `PLAYER_NORMAL` and `CARD_SELECT` are player-decision input states.** Every other `InputState` (the `SELECT_ENEMY_ACTIONS`/`SHUFFLE_*`/`FILL_RANDOM_*` "random" family, etc.) is resolved inside `executeActions` and never surfaces as a decision. The C++ enumerator (`BattleScumSearcher2::enumerateActionsForNode`, reused by the binding's `enumerate_battle_actions`) only handles those two — full parity with the built-in search agent and `BattleSimulator`. Other player-choice states (`CHOOSE_STANCE_ACTION`, `SCRY`, `CHOOSE_*_CARDS`, `GAMBLE`, …) are reachable only via specific cards/potions that are out of milestone-1 scope; if one is hit, `enumerate_battle_actions` **throws** (a clear `simulator_error` naming the state) rather than returning an empty list that would silently stop the rollout `no_legal_actions`. Handling them is the natural next extension.
- **Combat-specific state rides in `state["combat"]`** (turn, input_state, battle_outcome, player hp/block/energy, `undefined_behavior_evoked`, and a structured `enemies` list — index/name/cur_hp/max_hp/block/intent/alive from `BattleContext.enemies()`); the only schema change was the additive `DecisionRecord.phase`. Combat actions reuse `LegalAction` (`bits`/`description`) and the `{reasoning, action_index}` agent JSON unchanged.
- **`undefined_behavior_evoked` is latched** at env level (`self._undefined_behavior_evoked`) because `exit_battle` clears `self.bc` before the after-state is recorded; it is surfaced as a top-level `summary()` field (OR'd with any in-progress battle) so a UB raised by a battle-ending action is not lost.
- **Combat reuses the neutral framing.** No prompt/agent-protocol change: combat decisions flow through the same `choose_action(state_text, legal_actions)` with combat-aware `state_text` and action descriptions. Card-play and potion actions name the card/potion and target; `SINGLE_CARD_SELECT` actions resolve and name the card from the task's source pile (`describeCardSelectChoice`, mirroring the engine's enumeration). Intent labels in the serializer are the engine's raw move strings (e.g. `ACID_SLIME_S_LICK`); fine for now, could be enriched later.
