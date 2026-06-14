# NextStep — Implement in-battle (full combat) LLM control

**Read this first, then `CLAUDE.md`, `docs/research_plan.md`, and `src/sts_ai/CLAUDE.md`.**

## Objective (reprioritized 2026-06-14)

Give the LLM control of **in-combat** decisions (which cards to play, targets,
potion use in battle, end-turn), not just out-of-combat ones. This was "Stage 10"
in the roadmap; it has been **moved up to be the immediate next priority** (ahead
of scaling Stage 5 neutral-rollout collection). The research needs in-combat
risk-taking, and right now the model never makes a combat decision.

## Why this is the gap

The current harness is **hybrid**: Python (Qwen) controls out-of-combat
decisions, and **every battle is auto-resolved by the built-in C++ search agent**
(`ScumSearchAgent2`, MCTS). Concretely, in `src/sts_ai/lightspeed.py`,
`advance_to_decision()` loops while `screen_state == BATTLE` and calls
`resolve_current_battle(gc, battle_agent)`, which plays the *entire* fight before
returning control to Python. So `ActionAgent.choose_action()` is never invoked
during combat. Verified: a full seed-3 rollout has **zero `BATTLE` decisions**.

Consequences to fix: combat tactics are not learned, and the strong search agent
masks the consequences of poor pathing/deck choices.

## What "done" looks like (acceptance criteria, from research_plan.md Stage 10)

- The LLM can complete at least one full combat through Python control.
- Legal combat actions are complete and valid (no illegal action reaches the sim).
- Hybrid mode (search-agent battles) **and** full-control mode both still work, so
  we can ablate. Don't delete `resolve_current_battle`; add alongside it.
- Combat state is serialized richly enough for a human to judge the choice.
- Rollout schema distinguishes combat vs out-of-combat decisions.

## The C++ you need to bind (it exists; it's just not exposed)

The binding is a **patch** applied to an untracked upstream clone — see the
workflow section below. Relevant upstream headers (under
`external/sts_lightspeed/include/`):

- `combat/BattleContext.h` — `struct BattleContext`. Key members/methods:
  - `void init(const GameContext &gc)` / `init(gc, MonsterEncounter)` — start a battle.
  - `void exitBattle(GameContext &g) const` — write battle results back to the run.
  - `void executeActions()` — drain the queued-action/effect engine.
  - `InputState inputState` (`combat/InputState.h`) — combat is **input-driven**:
    the engine runs `executeActions()` until it needs a player decision
    (`InputState::PLAYER_NORMAL`), e.g. choosing a card/target. This is the
    natural "yield a decision to Python" point.
  - `Outcome outcome` (UNDECIDED until the fight ends), `bool undefinedBehaviorEvoked`.
  - `setState(InputState)`.
- `sim/search/Action.h` — `struct Action` (combat micro-action) with
  `enum class ActionType`, `bool isValidAction(const BattleContext&)`,
  `void execute(BattleContext&) const`. **This is the in-combat action**, distinct
  from the out-of-combat `search::GameAction` already bound.
- `sim/search/BattleScumSearcher2.h` — the MCTS battle searcher. Read how it
  **enumerates legal actions** for a `BattleContext`; reuse that enumeration to
  produce the legal-action list shown to the LLM (mirror what
  `search::GameAction::getAllActionsInState` does for out-of-combat).
- `NNInterface` (already bound, `getObservation`) — may help cross-check what
  combat state matters; not required.

What's already bound (in the patch) for reference: `resolve_current_battle`,
`GameContext`, out-of-combat `GameAction` (with `.describe()`/`.execute()`),
`ScumSearchAgent2`, `NNInterface`.

## Suggested approach (incremental — land each milestone behind tests)

1. **Bind `BattleContext` + combat `Action`** through pybind: construct/init from a
   `GameContext`, expose `inputState`/`outcome`, an action-enumeration call
   (legal combat actions), `Action.isValidAction/execute/describe`, and
   `exitBattle`. Edit the patch (see workflow), rebuild, regen the patch.
2. **A `describeBattleState(bc)`** serializer (mirror `describeGameState`): hand,
   draw/discard/exhaust pile sizes (or contents), each enemy's HP/block/**intent**,
   player block/energy/powers, potions, turn counter, and the legal targets. Bare
   labels were misread by the LLM out-of-combat — spell out card/intent semantics
   (see the campfire enrichment in the binding patch for the pattern).
3. **A combat step loop** in `lightspeed.py`: add a mode where, instead of
   `resolve_current_battle`, the env drives `executeActions()` until
   `PLAYER_NORMAL`, surfaces legal combat actions + serialized state as a normal
   decision, applies the chosen `Action`, and repeats until `outcome != UNDECIDED`,
   then `exitBattle`. Keep the hybrid path available (a flag).
4. **Decide action granularity** (write the choice down): each card play = one
   decision (micro), vs. one whole turn = one decision (macro). Micro is simpler to
   validate and gives finer risk signal; start there.
5. **Schema/rollout**: extend records to mark `phase: "combat" | "out_of_combat"`
   and carry combat-specific state. `schemas.py` is the on-disk contract — treat as
   a deliberate, flagged change (we have not frozen combat training data yet, so now
   is the right time). Update `rollout.py` to record combat decisions, and the
   agent prompt in `prompting.py`/`agents.py` for combat (the JSON action schema can
   stay `{"reasoning","action_index"}`).
6. **Risk proxies / viewer**: extend `risk_proxies.py` (in-combat aggression: potion
   use, risky card plays) and `rollout_view.py` (render combat snapshots) once data
   exists. Not blocking for milestone 1.

## Workflow & gotchas (these will bite you — they're in CLAUDE.md too)

- **The Python↔C++ binding is a patch**, `patches/sts_lightspeed_python_api.patch`,
  applied to the untracked `external/sts_lightspeed/`. Edit the source in
  `external/.../bindings/slaythespire.cpp`, rebuild with
  `.venv/bin/cmake --build external/sts_lightspeed/build --target slaythespire -j 8`,
  then **regenerate the patch**: `cd external/sts_lightspeed && git diff > ../../patches/sts_lightspeed_python_api.patch`. The build script skips re-applying when the
  binding is already present, so a stale patch won't be caught locally — always regen.
- **Release build → `assert()` is a no-op.** Any combat guard that must fire in
  production must **throw**, not assert (see `BattleContext::executeActions` in the
  patch). The simulator also has known uninitialized-memory UB (potions); combat
  resolution must fail closed (throw a Python-visible error), never hang.
- **`undefinedBehaviorEvoked`**: some cards cause inconsistent outcomes; surface it,
  don't silently ignore.
- Run everything with `PYTHONPATH=src` and `.venv/bin/python`. Build first
  (`scripts/build_lightspeed.sh`) — `external/` and build outputs are gitignored.
- Tests are tiered: `tests/unit` (pure Python) and `tests/integration`
  (`@requires_simulator`). Add a unit test for the combat-state serializer parsing
  and an integration test that plays one full combat. Run:
  `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`
  (use `STS_REQUIRE_SIMULATOR=1` to force the integration tier to run).
- **Determinism / seed-2-class UB**: see `docs/simulator_issue_handoff.md`. Use
  `--seed-timeout-seconds` for batch runs; don't trust cross-machine identical traces.

## First concrete milestone to aim for

Bind `BattleContext` + combat `Action`, add `describeBattleState`, and write an
**integration test that plays one full battle to a terminal `Outcome` under a
scripted (non-LLM) action agent** — proving the combat step loop and legal-action
enumeration work end-to-end. Everything else builds on that.

## Status pointers

- Plan: `docs/research_plan.md` (Stage 10 + the reprioritization note in Near-Term Steps).
- Latest session summary: `docs/progress_report_2026-06-14.md`.
- Out-of-combat decision loop to mirror: `src/sts_ai/lightspeed.py`
  (`advance_to_decision`, `legal_actions`, `step`) and `src/sts_ai/rollout.py`.
