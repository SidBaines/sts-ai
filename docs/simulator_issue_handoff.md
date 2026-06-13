# Simulator Issue Handoff

## Resolution (2026-06-13)

**Root cause was misdiagnosed in the original handoff below.** It is *not* a stale
cached MCTS edge. The real cause is **uninitialized-memory undefined behaviour** in
the upstream simulator, surfaced (not caused) by our strict validation:

- `GameContext::potions` and `BattleContext::potions` (`std::array<Potion,5>`) had
  **no default initializer**. `BattleContext::init` copies `gc.potions` wholesale,
  so indeterminate bytes in unused slots propagated into combat state. In the search's
  internal playout copies this produced garbage potion values (e.g. `223`, `50`),
  which strict validation correctly rejected — the `type=1 source=2` "invalid battle
  action" crash.
- Because it is UB, the symptom is **build-/layout-dependent**: the same logical input
  threw on one build and **infinite-looped** on another. The loop is in
  `BattleContext::executeActions`, whose `while(true)` overflow guard was
  `assert(false)` — a **no-op in release builds** (`-DCMAKE_BUILD_TYPE=Release`), so a
  corrupted state spun forever instead of bailing.
- The real (`rootState`) battle state was verified **never** corrupt; corruption is
  confined to the search's internal playout copies, so recorded agent data is clean.

**Fixes (all in `patches/sts_lightspeed_python_api.patch`):**

1. Initialize `potions` to `EMPTY_POTION_SLOT` in both `GameContext` and
   `BattleContext` (removes the UB at its origin).
2. `BattleScumSearcher2`: skip non-potion slot values when enumerating potion actions;
   validity-filter stale stored edges in the MCTS non-leaf branch; cap playout length.
3. `ScumSearchAgent2::playoutBattle`: cap search-and-commit iterations per battle.
4. `BattleContext::executeActions`: **throw instead of `assert(false)`** so the
   overflow guard actually fires in release builds — converting the hang into a clean,
   catchable `simulator_error`.

**Status:** the original crash is fixed; the residual UB (a build-dependent
unresolvable battle) is now *contained* (clean error, never a crash or hang) per the
agreed scope, not fully root-caused. Verified via a fast LLM-free replay harness
(`scripts/`-style replay of recorded decision indices) reproducing seed 2 in seconds.
Regression test: `tests/test_battle_search_regression.py`. A deeper root-cause hunt
(systematic value-init of all context members, or a sanitizer build) remains a
possible follow-up if seed-2-class trajectories prove research-critical.

> The original handoff below is preserved for history; its "Likely Root Cause" and
> "Proposed Fix" sections were superseded by the findings above.

## Current Situation

We have a working Slay the Spire rollout harness, and the Qwen3-4B no-thinking path is now capable enough to reach late Act 1 and sometimes Act 2. The current blocker is not LLM parsing; it is simulator/search robustness when the built-in battle search encounters potion-heavy states.

The important failing case is:

```text
agent: mlx_Qwen3_4B_4bit_nothinking_256
seed: 2
max_decisions: 80
battle_simulations: 100
failure point: Act 1 floor 12 battle
```

The error sidecar is:

```json
{
  "decisions": 48,
  "error": {
    "decision_index": 48,
    "message": "invalid battle action seed=2 floor=12 input_state=1 bits=536870914 type=1 source=2 target=0",
    "phase": "step",
    "type": "RuntimeError"
  },
  "seed": 2,
  "stopped_reason": "simulator_error"
}
```

`type=1` is a battle `ActionType::POTION`. `source=2` means potion slot 2. `target=0` means it is trying to drink/use the potion rather than discard it. `input_state=1` corresponds to `InputState::PLAYER_NORMAL`, so the failure is not caused by a non-player-normal state. It is most likely because the cached/search-planned potion action references a potion slot that is empty or otherwise invalid in the current simulated battle state.

The trace immediately before the failure is in:

```text
data/qwen_long_100_256/mlx_Qwen3_4B_4bit_nothinking_256/seed_2.jsonl
```

Relevant lead-up:

- On floor 11, Qwen takes gold.
- Then Qwen takes `Entropic Brew`.
- Then Qwen proceeds.
- The next battle resolution fails with the invalid potion action above.

The same failure reproduces after rebuilding with the preliminary ScumSearchAgent2 guard:

```text
data/qwen_patchcheck_100_256/mlx_Qwen3_4B_4bit_nothinking_256/seed_2.error.json
```

## What Is Already Implemented

The current repository already has robust outer handling:

- `Action::execute(BattleContext&)` validates battle actions in release builds and throws instead of silently mutating state.
- `BattleContext::drinkPotion` / `discardPotion` reject invalid, empty, unknown, or out-of-range potion slots.
- Python rollouts catch simulator exceptions and write `.error.json` sidecars.
- Batch rollouts support per-seed subprocess timeouts via `--seed-timeout-seconds`.

This strict behavior should be preserved. It is catching real simulator/search invalidity rather than causing the bug.

## Experimental Patch Tried

I tried patching final action execution in `ScumSearchAgent2`:

- `stepThroughSolution(...)` now returns `bool`, checks `a.isValidAction(bc)` before `takeAction`, and clears stale `bestActions`.
- `stepThroughSearchTree(...)` now skips invalid edges before selecting by simulation count.
- `playoutBattle(...)` throws if no progress can be made.

Files changed in the external checkout:

```text
external/sts_lightspeed/include/sim/search/ScumSearchAgent2.h
external/sts_lightspeed/src/sim/search/ScumSearchAgent2.cpp
```

This did **not** fix seed 2. The failure still occurs with the same invalid potion action.

Interpretation: the invalid action is likely being executed inside `BattleScumSearcher2::search()` / `BattleScumSearcher2::step()` during MCTS simulation, before the final `ScumSearchAgent2` path-following code gets a chance to validate anything.

Important repo-state note: the `ScumSearchAgent2` patch is currently only in `external/sts_lightspeed`; it has not been refreshed into `patches/sts_lightspeed_python_api.patch`. The tracked patch file includes the earlier strict validation/binding changes, not this failed experiment.

## Likely Root Cause

`BattleScumSearcher2` stores a tree of `Edge { Action action; Node node; }`.

In `BattleScumSearcher2::step()`:

```cpp
BattleContext curState;
curState = *rootState;

...
auto &edgeTaken = curNode.edges[selectIdx];
edgeTaken.action.execute(curState);
```

For existing tree nodes, the selected `edgeTaken.action` may have been generated under a prior simulated state. With potion effects, especially `Entropic Brew`, the live `curState` reconstructed from `rootState` may not contain the same potion slots as the previously simulated branch. A cached potion action such as "use slot 2 on target 0" can therefore become invalid when the tree is traversed again.

The current MCTS code assumes stored edge actions remain valid whenever the same tree node is revisited. That assumption appears false for stochastic or state-mutating potion branches.

## Proposed Fix

Patch `BattleScumSearcher2` itself, not just `ScumSearchAgent2`.

The main idea:

1. When traversing an existing node in `BattleScumSearcher2::step()`, select only edges whose `edge.action.isValidAction(curState)` is true.
2. If the preferred/best edge is invalid, ignore or erase it and try the next best valid edge.
3. If all existing edges are invalid for `curState`, clear/re-enumerate actions from `curState` and treat the node as a leaf again.
4. Keep `Action::execute` strict, so invalid execution still throws if a bad action slips through.

Likely implementation sketch:

```cpp
int selectBestValidEdgeToSearch(const Node &cur, const BattleContext &state) {
    int bestEdge = -1;
    double bestEdgeValue = 0;
    for (int i = 0; i < cur.edges.size(); ++i) {
        if (!cur.edges[i].action.isValidAction(state)) {
            continue;
        }
        const auto value = evaluateEdge(cur, i);
        if (bestEdge == -1 || value > bestEdgeValue) {
            bestEdge = i;
            bestEdgeValue = value;
        }
    }
    return bestEdge;
}
```

Then in the non-leaf branch of `BattleScumSearcher2::step()`:

```cpp
const auto selectIdx = selectBestValidEdgeToSearch(curNode, curState);
if (selectIdx == -1) {
    curNode.edges.clear();
    continue;  // next loop iteration re-enumerates this node as a leaf
}
auto &edgeTaken = curNode.edges[selectIdx];
edgeTaken.action.execute(curState);
```

Also consider adding a validity guard in the leaf branch immediately after enumeration and before executing the randomly selected leaf edge, although newly enumerated actions should already be valid:

```cpp
if (!edgeTaken.action.isValidAction(curState)) {
    curNode.edges.clear();
    continue;
}
```

`playoutRandom(...)` probably does not need the same fix because it enumerates actions from the current playout state immediately before choosing, but a defensive guard there would be reasonable.

## Alternative Workarounds

If a full MCTS validity patch is too invasive:

- Lower-risk workaround: filter potion reward/action handling for problematic potions such as `Entropic Brew` in the Python-facing out-of-combat policy. I do **not** recommend this for research data because it biases the action space.
- Battle-search workaround: disable potion actions in `BattleScumSearcher2::enumeratePotionActions`. This is likely robust but changes combat quality and would weaken the ecological validity of potion-related decisions.
- Operational workaround: keep the strict error sidecars and exclude simulator-error seeds from training/eval. This is acceptable short-term, but it will discard interesting model-induced states and may hide exactly the risk-relevant trajectories we care about.

Preferred path: fix MCTS edge validity in `BattleScumSearcher2`.

## Reproduction Commands

Rebuild simulator:

```bash
scripts/build_lightspeed.sh
```

Run direct failing seed:

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_batch.py \
  --agent mlx \
  --seeds 2 \
  --max-decisions 80 \
  --battle-simulations 100 \
  --max-tokens 256 \
  --temperature 0 \
  --max-retries 1 \
  --output-dir data/qwen_patchcheck_100_256 \
  --overwrite
```

Check failure:

```bash
cat data/qwen_patchcheck_100_256/mlx_Qwen3_4B_4bit_nothinking_256/seed_2.error.json
```

## Broader Qwen Result Context

The long 10-seed Qwen pass is:

```text
data/qwen_long_100_256/mlx_Qwen3_4B_4bit_nothinking_256
```

Summary:

- 10 seeds: `2,3,4,5,6,7,8,9,10,12`
- 569 total decisions
- 99.5% valid action rate
- 5 total retries
- 3 invalid model actions, all handled by fallback
- 1 simulator error: seed 2, Entropic Brew/potion-action issue
- Seeds 3 and 4 reached Act 2
- Most other seeds died in late Act 1

This means the LLM loop is viable. The next blocker is simulator search robustness, not model parsing.

