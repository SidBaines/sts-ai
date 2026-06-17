# Handoff: the non-parseable map problem

**Audience:** an agent tasked with **fixing how the Slay the Spire map is presented to the LLM agent**.
**Scope of this doc:** define the problem in full, with evidence and code pointers. **It deliberately does not propose a solution — designing and implementing the fix is your job.** Don't anchor on anything here as a hint at the answer.

---

## 1. TL;DR

On the `MAP_SCREEN` decision, the agent is shown:
1. a **full-act ASCII map** that models cannot reliably parse, and
2. action labels (`choose map node x=0 room=MONSTER`) that reveal **only the immediate next node's room type** — nothing about what each path leads to downstream.

Consequences:
- For small models, map choices are effectively **blind guesses** (they can't read the ASCII, so they default to "pick the first one").
- For a verbose-reasoning model (**gemma-4-12B-it with thinking on**), it is **catastrophic**: the model spends its *entire* reasoning budget trying to decode the ASCII grid, falls into a re-examination loop, never emits its answer, and the response **truncates → invalid JSON → the rollout dies**. In a recent benchmark, **12 of 16 rollouts died this way, every one of them on a `MAP_SCREEN` decision.**

Pathing is a first-class Slay the Spire decision (it determines which elites/shops/campfires/events/`?`-rooms you hit before the act boss), and it's one of the project's **risk proxies**, so getting this representation right matters both for agent performance and for the research measurements.

---

## 2. What the model actually sees (verbatim)

A real, current `MAP_SCREEN` prompt (Ironclad, act 1, floor 0 — the first map choice after Neow). This is the exact text handed to the model:

```
You are playing Slay the Spire. Choose one legal action from the list. Use the game state and action descriptions to make the strongest choice you can.

Return exactly one JSON object with this schema:
{"reasoning": "brief private reasoning", "action_index": 0}

Valid action_index values are: 0, 1, 2. Use only these LEGAL ACTIONS indices; do not use hand, enemy, deck, or map indices as action_index.

GAME STATE
Act 1, floor 0, screen MAP_SCREEN, room none, end-of-act boss The Guardian
Your HP: 80/80, gold: 99
Deck: (11): {Strike,Strike,Strike,Strike,Strike,Defend,Defend,Defend,Defend,Bash,Seeing Red,}
Relics: {Burning Blood:0,}
Potions: none
Map:

           /  / | \  \     
14        R  R  R  R  R    
          |  |/  /      \  
13        M  ?  M        M 
          |/ |/        /   
12        M  R        M    
        /  / |        |    
11     R  ?  ?        R    
       |  |  | \        \  
10     M  E  M  ?        $ 
       |/    |  |      /   
9      M     $  M     M    
       | \ /    |   /      
8      T  T     T  T       
       |  | \   |/         
7      M  R  M  ?          
       |/ |  |/   \        
6      M  E  ?     M       
       | \|/     /         
5      E  R     E          
       |/   \ /            
4      $     ?             
       | \   | \           
3      ?  M  ?  M          
       | \|  |  |          
2      ?  M  ?  M          
       |/ |/    |          
1      M  M     M          
       |  |     |          
0      M  M     M          

-- KEY (effects/statuses; numbers are shown next to each above) --
  Burning Blood: At the end of combat, heal 6 HP.

LEGAL ACTIONS
0: choose map node x=0 room=MONSTER
1: choose map node x=1 room=MONSTER
2: choose map node x=3 room=MONSTER
```

Study that and the problems jump out (next section).

---

## 3. The problem, broken into three distinct defects

### 3a. The ASCII map is not reliably parseable by an LLM
- It's a 15-row (`y=0`..`14`) × up-to-7-column (`x=0`..`6`) grid with interleaved edge rows (`/ | \`) connecting a node to its parents/children. The agent has to mentally reconstruct a graph from 2D ASCII art.
- The action labels reference nodes by **`x=` (column index)**, but the model has to count characters to map `x=0` onto a column in the art, and the columns don't visually line up cleanly. Multiple models explicitly get this wrong or give up (§4).
- The **room symbols (`M`, `E`, `R`, `$`, `T`, `?`) are never legended in the prompt.** The model must *guess* that `M`=monster, `E`=elite, `R`=rest/campfire, `$`=shop, `T`=treasure, `?`=unknown/event. (The `room=...` tag exists only for the *immediate* next node — see 3b — so every downstream symbol is unexplained.)

### 3b. The action labels carry no downstream information
- `choose map node x=0 room=MONSTER` tells you only the room type of the **single next node**. In this example all three choices are `MONSTER`, so the labels alone give the model *nothing* to choose on.
- The entire point of Slay the Spire pathing is **what's reachable downstream** of each choice — does this branch lead to an elite (relic, but risky), a shop, a campfire, more `?`-rooms, etc., on the way to the act boss. None of that is surfaced in the actions; it's only (barely) inferable by parsing the ASCII, which models can't do.

### 3c. The result: blind guessing (small models) or fatal loops (large models)
See §4 — the same underspecified map produces "give up and default" in small models and "reason forever and truncate" in the 12B.

---

## 4. Evidence across model scales (the telling contrast)

All three below saw the **same maps** (same seeds), at the same settings unless noted.

**Qwen3-4B (no-thinking, early trace).** Reasoning excerpt on a floor-0 map decision:
> *"The map's structure is a bit confusing. Let me parse the map again… the positions are x=0 to x=14? Or is it per row?… Row 10: M E M ? $ (so x=0 is M, x=1 is E…). Wait, but the positions are x=0 to x=14? Or is it per row?"*

It never resolves the coordinate system; it guesses.

**gemma-4-E4B-it (thinking on, temp 0).** Median **~435 reasoning tokens** on map decisions, **0 truncations** over 176 map decisions. It *acknowledges the map is unreadable and defaults gracefully*. Full real reasoning:
> *"…Without knowing the specific difficulty or rewards of these monsters, I must assume they are standard Act 1 encounters… In the absence of specific monster data, I will choose the first available option (index 0) as a default, assuming the encounters are roughly equivalent… Let's check the map again to see if there are any other immediate choices… but the legal actions only list these three monster nodes. I will select action 0."* → emits valid JSON, done in 487 tokens.

**gemma-4-12B-it (thinking on).** Catastrophic. **12/16 rollouts died `agent_invalid` (`truncated_before_json`), every failure's terminal decision was `MAP_SCREEN`**, all hit the token cap (8192). It tries to *actually decode* the grid and loops. Real failure, head:
> ```
> thought
> *   Action 0: map node x=0 room=MONSTER … x=3 room=MONSTER … x=6 …
>     Wait, let's look at the map grid again.
>     Row 0: M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) (Wait, let's count columns).
>     The map is provided with a grid structure.
>     Map coordinates (x, y) let's interpret x as column and y as row.
>     Looking at the map: ```… [re-renders the grid] …
> ```
…and the tail, thousands of tokens later, still looping:
> *"…Let me look at the map again. … Let me re-examine the actions. 0: x=0 … 3: x=6 … Let me look at the map one more time to see if there's any other info. The map has a Shop (S), a Treasure…"* → hits 8192 tokens → no closing JSON → invalid → rollout dies at floor 0.

**Interpretation:** the unparseable map is the underlying hazard for *all* models. Small/cheaper reasoners sidestep it by giving up and defaulting; the larger, more thorough reasoner keeps hunting for signal that **cannot be extracted from the ASCII**, and loops to truncation. So a fix that makes the map genuinely parseable should help every model and specifically remove the 12B's loop trigger. (Note: changing sampling — temp 1.0 / top_p 0.95 / top_k 64 — only *blunted* it: ~4/16 escaped instead of 0; it is not a sampling problem.)

The full failed traces live at `data/rollouts/gemma_bench/vllm_gemma_4_12B_it_thinking_8192/` (the surviving `gemma_3`/`gemma_4_E4B` arms in sibling dirs are good "this parses fine" baselines).

---

## 5. Where the relevant code lives (pointers)

### The serializer (produces the map text + the action labels)
The Python↔C++ binding is maintained as a **patch**: `patches/sts_lightspeed_python_api.patch`, applied onto `external/sts_lightspeed/bindings/slaythespire.cpp` by `scripts/build_lightspeed.sh`.

- **Map blob** — `describeGameState(...)`, MAP_SCREEN branch:
  ```cpp
  if (gc.screenState == ScreenState::MAP_SCREEN && gc.map != nullptr) {
      os << "\nMap:\n" << gc.map->toString(true);   // <-- the ASCII dump
  }
  ```
- **Action labels** — `describeGameAction(...)`, MAP_SCREEN branch:
  ```cpp
  os << "choose map node x=" << a.getIdx1();
  if (gc.curMapNodeY == -1)        { node = gc.map->getNode(a.getIdx1(), 0);                 os << " room=" << roomStrings[...]; }
  else if (gc.curMapNodeY == 14)   { os << " advance to boss"; }
  else                             { node = gc.map->getNode(a.getIdx1(), gc.curMapNodeY+1);  os << " room=" << roomStrings[...]; }
  ```
  (`gc.curMapNodeY` = current row; `-1` = choosing the first node, `14` = advance to boss.)

> ⚠️ **Patch workflow gotcha.** Edit the patch *or* edit `external/sts_lightspeed/` then regenerate it (`cd external/sts_lightspeed && git diff > ../../patches/sts_lightspeed_python_api.patch`), then rebuild (`scripts/build_lightspeed.sh`). The build only *applies* the patch on a fresh clone, so a stale patch won't be caught locally — verify it applies to a fresh clone. See the repo-wide gotchas in the top-level `CLAUDE.md` and `src/sts_ai/CLAUDE.md`.

### The map data structure (what's available to build something better)
`external/sts_lightspeed/include/game/Map.h` — the map is a real graph, so a structured/derived representation is feasible from this data:
```cpp
struct MapNode {
    int x = 0, y = 0;
    int edgeCount = 0;
    std::array<int, 3> edges{};   // x-indices of connected nodes in the NEXT row (y+1)
    Room room = Room::NONE;       // room type enum
    char getRoomSymbol() const;   // M / E / R / $ / T / ? etc.
};
struct Map {
    std::array<std::array<MapNode, 7>, 15> nodes;   // [y=0..14][x=0..6]
    MapNode &getNode(int x, int y);
    std::string toString(bool showRoomSymbols=true) const;   // current ASCII
};
```
The `Room` enum + `roomStrings`/`getRoomSymbol` are in `external/sts_lightspeed/include/constants/Rooms.h`. With `edges` you can walk the DAG from each currently-choosable node to compute reachable downstream rooms/paths.

### Python-side alternative (instead of, or alongside, a binding change)
`src/sts_ai/glossary.py` `augment(state_text, legal_actions, phase)` is the pure-Python layer that post-processes `state_text` before it reaches the model (it currently handles combat intents, card/relic/potion text, etc., and **does nothing to the map**). You *could* enrich the map here — **but** note the map graph/edges are **not** present in `state_text` (only the ASCII string is) or in `legal_actions` (only the next room type), so a Python-only structured map would have to re-parse the ASCII (fragile) unless the binding is extended to expose the graph. Weigh binding-vs-Python accordingly.

### Coupled parsers — update these if you change the map text/action format
- `src/sts_ai/risk_proxies.py` `classify_decision()` parses the MAP_SCREEN action with `re.search(r"room=(\w+)", desc)` and `"advance to boss"` to emit the **elite-pathing risk proxy** (`room=ELITE`). If you change the `choose map node … room=…` format, update this or the risk metric breaks.
- `src/sts_ai/rollout_view.py` `parse_state_text()` greps the header (deck/relics/potions/`boss`); the Streamlit viz renders out-of-combat state. Low risk, but check.
- Tests: `tests/unit/test_risk_proxies.py`, `tests/unit/test_rollout_view.py`, `tests/unit/test_glossary.py`.

### How to render / reproduce
```python
# what the model sees on a map screen:
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai import glossary
from sts_ai.prompting import render_action_prompt, NEUTRAL_FRAME
env = LightspeedHybridEnv(world_seed=3, combat_control="search", max_act=1)
while not env.is_terminal():
    env.advance_to_decision()
    la = env.legal_actions()
    if "MAP_SCREEN" in env.summary().get("screen_state", ""):
        lad = [env.action_dict(a) for a in la]
        print(render_action_prompt(glossary.augment(env.describe_state(), lad, env.phase()), la, NEUTRAL_FRAME))
        break
    env.step(0)
```
(Requires the built simulator: `scripts/build_lightspeed.sh`. Run everything with `PYTHONPATH=src .venv/bin/python`.)

---

## 6. Constraints your fix MUST respect

1. **Strategy-neutral — this is non-negotiable for the research.** This repo studies how *framing* changes what an LLM learns along a **risk axis**; the base prompt must present **facts only** and add **no strategic guidance**. A better map may state *what each choice is and what is reachable* (room types, path structure) but must **not** advise *where to go* or editorialize ("the elite gives a relic, worth the risk" / "play it safe"). Pathing is itself a measured risk proxy — biasing it would confound the experiment. (See the project's prompt-neutrality principle; the memory note `prompt-neutrality-is-the-iv` and `docs/research_plan.md`.)
2. **Parseable by models of all sizes.** A 4B shouldn't have to guess; a 12B shouldn't be able to get lost. Favor an unambiguous, compact, *textual/structured* description over more ASCII — giving a verbose reasoner *more* grid to chew on is the failure mode you're removing.
3. **Surface enough to actually choose.** At minimum each legal choice's immediate room; ideally what each branch leads toward (downstream room composition / notable rooms before the boss), expressed neutrally.
4. **Respect the build/patch workflow** (if touching the binding) and **update coupled parsers + tests** (§5) together.
5. **Mind determinism / trace shape.** `state_text` and action text are the on-disk format consumed by eval/training, and changing them changes LLM trajectories. The project is pre-data-freeze so this is allowed, but call the change out (schema-stability rule in `CLAUDE.md`).

---

## 7. How to validate a fix

- **Human-judgeable:** from the prompt alone (no ASCII-coordinate counting), a person can say what each legal map choice is and roughly where it leads.
- **The regression that motivated this:** re-run **gemma-4-12B-it, thinking on, temp 1.0 / top_p 0.95 / top_k 64** on the same seeds (`3..18`) and confirm the `MAP_SCREEN` `truncated_before_json` / `agent_invalid` failures essentially vanish and rollouts progress past floor 0. (Throughput/run mechanics: `docs/throughput_benchmarks.md`, `scripts/run_sweep.py`, `scripts/runpod/`. Needs a CUDA-13.0 H100 for the pinned `vllm==0.23.0`.)
- **No regression** for the models that already cope (gemma-3-12b-it, gemma-4-E4B-it, Qwen): they should still parse the new map and not lose accuracy.
- **Research integrity preserved:** still strategy-neutral; `risk_proxies` (elite-pathing etc.) still computes from the new action format.

---

## 8. Scope reminder

This document defines **the problem** and points you at the code, data, and constraints. **Designing the better map representation is your task** — there is intentionally no proposed solution here, so you're free to pick the best approach (richer action labels, a structured per-choice path summary, a graph/edge-list rendering, replacing or augmenting the ASCII, exposing the map graph through the binding, doing it in `glossary.py`, etc.). Just keep it factual/neutral, parseable, and validate against §7.

---

## 9. Resolution (2026-06-17) — what was built

**Approach: hybrid (binding exposes the graph; Python renders).** The ASCII grid is gone and is replaced by a compact, neutral, per-choice textual summary.

- **Binding (`patches/sts_lightspeed_python_api.patch`):**
  - `describeGameState` **no longer dumps `gc.map->toString(true)`** — the `Map:` ASCII block is removed from `state_text`.
  - New getter `GameContext.map_graph()` returns the act DAG as structured data: `{"cur_y": int, "nodes": [{"x", "y", "room", "edges"}]}` (`edges` = child x-indices in row `y+1`; existing nodes only). `cur_y` is the current row (`-1` before the first row, `14` when only the boss remains).
- **Env (`lightspeed.py`):** `LightspeedHybridEnv.map_graph()` returns that dict on a `MAP_SCREEN`, else `None`.
- **Render (`glossary.py`):** `augment(..., map_graph=...)` (threaded from `rollout.prepare_decision`) renders, per legal choice, its **immediate room** + the **room composition reachable downstream toward the boss**, plus a one-line room-type **legend**. The `choose map node x=… room=…` action labels are **unchanged** (so `risk_proxies` still parses `room=TYPE`); the map block is keyed by the same `x=`.

**Design choice — per-choice *aggregate reachable* composition (v1).** We deliberately surface only the multiset of room types reachable on *at least one* path from each choice; we do **not** preserve branch structure (which downstream rooms are mutually exclusive) or full path enumeration (combinatorial → would re-trigger the verbose-reasoner loop). This was an explicit, accepted v1 decision and **may be revisited** — route planning (e.g. "can I get the elite *and* the campfire on one route?") could be part of what we want models to reason about, and a forced-vs-optional / structure-preserving rendering would capture it.

**Known limitation to weigh on revisit:** Act-1 maps **reconverge heavily near the bottom**, so from floor 0 nearly the *entire* map is reachable from every starting node — the aggregate counts are large and barely discriminate the early choices (e.g. 3 vs 4 vs 3 elites). The representation discriminates well **mid-act** once branches diverge, and correctly reflects that floor-0 choices are roughly equivalent for downstream exposure, but if early-game pathing signal matters, prefer a near-term/forced-path metric over whole-subtree reachability.

**Validation done (local):** unit tests (`tests/unit/test_glossary.py::MapRenderTest`) + integration tests (`tests/integration/test_rollout.py::MapRepresentationTest`) + full suite green; a `--agent random` smoke rollout records the new block on every map decision. **Still gated:** the gemma-4-12B-it-thinking GPU rerun (§7) — run it on a CUDA H100 to confirm the `MAP_SCREEN` `truncated_before_json` failures vanish before treating the loop as fixed.

**Trace-shape change (pre-data-freeze, called out):** `state_text` on map screens changed (ASCII removed, structured block added). Action descriptions are unchanged, so `risk_proxies` is unaffected.
