# Fresh-start training plan — approved 15 October 2026

This plan is for a new non-LLM agent trained from scratch on environment 0.1.5.
It uses a new structured encoder for both combat and out-of-combat (OOC) play,
warm-started by copying teachers and then improved with a training setup that
can actually move the policy. Sid approved it on 15 October 2026. The reasoning
behind it is in [next-study options](next_study_options_2026-10-15.md); read
that first if anything here is unclear.

## Status and authorization

- **Approved:** steps A, B and C below, plus the monitoring in step E, using up
  to about **10 hours of laptop compute**.
- **Gate C is a check-in.** When step C finishes, stop, write up the gate
  results and ask Sid before starting step D. Step D is described here so the
  design is ready, but it is not yet authorized.
- **Backups:** local only (Sid's choice). Never delete existing artifacts or
  worktrees.
- **Work here:** branch `nonllm/fresh-start`, worktree
  `/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start`,
  cut from `ws/nonllm` at `31b6379`. Commit as you go. Merge back into
  `ws/nonllm` at milestones, fast-forward only. Push only when Sid asks.
- Record progress, decisions and every result in
  [the progress log](fresh_start_progress.md); create it on first use and keep it
  newest-first. Keep the README opening paragraph pointing at the current state.

## Goal and success criteria

The goal is a fresh agent whose combat network sees the game properly and whose
training setup can learn, measured honestly.

1. **Representation:** a new structured encoder that copies the combat search
   teacher clearly better than the old hashed encoder does, on the same labels.
2. **Warm start:** a fresh composite (new combat model plus new OOC model) that
   is at least as good as the current champion on paired 4,096-world games
   under 0.1.5.
3. **A training setup that moves:** for step D, per-update policy movement
   comparable with the OOC recipe that improved (about 0.005–0.02 KL per
   update), with monitoring that shows whether it is moving and learning.

A big win-rate jump is not expected within the approved budget. Honest,
well-measured progress on 1–2, and a ready step D, count as success.

## Ground rules

- **Runtime:** `.venv` in this worktree is a symlink to the frozen research
  runtime `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv`
  (Python 3.12.13, Torch 2.14.0, NumPy 2.5.4). Never `pip install` into it. If a
  new dependency is truly needed, stop and ask Sid.
- **Environment:** do not fix the environment on this branch.
  - An environment bug or missing public field goes to `main` as its own pull
    request with a new `env-v*` tag (see `CLAUDE.md` and `WORKSTREAMS.md` on
    `main`).
  - Research-only native helpers stay in `patches/sts_lightspeed_rl_panel.patch`.
- **Faults:** simulator faults, timeouts and undefined behaviour are never
  losses. Keep process isolation and finite deadlines.
- **Long jobs:** run anything longer than a few minutes detached, with a status
  file to poll. Harness background tasks on this Mac get killed at random.
  - Use `subprocess.Popen(..., start_new_session=True)` or the existing
    `scripts/rl/queue_runner.py`, which pauses on battery, keeps the machine
    awake on AC and has a stall watchdog.
  - Give every pipeline a control file it checks between stages, e.g. stop if
    `STOP` exists.
  - Never use `pkill -f` with a pattern that also appears in your own command
    line. Kill by PID after checking with `ps -eo pid,etime,args`.
- **Native builds:** build into a new directory, never over a module a running
  job has loaded (`BUILD_DIR=...`).
- **Reproducibility:** every run records source commit, config, native module
  SHA256, parent checkpoints and seeds, as the existing `sts_rl.recording` does.
  Evaluations are greedy argmax with the single-observation inference protocol.
  Reused panels are development evidence, not confirmation.
- **Implementation method:** use the `codex-driven-development` skill for
  mechanical, well-specified pieces. Each gets a written spec, a Codex builder,
  then separate Claude spec and quality reviews, and the orchestrator commits.
  Design the encoder and model yourself, or keep them under close review.
  Run independent Codex tasks in separate worktrees.

## World-seed namespaces for this study

These are unused by earlier studies (checked on 15 October).

| Use | Seeds |
|---|---|
| Training rollouts and label collection | from 90,000,000,000 (10,000 per round or batch) |
| Teacher label collection (combat and OOC) | from 90,500,000,000 |
| DEV evaluation (4,096 worlds; first 1,024 for quick checks) | from 91,000,000,000 |
| Confirmation (embargoed; open only for a gate candidate) | from 91,200,000,000 |
| Fight-panel source worlds | from 92,000,000,000 |
| Fixed probe states for drift monitoring | from 93,000,000,000 |

Keep label-collection worlds and evaluation worlds disjoint, and split labels
into fit and held-out sets by whole world.

## Step A: adopt environment 0.1.5 (about 1 hour)

1. Merge `env-v0.1.5` into `nonllm/fresh-start`. A test merge showed conflicts
   in about 12 files. Resolve them as follows:
   - **Take `main`'s version:**
     - everything under `src/sts_env/`;
     - `patches/sts_lightspeed_python_api.patch`;
     - `src/sts_env/_backend/native_manifest.json`;
     - environment tests, e.g. `tests/integration/test_canonical_search_sampling.py`
       and `test_native_discovery.py`;
     - `tests/native/hidden_pile_fixture.cpp`;
     - `docs/canonical_search_014.md`, `docs/shop_maintenance_013.md` and
       `docs/environment_base.md`.
   - **Keep this branch's version:** `docs/research_plan.md`, the workstream
     parts of `README.md` and `CLAUDE.md`, and `pyproject.toml`, keeping its
     policy extras and the `sts_policy` package entry but taking version 0.1.5.
   - Check the auto-merged `README.md` and `CLAUDE.md` by eye.
2. Build the core module (`scripts/build_lightspeed.sh`) and the research
   module with the overlay (`BUILD_DIR=build015 scripts/rl/build_native_rl.sh`).
   - The overlay passes `git apply --check` on the 0.1.5 core patch, but has not
     been built there yet.
   - Record both module SHA256s.
3. Run the suite with the simulator required, the test-only fixture built
   (`scripts/build_test_fixtures.sh`), and the bounded smoke from `CLAUDE.md`.
   - Some research tests may pin old version strings ("0.1.3", "0.1.4",
     `sts_env_public_v2`) or old menus; update those deliberately.
   - Note each such change in the progress log.
4. **Champion on 0.1.5.**
   - Load the current champion
     (`/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/slay-spire-rl-debug-da6d22/data/rl_runs/ooc_r2_lam1/iter_0020.pt`,
     SHA256 `3d135d9bd53b92b5a5eb3dcc8bf26458140e282ee8ea63d9aaaf6ca3c2f222bb`).
   - It has never seen 0.1.5's out-of-combat potion actions. Hide those from it
     with an adapter-side menu filter (like the existing dead-action filter in
     `src/sts_rl/common.py`), recorded in every receipt.
   - Measure greedy play on the 4,096 DEV worlds: wins, mean floor, Act 1/2/3
     boss reach and pass rates, and boss entry HP.
   - This is the baseline for this study. Do not compare it numerically with
     0.1.4 results.
5. **Fight panel.**
   - Have the champion play the 92e9 source worlds and record fight starts:
     Act 1 boss, Act 2 and Act 3 bosses and elites, and Act 2 and Act 3
     hallways.
   - Record each start as its out-of-combat state just before the fight (the
     action prefix), because resampling needs an out-of-combat state.
   - The evaluator copies the `GameContext`, salts `gc.seed` with a fixed
     resample index, enters the fight and plays it, using the existing
     `src/sts_rl/panel.py` machinery (`exit_battle` is stubbed for copies).
   - Score win rate and HP lost per fight. Compare policies on identical starts
     and salts.
   - Report each panel's standard error and the smallest difference it can
     detect.
   - Size it so the Act 2 boss panel has at least about 300 starts × 8
     resamples.
   - Old banks from 0.1.3 and 0.1.4 cannot be reused: replay is
     provenance-pinned and trajectories differ under 0.1.5.

## Step B: new structured encoder and model (mostly agent time)

Build this in new modules, e.g. `src/sts_policy/entity_encoding.py` and
`src/sts_policy/entity_model.py`. Leave the old encoder untouched, since it is
the control.

**Inputs, from the 0.1.5 public observation only:**

- **Combat entities:**
  - **Player:** HP, max HP, block, energy and energy per turn, each player power
    by name and amount, stance, draw per turn.
  - **Each enemy (up to 5, in slot order):**
    - name ID, HP, max HP, block;
    - intent move ID, `intent_damage`, `intent_hits`;
    - strength, vulnerable, weak, poison;
    - each power by name and amount;
    - alive flag and slot index.
  - **Each hand card (up to 10):** card ID, upgrade count, `cost`,
    `cost_for_turn`, `free_to_play_once`, `retain`, type.
  - **Draw, discard and exhaust piles:** a multiset of card IDs with counts and
    upgrades, order-free, as order is hidden.
  - **Relics:** name ID plus counter.
  - **Potions:** per slot, with name ID.
  - **Turn history:** cards played this turn, as an ordered short sequence of
    card IDs, from `recent_actions` or the turn counters.
- **OOC entities:**
  - run context: act, floor, HP, max HP, gold, ascension, keys, boss name;
  - the deck, as a multiset of card IDs with upgrades;
  - relics, potions;
  - map: the next reachable nodes with room type and burning-elite flag, plus
    simple route statistics;
  - screen context: event ID and options, shop items with prices, reward
    options, rest options.
  - The deck, relics and potions out of combat currently exist only as text
    (e.g. `Deck: (14): {Strike,...}`). Parse that text in the adapter for now,
    with tests. A structured `inventory` field would be a separate environment
    PR on `main`; propose it to Sid only if parsing becomes a problem.
- **Vocabularies:** stable integer IDs for card, relic, potion, monster, power,
  intent-move, event and room names. Generate them from the simulator's
  enums where the bindings expose them, and otherwise from observed names, with
  an UNK bucket. Commit the vocab files. Log UNK rates as a coverage metric.
- **Candidate actions:**
  - Each legal menu entry becomes a typed action token, parsed from its
    description: play card (references the matching hand-card token and the
    target enemy token), potion (slot and target), end turn, choose card or
    reward, buy, map node, rest option, event option.
  - Long-tail free text, such as event option wording, falls back to a small
    hashed bag inside that action token only.
  - Duplicate identical cards are interchangeable, so map to the first matching
    hand token.
- **References:** `archive/nonllm/critique`, notably
  `docs/environment_public_v2.md` and its `public_entities_v4` encoder, has
  typed per-entity public features. Use it as a reference, not a dependency.

**Model, one per phase (combat and OOC), with no shared weights between
phases:**

- Token embeddings: ID embedding plus a small MLP over numeric fields, plus a
  token-type embedding.
- A small transformer encoder (start with 2–3 layers, width 128, 4 heads).
- The actor scores each action token from its own embedding, the embeddings
  of the entities it references, and a pooled state vector, through a small
  MLP.
- The critic is a separate network of the same kind, so value training cannot
  disturb the actor.
- Keep CPU inference fast. Measure decisions per second in the real collector
  against the old model before training anything big; collection throughput
  must not fall much below the current one (training speedups doc: about
  1.6× on the fast config).

**Tests:**
- encoding of real 0.1.5 observations, combat and OOC, including multi-enemy,
  duplicate cards, free-once and retained cards, potions and burning-elite maps;
- that every legal action gets a token;
- determinism, and a save/load round trip;
- a tiny overfit test as a plumbing check.

## Step C: warm-up by copying teachers (about 3–4 hours of compute)

**Data:** store raw public observations, not encoded features, so both encoders
train on exactly the same examples. Use gzip JSONL with full provenance.

- **Combat labels:**
  - **Source states:** the canonical RNG-resampled native search teacher (24
    samples × 1,000 simulations, plurality vote; `src/sts_rl/panel.py` and the
    `collect_public_expert.py` / `teacher.py` paths), on states the champion
    reaches while playing 90.5e9 worlds.
  - **Coverage:** every combat room type, including Act 1 and hallways,
    weighted towards Act 2 and Act 3 bosses and elites.
  - **What to keep:** the full vote distribution, the plurality action and the
    teacher protocol ID.
  - **Volume:** about 100–150k states. At about 0.26 s per decision across
    about 12 threads, that is roughly an hour.
  - **Check first:** measure teacher self-consistency (re-query a held-out
    subset with different sampling seeds) on a small batch, so later
    agreement numbers have a ceiling.
- **OOC labels:**
  - The champion's greedy choices on every OOC decision it meets on 90.5e9
    worlds, with OOC potion actions hidden as in step A. About 2,000 games, a
    few minutes.
  - These carry its distilled smithing and elite rules and everything PPO
    learned.

**Training:**

- Soft targets (the vote distribution) for combat and hard targets for OOC.
  Use AdamW, a held-out split by world, and early stopping on held-out loss.
- Run enough optimizer steps to converge (thousands), unlike the 32-step
  pilots before.
- **Learning curve:** fit the new combat encoder at 5k, 20k and full label
  counts, and the old encoder (`public_candidate_v2` model, trained from
  scratch with the same budget and lightly tuned) at full labels. Use the same
  splits for both.
- **Metrics:**
  - held-out agreement with the teacher's plurality choice;
  - held-out soft cross-entropy;
  - agreement on states where the teacher is confident;
  - per room type and act.
  - Report against the teacher self-consistency ceiling.
- Train the OOC student on its labels and check that it reproduces key
  behaviours: smith rate at HP ≥85%, elite choice at HP ≥90%, card-skip and
  removal rates.

**Gate C (stop here and report to Sid):**

- **G1 (representation):** on held-out labels, the new encoder's agreement
  with the teacher beats the old encoder's by a clear margin (world-clustered
  95% CI excluding zero).
  - It is also no worse on the fight panel (win rate and HP lost) when each
    combat student plays with the champion's OOC head.
  - If G1 fails, say so plainly: the representation hypothesis is not
    supported. Continue with the better encoder.
- **G2 (warm start):** the fresh composite (new combat student plus new OOC
  student) against the 0.1.5 champion on the paired 4,096 DEV worlds, reporting:
  - wins, with the paired exact test and bootstrap CI;
  - mean floor;
  - boss pass rates and entry HP.

  Target: not worse than the champion beyond noise. Also report each student
  alone, combined with the champion's other head.
- **Report:** the numbers above, the learning curves, the monitoring plots,
  surprises, and a recommendation for step D. Do not open the confirmation
  set unless a candidate clearly passes and Sid agrees.

## Step D: learning with a setup that moves (after Sid's go-ahead)

- **OOC PPO:** reuse the recipe that improved (`src/sts_rl/ooc_trainer.py`,
  `scripts/rl/train_ooc.py`), ported to the new model:
  - combat frozen and played greedily, with macro credit between OOC
    decisions;
  - reward 0.05 per floor plus +1 win / −1 defeat;
  - λ = 1;
  - 768 games per update, learning rate 3e-4, 3 epochs, clip 0.2, minibatch
    4,096, entropy 0, with the KL guard at 0.05 (in `ooc_r2_lam1` this setting
    gave 0.006–0.016 KL per update);
  - use the fast collector layout from `docs/joint_training_speedups.md`.

  Do not add a rehearsal pull towards the starting policy. Catch forgetting
  with the retention panels instead.
- **Combat:** expert iteration rather than PPO. Each round plays the current
  student on fresh 90e9 worlds, labels the states it visits with the search
  teacher (DAgger), retrains on the aggregated labels, and scores on the fight
  panel before any full-game evaluation.
- Alternate the two loops (OOC PPO with the latest combat student frozen,
  then a combat round) and keep each loop's checkpoints separate.

## Step E: monitoring (applies from step A onward)

Emit these as JSONL per update or round. Interpret them at least every 20–30
minutes while jobs run, and record each review in the progress log. A live
process or a falling loss alone does not show learning.

- **Movement:**
  - per-update approximate KL and clip fraction;
  - KL and argmax-change rate since the start, on a fixed probe set (93e9
    states, per phase and act);
  - entropy and top-action probability;
  - weight change per update.
  - **Warn** if per-update KL stays below 0.002 for three updates ("not
    moving").
  - **Alert** above the 0.05 guard.
- **Learning:**
  - training-game win rate and mean floor (stochastic; label it as such);
  - critic explained variance by phase and act, on held-out worlds;
  - for distillation, train and held-out loss and agreement per room type.
- **Behaviour:** smith rate by HP band, elite-path choice rate, card-skip rate,
  removal rate, potion use and discard rate, shop spend.
- **Outcomes:**
  - fight panel every round;
  - paired 1,024-world DEV greedy check every few updates;
  - 4,096 DEV at milestones.
  - Always print paired differences with CIs and the detectable effect size.
- **Health:** throughput (games/s, decisions/s), faults, censored worlds,
  memory (RSS), free disk (keep at least 6 GiB; 45 GiB free on 15 October).
- **Dashboard:** adapt `scripts/joint_sparse_dashboard.py` or
  `scripts/rl/dashboard.py` to show these curves with a legend. Read-only, on
  localhost.

## Budget and timeline (about 10 hours of compute, then Gate C)

| Step | Agent time | Compute |
|---|---|---|
| A: adopt 0.1.5, baselines, fight panel | about 1–1.5 h | about 1 h (8,192 champion games plus panel collection) |
| B: encoder, model, tests | most of the agent effort | minimal |
| C: labels, training, comparisons, Gate C evaluation | about 2 h | about 3–5 h |

Leave headroom for repeats. If the budget is clearly running short, stop at a
clean point and report rather than cutting corners on evaluation.

## Risks and fallbacks

- **The new encoder is slower or harder to train than expected.** Shrink the
  transformer, profile Python encoding, or batch inference across workers.
  Report the trade-off.
- **The teacher is noisier on 0.1.5, or the Curiosity fix makes Act 3 much
  harder.** Report it, adjust the label mix, and don't hide changes in
  baseline difficulty.
- **The OOC student falls short of the champion.** Distil longer, add rule
  overlays only if measured, or keep the champion's OOC head for step D's first
  round.
- **Parsing OOC text proves brittle.** Raise the structured-inventory
  environment PR with Sid.
- **Labels or states turn out privileged.** Search is native-assisted. Labels
  come from the canonical resampled teacher, but are not certified public-only.
  Never feed native bits, seeds or search values to the student's inputs.

## Key paths and identities

| What | Where |
|---|---|
| Current champion | `/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/slay-spire-rl-debug-da6d22/data/rl_runs/ooc_r2_lam1/iter_0020.pt` (SHA256 `3d135d9b…22bb`) |
| Frozen runtime | `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv` |
| 0.1.4 canonical research build (old, for reference) | `/Users/sidbaines/.codex/worktrees/public-search-canonical/SlayTheSpireAI/external/sts_lightspeed_rl_v2/build014_canonical` (module SHA256 `936734a2…1a04`) |
| Completed joint run and its records | `/Users/sidbaines/.codex/worktrees/4a3c/SlayTheSpireAI/data/rl_runs/joint_sparse_win_v2_20261015/` |
| Earlier handoff and results | [handoff](handoff/joint_sparse_win_20261015.md), [training review](training_review_2026-10-14.md), [OOC findings](rl_findings_2026-10-13.md) |
| Training speedups | [joint_training_speedups.md](joint_training_speedups.md), `configs/joint_sparse_win_v2_fast.json` |
| Environment releases | tags `env-v0.1.0` to `env-v0.1.5` on `main`; `docs/release_v0.1.5.md` on `main` lists 0.1.5's changes |
