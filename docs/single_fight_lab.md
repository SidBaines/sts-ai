# Local single-fight lab (6 September 2026)

New evaluations now default to `combat_public_v5` with corrected final-channel
parsing. See the [8 September fixes and validation](gemma_template_fixes_2026-09-08.md).
The existing v3/v4 experiment artifacts remain frozen and inference remains
paused. Use a fresh arm for the corrected runtime; do not overwrite an old
manifest's fingerprints to force a resume.

## Compare three saved fight trajectories

The first completed thinking fight (three Sentries, floor 10, world 4050) has a
standalone comparison at `http://127.0.0.1:8013`, with an offline copy at
`outputs/fight_comparison_seed_4050_r0_d95/index.html`. Use shared turn jumps to
align the runs, then step each column independently. The Thinking and Response
tabs show the complete saved text; the Observation tab shows the saved state
and legal menu. This viewer cannot start inference.

Rebuild with `PYTHONPATH=src .venv/bin/python -m scripts.view_fight_comparison`.
Pass `--state-id` and the three arm-directory options to inspect another fight.
Each input must be a finished, hash-verified trajectory from the same start.

The [versioned template repairs and paired rerun](gemma_template_fixes_2026-09-06.md)
use `combat_public_v4` for the 202 previously attempted bank fights. The new arm
is `search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v4_v1`; its helper
scripts `status.sh`, `stop.sh`, and `resume.sh` control this run. Thinking remains
off. The original v3 sampled arm is preserved and paused.

An overnight thinking-on follow-up is queued in
`search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v4_thinking_v1`.
It copies all 202 scheduled starts, order, and seeds from the v4 arm, enabling
`thinking=true`; temperature 1.0, top-p .95, and top-k 64 remain fixed.
The user requested a larger adaptive budget after the initial queue was set up:
the total thinking-plus-answer cap now starts at **16,384**. At least two cap
hits in the first 20 generations, with a rate of at least 10% and at least five
observations, doubles the cap to 32,768, then 65,536, then 131,072 if needed.
Each increase archives early trajectories/results under `budget_calibration/`
and restarts all 202 starts with their original seeds; the main cohort therefore
uses one final cap. `budget_status.json` records the check and ceiling condition;
`budget_history.json` records increases. The original unstarted 4,096-token
configuration is preserved in `configuration_history/initial_4096/`.
These are generation limits, not forced response lengths. Cap-hit reporting
uses each episode's actual limit. Thinking watchdog timeouts scale with the cap
to allow longer legitimate generations. The changed budget is an additional
experimental difference from thinking-off, and cap-hit counts remain important
when interpreting the comparison.
The separate `queue_thinking_fights.py` launcher waits for predecessor status
`finished`, all expected result files, and released run/model locks. Errors or
a paused predecessor do not launch the next model prematurely.

The queued arm's `status.sh` shows both queue and sampling progress; `stop.sh`
cancels the queue or interrupts sampling, and `resume.sh` explicitly resumes it.
`queue.log`, `queue_status.json`, `progress.json`, and per-decision JSONL remain
on disk. An idle-sleep assertion is tied to the queue process, including its
wait for the thinking supervisor to finish. The local Mac must remain running.
Tests and tokenizer checks are recorded under
`outputs/gemma_thinking_queue_2026-09-06/` and
`outputs/gemma_thinking_budget_2026-09-06/`; no second model was loaded while
the predecessor was active.

Open **http://127.0.0.1:8010** for the running local viewer. Its pilot collection
is `data/experiments/single_fight_v1/pilot`. The GUI filters game states, launches
Gemma from the displayed source position, exports frozen selections, and displays complete source rollouts and model/adapter
trajectories with the existing Rollout Studio combat board, timeline, actions,
and raw model responses. It serves local files only; no hosting or external
services are involved. Browser automation was unavailable in this session;
the API rendered all 89 battle starts and the completed policy trajectories,
and the page's JavaScript passed syntax and event-handler tests, but visual QA
remains pending.

## Run a selected combat from the GUI

1. Refresh the viewer and choose an interesting state from the filtered bank.
2. With **Source heuristic rollout** selected, optionally move the timeline to
   another combat decision. The run panel identifies the exact world, floor,
   turn, and decision currently displayed.
3. Click **Run Gemma from here**. This freezes just that position and runs the
   pinned model until the fight ends, with the same replay checks and defaults
   as the CLI (greedy, thinking off, 4,096 tokens, no retries).
4. Watch the saved-decision count. The completed trajectory opens automatically
   if you are still inspecting that source position. Otherwise use **View
   completed trajectory**. Both paths add it to the trajectory dropdown.

**Export this state** exports only the displayed combat position; **Export all
matches** keeps the existing filter-export behavior. The single-state sidecar
adds explicit `state_ids`; validation still verifies its collection hash and
exact source row. Existing filter selections remain byte-compatible.

The server keeps browsing responsive while a background CLI subprocess runs.
One GUI fight per collection can run at a time; another click/tab receives a
busy response. **Cancel run** kills the launcher and native/model worker process
group. There is also a 30-minute fight deadline. Status survives page refresh,
and polling never launches another fight. Outputs are fresh directories under
`runs/gui_<id>/`; `.gui_jobs/` holds launch inputs and persisted job status.
Closing the viewer server gracefully cancels active jobs; a hard restart labels
unfinished job records interrupted. The inherited file lock prevents a second
GUI model from loading while the old launcher is still alive. Separately started
CLI jobs do not use this GUI lock.

The GUI path was exercised on `seed_1004_r0_d3`: one exact state, 13 Gemma
decisions, a win, and an automatically discoverable trajectory. Its full trace
matched the previous CLI run except latency. Tests cover target selection,
duplicate submissions, status, cancellation, timeout, error reporting, reload
persistence, and the source-position run button.

## Model choice and local checks

Use [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit),
revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. The model card identifies
Google's instruction-tuned source at revision
`20da991ab4afab98e8f910c4a2e8f4fbefc404ad`. This is a fresh MLX safetensors
download, not the earlier Ollama NVFP4/MXFP8 conversion. The 15.37 GB download
fits the 48 GB M5 Pro and supports quantized-base LoRA in the existing optional
MLX environment. BF16's approximately 52 GB of weights alone would exceed the
machine's memory. Quantization is an intervention: do not merge these scores
with the Ollama results as if they used the same weights.

The snapshot is in the normal HF cache. Exact file SHA-256s and installed
runtime versions are recorded in
`data/experiments/single_fight_v1/model_manifest.json`. The existing environment
already had `mlx-lm==0.31.3`, `mlx==0.31.2`, and `transformers==5.12.1`; no new
dependencies were installed. The baseline and initial adapter settings are in
[`configs/competence/gemma4_26b_local.json`](../configs/competence/gemma4_26b_local.json).

`model_check.json` records successful loading, repeated greedy generation, one
finite-loss LoRA update, and adapter save/reload. The smoke adapter trains only
the last four layers' attention Q/O projections, rank 8: 507,904 parameters.
Q/O targeting also avoids assuming that KV-shared layers have K/V projections.
The tiny synthetic check peaked at 14.28 GB of MLX memory. It does **not** measure
memory for full combat prompts, optimizer accumulation, or a training campaign.
`mechanics_adapter/` is a disposable mechanics artifact, **not a combat-trained
policy**. No competence improvement is claimed.

```bash
# Reusable pinned downloader; output manifests must be fresh.
PYTHONPATH=src .venv/bin/python scripts/fight_lab_model.py download \
  --output data/experiments/single_fight_v1/model_manifest_repeat.json

# Optional repeat of the short mechanics check (run without another MLX job).
PYTHONPATH=src .venv/bin/python scripts/fight_lab_model.py check \
  --output data/experiments/single_fight_v1/model_check_repeat.json \
  --adapter-output data/experiments/single_fight_v1/mechanics_adapter_repeat
```

## Collect and select states

```bash
# Fresh directory; partial full-control trajectories from the existing heuristic.
PYTHONPATH=src .venv/bin/python scripts/fight_lab.py collect \
  --root data/experiments/single_fight_v1/collection2 \
  --seeds 1013,1014,1015,1016 --max-decisions 240 --max-act 1 --timeout 60

# All Nob battle starts. Spaces and underscores in enemy names are equivalent.
PYTHONPATH=src .venv/bin/python scripts/fight_lab.py filter \
  --root data/experiments/single_fight_v1/pilot \
  --enemy 'Gremlin Nob' --battle-start \
  --output data/experiments/single_fight_v1/pilot/nob_starts.json

# Any combat micro-action below half HP in held-out worlds.
PYTHONPATH=src .venv/bin/python scripts/fight_lab.py filter \
  --root data/experiments/single_fight_v1/pilot --hp-below 0.5 --split eval \
  --output data/experiments/single_fight_v1/pilot/low_hp_eval.json
```

Additional filters: `--phase`, `--act`, `--turn`, `--floor-min`, `--floor-max`.
Filters intersect. Health comparisons are strict. Battle start means the first
recorded decision of that fight, not every micro-action on turn 1. Enemy filtering
matches a case-insensitive name substring among currently living enemies; the
index retains the full enemy list, including repeated names, for composition
analysis. `--phase ''` includes out-of-combat states in the browser/export;
single-fight evaluation requires combat states.

The pilot contains 12 worlds (1001–1012), 1,898 states, 89 battle starts, 22 enemy
names, and 27 encounter compositions. 508 combat states have less than half HP.
All 12 worlds passed collection checks. Sampling uses `SimpleHeuristicAgent`
under full combat control and `combat_public_v3`; its combat fallback is the
first legal action. This is a distribution sampler, **not an expert policy**.
`max_decisions` truncation is expected for partial rollouts and is labeled as
such. Increasing `--max-act` and varying `--ascension` provides other settings
in separate, explicitly identified collections.

Each source runs in a subprocess with a hard deadline, twice independently.
The full traces must match (only generation latency is omitted from comparison),
and a third process replays and checks every recorded public choice. Simulator
errors, UB flags, divergence, timeouts, and known Nob phantom-power detections
are excluded with logs/reasons; other encounters lack a comprehensive power
sanity detector. Run exclusions are visible in the GUI. This checks the realized
trajectories; it does not fix the native simulator or guarantee cross-build
determinism for arbitrary future policies.

Collections and selections are additive version-1 sidecars; `schemas.py` and
the frozen policy observations are unchanged. Source file hashes, public-state
hashes, collection/selection hashes, native binary/patch/serializer/glossary
identities, seeds, settings, and per-run metadata are recorded. World seeds
divisible by four are eval; all other worlds are train. All states from the same
world share a split. This is a **pilot split**, not the historical embargoed
final cohort. Enemy-held-out studies must additionally exclude the evaluation
enemy from training; a world split alone does not test unseen enemies.

## Evaluate and inspect a selection

```bash
PYTHONPATH=src .venv/bin/python scripts/fight_lab.py eval \
  --root data/experiments/single_fight_v1/pilot \
  --selection data/experiments/single_fight_v1/pilot/opening_eval.json \
  --output data/experiments/single_fight_v1/pilot/runs/gemma4_opening_4096

# Same command and settings for an adapter, changing only output and this flag:
# --adapter /absolute/path/to/combat_adapter

# Restart the viewer if needed; binds only to localhost.
PYTHONPATH=src .venv/bin/python scripts/fight_lab.py serve \
  --root data/experiments/single_fight_v1/pilot --port 8010
```

Evaluation supports `--agent mlx` (default), `heuristic`, `first`, and `random`.
MLX always resolves the pinned cached snapshot; it does not silently fetch a
new revision. Defaults are greedy, no thinking, semantic `action_text`, no
retries, 4,096 generation tokens, 200 decisions, and 1,800 seconds per fight.
Use `--thinking`, `--temperature`, and `--rollout-index` for explicit follow-up
arms. Run one MLX job at a time to avoid competing model copies in memory.

Before any model action, replay reconstructs the exact source position while
checking **every** prefix state, action menu, selected action, and after-state.
It preserves production `prepare_decision` getter ordering. Changed source
files/interfaces fail closed. Fights stop immediately on exit/death; each runs
in a fresh process so a native hang cannot freeze the whole batch. Timeout/error
records remain in the report alongside complete wins, losses, invalid decisions,
and generation-cap hits. There is no automatic retry or seed substitution.
Commands require fresh output directories; partial runs remain inspectable.

Save CLI evaluation output under `<collection>/runs/<arm>/` for discovery
by the GUI. Select a matching source state, then choose a policy trajectory in
the inspector dropdown. Reload to discover newly completed external CLI fights;
GUI-launched fights appear automatically.
The timeline traverses the whole source rollout or the selected fight trajectory.
Selection export from the GUI downloads `selection.json`, which is accepted by
the same evaluation CLI.

### Initial mechanics cohort

Three opening fights from eval worlds were run at **1,024** generation tokens,
greedy/no-thinking/no-retries; this smaller cap is recorded in each run. All
three completed with zero invalid decisions. These are setup results, not an
estimate of general combat competence.

| World | Enemy | Heuristic net HP loss | Gemma net HP loss | Gemma decisions |
|---|---|---:|---:|---:|
| 1004 | Cultist | 3 | 0 | 13 |
| 1008 | Jaw Worm | 9 | 7 | 19 |
| 1012 | Cultist | 0 | 6 | 17 |

Both policies won 3/3. Net HP loss is entry HP minus post-fight HP, floored at
zero; it includes post-combat healing (e.g. Burning Blood), so it is not gross
damage taken. Source, heuristic, and model trajectories are available in the GUI.
All three Gemma fights were repeated in fresh processes: all 49 decision records
matched exactly after excluding only latency, including raw model text and game
states. Evidence: `pilot/gemma_determinism.json` and `runs/gemma4_opening_repeat/`.

**Generation caveat:** thinking-off does not guarantee action-only text. Gemma
sometimes emits JSON, reconsiders in prose, then emits revised JSON. The existing
parser uses the **last complete JSON object**. A capped response can still be
valid if it contains an earlier complete object; one pilot decision reached
1,024 tokens this way. Always inspect cap hits/raw responses beside validity.
Do not silently change the parser or stop-at-first-JSON rule between base and
adapter arms. The original pilot reports predate the additive cap-hit aggregate;
the raw traces retain the evidence.

## Next experiment

### Rerunnable interim analysis

`outputs/gemma_fight_analysis_2026-09-06/index.html` opens the latest analysis
gallery. Run `outputs/gemma_fight_analysis_2026-09-06/rerun.sh` from the repository
root to capture the latest finished attempts without changing sampler status. Equivalently,
run `PYTHONPATH=src .venv/bin/python scripts/analyze_fight_bank.py`. Each invocation
retains its own dated `snapshots/` directory, with 53 captioned figures in both
PNG and SVG, summary JSON/Markdown, per-fight and per-response CSVs, enemy counts,
raw invalid responses, and frozen input hashes. `--snapshot <snapshot.json>`
rebuilds an earlier cohort. The top-level `LATEST.txt` gives the current full path.

Figures cover denominator-aware outcomes, world-bootstrap descriptive intervals,
enemy/encounter/floor/HP coverage, same-start search comparisons, HP preservation,
fight lengths, token/runtime costs, format failures, source split and difficulty,
and all 25 enemy profiles. Active traces and archived interrupted attempts are
excluded. Completed result/trace/meta hashes and source start identities are
verified before analysis; earlier snapshots reproduce after new results arrive.
Missing timeout durations are omitted from runtime plots, not counted as zero.
The source search HP/outcomes were checked against the independent collection
audit. Validation: 806 tests passed (11 skipped), bounded seed-3 smoke passed,
and standalone image layouts were inspected.

The 117-attempt interim snapshot has 104 wins, six deaths and seven incomplete
attempts (six invalid responses, one watchdog timeout). Win fractions are 88.9%
of all finished attempts and 94.5% of the 110 resolved fights. On those matched
starts, Gemma loses 14.67 net HP on average versus search's 5.45, a paired gap
of +9.22 HP (descriptive world-bootstrap 95% interval +6.90 to +11.83). Net HP
includes post-fight healing. These are early single-sample observations, not a
generalization claim or a training result; small enemy samples remain unstable.

### Full-bank Gemma sampling (paused)

Paused at the user's request on 6 September, 20:13 UTC. The worker has exited;
the bank contains 201 finished attempts: 178 wins, 11 deaths, 11 invalid-action
stops and one timeout. The interrupted fight `seed_4237_r0_d187` retains 23
decision records and is not included in those totals. There are 1,931 pending
attempts, including this interruption, and approximately 53.2 MB of saved run
data. Resume using the arm's `resume.sh` only when requested.

The refreshed analysis snapshot is
`outputs/gemma_fight_analysis_2026-09-06/snapshots/20260906T201401581014Z/`:
all 53 PNG/SVG pairs were verified, and `yield.json` / `YIELD.md` record the pause
inventory. Win fractions are 88.6% overall and 94.2% among the 189 resolved
fights. On resolved pairs, mean net HP loss is 13.99 for Gemma versus 4.52 for
search, a +9.48 HP gap (world-bootstrap descriptive 95% interval +7.49 to +11.40).
Finished attempts contain 3,663 responses and 190,586 completion tokens.

The arm `data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1/`
samples all 2,132 source combat starts once using the pinned Gemma checkpoint,
temperature 1.0, top-p 0.95, top-k 64, thinking off, 4,096 response tokens,
no format retries, and a 1,000-decision fight cap. Invalid actions, simulator
errors, and watchdog timeouts remain explicit incomplete attempts rather than
being replaced with heuristic actions or dropped. `completed` counts actual
fight endings; `finished` counts all attempts with result files.

`scripts/sample_fight_bank.py` keeps one model loaded, saves each decision as
JSONL, and writes an atomic per-fight result plus a live aggregate every two
seconds. Use `.result.json` files or `report.json` for preliminary analysis;
an active trace can still be receiving its last JSONL line. Model trajectories
appear in the bank viewer once their metadata is available. The schedule is a
recorded deterministic shuffle, so a prefix covers multiple worlds/encounters.
Each fight's sampling seed is independently derived from its state ID and
replicate index 0. Two fresh-process preflight runs matched all 26 decisions
excluding latency. That fight used 39.59 seconds of model generation, suggesting
roughly 24 hours for the cohort at that rate; this is an initial estimate, not
an elite/boss timing benchmark.

The arm contains executable `stop.sh`, `resume.sh`, and `status.sh` helpers.
`stop.sh` stops the active worker within the polling interval and preserves its
partial trace. `resume.sh` skips finished attempts and archives unfinished files
under `partials/` before rerunning the interrupted fight with the same seed.
It checks selection, settings, source interface, code/runtime identities, and
completed trace hashes. The helper uses `caffeinate -i` to prevent idle sleep
while running. Ctrl-C also pauses a foreground supervisor. The run holds the
bank's GUI worker lock; browsing remains available during sampling.

Validation before leaving this run active: 799 tests passed (11 skipped), the
bounded random seed-3 rollout passed, and a live stop/resume preserved completed
results plus an interrupted trace whose regenerated prefix matched exactly
excluding latency. Evidence is under `outputs/gemma_sampling_preflight_2026-09-06/`.

Example from the repository root:

```bash
data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1/status.sh
data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1/stop.sh
data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1/resume.sh
```

### Search generation timing probe

`outputs/combat_generation_timing_2026-09-06/report.json` records five consecutive
fresh combats from world 2001 (floors 1–5), using `SimpleHeuristicAgent` outside
combat and native `ScumSearchAgent2` at 2,000 simulations in combat. All five
resolved as wins. Combat-only wall times were 10.84, 8.85, 6.86, 8.98, and
7.22 ms (42.75 ms total); full process/rollout/public-start capture and I/O took
130.99 ms. A linear projection is 26.2 seconds per 1,000 similar starts, or
4.37 minutes per 10,000, on this Mac. These are five early Act 1 fights, not a
whole-game throughput estimate or an elite/boss/low-HP benchmark.

The probe saves public combat-start views and the hybrid OOC trace, **not every
combat micro-action**. It therefore does not measure the full-control bank's
action-trace capture or repeated collection/replay validation cost, and its
artifacts are not automatically imported into the existing full-control bank.
The native-search continuation policy also differs from the current bank's
first-legal combat heuristic. `benchmark.py`, raw timing rows, source/start
records, and a fresh-process repeat are retained alongside the report. The
89-fight bank size reflects the deliberately bounded 12-world pilot.

Coverage figures for the pilot are saved under
`outputs/single_fight_coverage_2026-09-06/`: `index.html` is a local gallery,
`enemy_histograms_overview.png` contains all 22 enemy panels, `by_enemy/` has
individual PNGs, and the CSVs retain per-enemy counts, histogram bins, and all
89 source combat starts. `generate.py` rebuilds the figures after verifying
source hashes and reconstructing the collection index. Axes use absolute player
HP (10-HP bands) and floor (one-floor bins), with a shared enemy-panel colour
scale. Each fight counts once per distinct enemy type present: three Sentries
count once for SENTRY; mixed encounters appear under each included enemy. Thus
the enemy-panel totals sum to **114 memberships across 89 unique starts**, not
114 independent fights. Model trajectories and repeat/replay-check copies are
excluded. There are 72 train / 17 eval starts, all in Act 1; individual enemy
coverage is small (Nob 5, Lagavulin 6, Sentries 3).

### Full-trace search collection

The completed run lives at
`data/experiments/single_fight_v1/search_10min_2026-09-06/` and is served at
`http://127.0.0.1:8012` (the pilot remains on port 8010). Collection and validation
took 600.01 seconds, followed by final index serialization and audit. Seeds
4001–4239 were attempted; 227 worlds yielded 49,462 positions, including 33,981
combat decisions and 2,132 combat starts, all with completed fights (2,071 won,
61 lost). There are 1,598 train / 534 eval starts, 25 enemy types, and 55 starts
below half HP. Enemy membership counts can overlap within a fight.

Storage is 1.223 GB (1.139 GiB): 598.5 MB source files, 595.4 MB validation files,
plus the index and metadata. Nine worlds were excluded by the Nob phantom-power
detector, two by `undefined_behavior_evoked`, and the last by the time cutoff.
All their data is retained. `outputs/search_bank_audit_2026-09-06/audit.json`
records coverage, outcomes, exclusions, and a source-hash/index reconstruction
check; `combat_starts.csv` lists every included fight and its outcome. The audit
script and collector snapshot are retained for reproducibility. Validation:
792 tests passed (11 skipped), four collector unit cases included, bounded
seed-3 rollout smoke passed, and the new viewer API returned the expected index
and rendered the first combat board.

`scripts/collect_search_bank.py` expands the bank using native search at each
combat decision and the existing heuristic outside combat. Every source retains
the full canonical JSONL (public states, menus, executed actions, after-states,
decision timings), outcome/provenance metadata, and native search diagnostics
including root edges and proposed winning sequences. Opaque native Action objects
are omitted from diagnostics; their bits and descriptions are preserved. Search
diagnostics are privileged labels in agent metadata, never model observations.
This policy replans after each action with search seed 1 and 2,000 simulations,
including bosses; it is different from the earlier hybrid timing probe.

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_search_bank.py collect \
  --root data/experiments/single_fight_v1/search_repeat \
  --seconds 600 --seed-start 4001
```

The budget includes independent duplicate generation and exact prefix replay
for each included world. Each child has a 30-second watchdog, capped by the
remaining collection time. `progress.json` updates during collection;
`collection.json` and `summary.json` finalize afterward. All failed, partial, and
duplicate files remain on disk under `sources/` and `checks/`, but excluded worlds
never enter the bank index. The existing Nob phantom-power detector is included
in replay validation; this does not constitute a general simulator-correctness
audit. Act 1, ascension 0, max 1,500 decisions per world; train/eval splits remain
world-based. Sources stopped at the decision cap can contain an unfinished fight.

The time cutoff determines how many seeds are attempted on this machine; exact
cohort reproduction uses the recorded seeds/configuration, not the time limit:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_search_bank.py worker \
  --root data/experiments/single_fight_v1/search_10min_2026-09-06 \
  --seed 4001 --output outputs/search_world_4001_repeat.jsonl
```

Compare `fight_lab.trace_digest` against the source digest; only decision latency
is excluded. Use a fresh output path. These sources can be filtered, replayed,
and used for model fights with the same `fight_lab.py` commands as the pilot.

Freeze a modest baseline panel spanning enemies, entry-health bands, deck
strength, and combat-start versus mid-fight states. Establish paired outcomes
and format/cap behavior before choosing labels. Reuse the existing teacher/SFT
machinery where its task/interface contract applies; the new generic state
selection format is not automatically a legacy strict-teacher manifest. Teacher
collection and training-data construction for this new generic panel are the
next step, not completed competence training.

Start QLoRA with the recorded rank-8 Q/O recipe, batch 1 and gradient
checkpointing; measure memory on real, untruncated combat prompts before a long
run. Train only on train-world data. Keep a held-out-enemy arm and a same-enemy,
held-out-world arm, then vary health/deck/ascension settings separately. Evaluate
base and adapter on the identical selection and generation settings, reporting
complete expected cohorts and paired wins/net HP changes alongside incomplete,
invalid, timeout, simulator-error, and generation-cap rates.
