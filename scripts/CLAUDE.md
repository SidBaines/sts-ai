# scripts

CLI entry points and build tooling. See the top-level [`CLAUDE.md`](../CLAUDE.md) for project goals and repo-wide rules.

## Contents

- `view_fight_comparison.py` — builds a self-contained, read-only three-column
  fight viewer from hash-checked saved trajectories. Run with
  `PYTHONPATH=src .venv/bin/python -m scripts.view_fight_comparison --state-id …`.
  Shared jumps align recorded turns (opening is turn 0), not decision indices;
  each arm steps independently and has an explicit final-outcome position.
  A missing later turn shows the completed outcome, not a fabricated board.
  Includes full saved observations, legal menus, thinking, responses, and JSON.
  No model/simulator calls, dependencies, or live sampling routes. The HTML
  works directly from disk; serving its output directory is optional.
- `compare_fight_formats.py` — rerunnable three-arm comparison using
  `sts_ai.fight_comparison` and `fight_analysis`. Restricts the old larger schedule
  to the same 202 selected starts; explicitly separates unfinished, invalid,
  timeout, death, and win outcomes. Paired success uses jointly finished starts;
  paired HP uses jointly resolved fights and allows net healing. Confidence
  intervals resample worlds. Produces dated reports, CSV/JSON snapshots, and
  45 captioned PNG/SVG plots. `--snapshot <comparison.json>` rerenders frozen
  analyzed data without rereading live files. V4 start checking compares full
  structured public state and action identities after only the declared label
  normalization; requiring identical prompt text would wrongly reject v4.
  Plot/report pair ordering is explicit: JSON snapshots sort object keys, so
  relying on dictionary insertion order changes the paired overview figures
  when rerendering an otherwise identical snapshot.
  Cap diagnostics apply one rule across arms: saved truncation flags or unclosed
  preserved thought channels regardless of token count, counts at/above the actual
  limit, or invalid responses within eight tokens below it. These are
  indicators because saved counts re-tokenize text. The old frozen sampler's
  strict 4,096 threshold therefore differs from this derived diagnostic; the
  comparison report also lists at-limit and near-limit counts separately.
  For unattended local runs, `caffeinate -i` does not prevent clamshell sleep,
  thermal-protection sleep, or low-power sleep when the battery is exhausted.
  Do not override thermal protection. The wall-clock watchdog can mark the active
  fight timed out after waking. Check `pmset -g log` and monitor timestamp gaps
  before attributing such a timeout to the model; archive the infrastructure
  interruption before a same-seed recovery. The comparison includes recovery
  records from each arm's `infrastructure_interruptions/*/recovery.json`.
  Historical unreplayed system-sleep timeouts can be annotated in
  `infrastructure_annotations.json`; trace/result hashes must match. They stay
  in coverage counts but are excluded from paired performance and the common
  three-arm cohort. Ordinary invalid outputs and non-infrastructure failures
  remain in success-rate denominators.
- `monitor_fight_runs.py` — read-only process/trace/queue checks every 30 seconds,
  with progress and alert history under `outputs/gemma_overnight_monitor_2026-09-06`.
  Run as `PYTHONPATH=src .venv/bin/python -m scripts.monitor_fight_runs`.
  Refreshes three-arm reports after each of the first five thinking fights,
  then every ten fights, budget changes, and completion; it does not restart
  models on an observation failure. A final report requires every expected result
  and the thinking queue to finish, so pending budget escalation is not mistaken
  for a completed final arm. Model watchdogs remain in the run supervisor.

- `queue_thinking_fights.py` — separate launcher for a matched thinking-on
  follow-up without editing an active sampler's fingerprinted files. Run with
  `PYTHONPATH=src .venv/bin/python -m scripts.queue_thinking_fights ...` from
  the repo root. `prepare --after <arm> --output <fresh-arm>` copies the entire
  frozen schedule/settings and changes only thinking; `watch --output <arm>`
  waits for all predecessor attempts and released locks before invoking the
  existing resumable supervisor. `stop` cancels a pending queue or running arm;
  `watch --resume` explicitly clears STOP. A paused/errored predecessor is not
  mistaken for completion. Queue status and hashes live in sidecars; model
  results retain the existing schema. Do not change the queued launcher's file
  after preparation: its hash is checked before launch.
  An optional `budget_policy` activates `thinking_fight_budget.py`: observe the
  first 20 generations, raise frequent caps after stopping the worker, archive
  all early attempt data, and restart the full cohort at the next limit. The
  dedicated thinking process binds the existing supervisor's worker/result/
  validation hooks without editing the active sampler. Its runtime hash is
  checked separately. **The frozen sampler's result helper counts >=4,096 as
  a cap hit**; the thinking runtime overrides that count using the actual
  recorded episode max_tokens. Always launch this arm through its queue helpers,
  which select that runtime. Watchdog limits scale with the completion budget.

- `analyze_fight_bank.py` — read-only, rerunnable analysis of finished sampling
  attempts. Creates a new dated snapshot under `outputs/gemma_fight_analysis_2026-09-06/`
  with 53 captioned PNG/SVG figures, HTML gallery, CSV tables, raw invalid outputs,
  and hash-verified input manifest. `--snapshot <previous/snapshot.json>` rebuilds
  the same cohort even while collection advances. Core aggregation is in
  `sts_ai.fight_analysis`; no model or simulator is loaded. Incomplete attempts
  remain in overall win-rate denominators, while paired HP statistics use only
  resolved fights. Net HP includes post-fight healing; mixed-enemy groups overlap.
  Runtime plots omit missing durations rather than treating timeouts as zero.
  Uses the existing matplotlib/numpy/Pillow installation; no new deps were added.

- `sample_fight_bank.py` — resumable full-bank Gemma sampling with one persistent
  MLX model worker. The frozen selection includes every combat start; schedule
  order is shuffled with seed 20260906 and each state has its own recorded MLX
  sampling seed. `start --root <bank> --output <fresh-arm>` freezes settings;
  `stop --output <arm>` writes STOP, and `resume --output <arm>` clears it under
  an exclusive run/GUI lock. Every decision is appended immediately; each finished
  attempt gets an atomic `.result.json`, and `progress.json` / `report.json` update
  every two seconds. Interrupted attempts move to `partials/` before restarting
  from the same source/seed. Finished attempts (including invalid/error outcomes)
  are skipped on resume; their trace/meta hashes and implementation/runtime
  identities are checked. A 300-second no-decision watchdog and one-hour fight
  limit preserve timeout attempts and continue. Three unexplained worker crashes
  stop the supervisor. See the single-fight runbook for live run paths and helpers.
  `--baseline-run <arm> --combat-observation combat_public_v4` selects all starts
  with results or nonempty partial traces, retaining baseline order and seeds.
  Baseline settings/model/result hashes are verified and the subset is frozen;
  resume checks baseline files too. V4 replays the original v3 prefix before
  changing presentation. The paired arm and checks are documented in
  [`docs/gemma_template_fixes_2026-09-06.md`](../docs/gemma_template_fixes_2026-09-06.md).
  **Throughput is inference-bound, and existing batching changes seed semantics.**
  A read-only 59-fight / 1,171-decision v4 snapshot measured 98.6% of fight wall
  time inside agent calls, mean 1,291 prompt / 22 completion tokens. The worker
  keeps one model resident but generates serially without cross-request prompt
  cache reuse. Adjacent same-fight prompts share only 111 prefix tokens on average
  (8.6% of prompt tokens); repeated later glossary text cannot be reused as a
  causal prefix after earlier state changes. See
  `outputs/gemma_efficiency_audit_2026-09-06/measurements.json`.
  Benchmark batch sizes and memory before claiming a speedup. Existing
  `parallel_rollout.py` seeds by batch membership, so it cannot be substituted
  mid-arm while preserving the sampler's independent per-fight seed contract.

- `fight_lab.py` — bounded, process-isolated heuristic collection (two independent
  runs plus exact public-prefix replay), deterministic filters/frozen selections,
  generic single-fight evaluation from any selected combat position, and the
  local GUI (`serve`, port 8010) with single-state model launch, progress, cancel,
  and automatic trajectory loading. `fight_lab_model.py` downloads/hashes
  the pinned HF Gemma 4 26B-A4B MLX checkpoint and checks generation + a disposable
  one-step QLoRA adapter. See [`docs/single_fight_lab.md`](../docs/single_fight_lab.md).
  Use fresh output directories. Keep eval arms under `<collection>/runs/<arm>/`
  for GUI discovery. Do not run multiple MLX workers concurrently on the 48 GB Mac.

- `build_lightspeed.sh` — clones `sts_lightspeed`, applies `patches/sts_lightspeed_python_api.patch` (only if not already applied), and builds the `slaythespire` pybind module. Run this before any rollout.
- `run_rollout.py` — runs a single hybrid rollout from the CLI. Always invoke with `PYTHONPATH=src`.
- `run_batch.py` — batch rollouts over a seed range/list. Seeds come from `--seeds`, `--seed-start/--seed-count`, or `--seeds-config configs/frozen_seeds.json --split {smoke,dev,eval}`. For `--agent mlx` add `--thinking` (captures `<think>` chain-of-thought into `agent.thinking`), `--max-tokens`, `--temperature`. Use `--seed-timeout-seconds` for LLM/thinking runs (per-seed subprocess + kill) — thinking mode can hang on the seed-2-class UB and is slow.
- `run_sweep.py` — batched multi-model sweep: models × {thinking on/off} × seeds via the backend-appropriate cross-rollout orchestrator. `--backend mlx` uses `sts_ai.parallel_rollout` lockstep batching with `--batch-size` K; `--backend vllm` uses `sts_ai.streaming_rollout` continuous batching with `--concurrency` in-flight rollouts (effective concurrency is capped by the number of specs). Writes per-rollout JSONL + `.meta.json` under `<output-dir>/<agent_label>/`. One model load is reused across both thinking modes; a model that fails to load is skipped. `--enable-prefix-caching` is vLLM-only and defaults on. `run_rollout.py`/`run_batch.py` remain the serial path. **For `--backend vllm`, run ONE model per process** (vLLM does not free GPU memory between in-process loads): pass a single model per invocation and loop in the shell — `run_sweep` prints a warning and does only best-effort cleanup if given >1 model. Measured per-model rollout speeds and local-backend guidance are in the "Run planning notes" section below (`--backend vllm` also runs locally on Apple Silicon via the `vllm-metal` plugin in a separate `~/.venv-vllm-metal`; mind the macOS `spawn` `__main__` guard; detailed setup in git history, `docs/vllm_metal_local.md`).
- `run_until.py` — single-model full-game runner. Launches either `--target` fresh ascending world seeds (skipping exclusions/existing outputs) or a frozen `--split` (`--seeds-config configs/frozen_seeds.json --split eval`), each expanded to `--rollouts-per-seed` K rollouts (default 1) — the K-per-seed path feeds the lower-variance paired eval and GRPO groups. Keeps N live (`--concurrency` for vLLM or `--batch-size` for MLX). Skip granularity is the `(world_seed, rollout_index)` pair.
- `runpod/` — orchestration for running the sweep on rented NVIDIA GPUs (A40s): `setup_pod.sh` (apt deps + venv + `.[vllm]` + build sim), `run_sweep_on_pod.sh` (loops one model per `run_sweep.py` process, both reasoning arms, continue-on-failure; `CONCURRENCY` overrides the vLLM streaming in-flight cap, default 48), `sync_back.sh` (crash-resilient local rsync loop), per-pod model lists, and a `README.md` runbook (pod creation via the `runpod-spinup` skill, CUDA≥12.1 caveat, two-pod split, teardown). See [`runpod/README.md`](runpod/README.md). **Training/RL stages** (CUDA-13 H100): `run_iter2_pipeline.sh`/`run_iter2_resume.sh` (offline RWR+hinted SFT, generate→build→train→eval→compare), and the additive `eval_paired.sh` (K-per-seed base-vs-adapter eval + `compare_paired.py`), `run_offline_pg.sh` (offline signed-advantage PG stepping stone), `run_grpo.sh` (the on-policy GRPO stage). Always dry-run a GPU script before a full launch.
- `compare_models.py` — per-model/mode comparison report over a sweep tree (outcomes, behaviour proxies incl. full-block-forgone & lethal-taken from `affordances`, reasoning length/latency from per-decision tokens, invalid-JSON rate).
- `summarize_rollouts.py` — JSONL rollouts → summary CSV.
- `compute_risk_proxies.py` — JSONL rollouts → risk-event CSV + aggregate JSON (deterministic, computed from stored traces).
- `build_sft_dataset.py` — offline filtered-BC dataset builder: a rollout label dir (`run_until`/`run_sweep` output) → `{prompt, completion}` SFT JSONL + a `.manifest.json`. Joins per-trajectory reward labels, filters to act-boss-clears (binary signal), and prints the positive count / stopped-reason breakdown / invalid rate. **Competence default:** `--loss-mask action`; pass `--loss-mask completion` only to reproduce the historical full-response objective. The manifest records the mode and format/thought/action token accounting. **Sparsity guardrail:** if positives fall below `--min-positives` it engages a top-floor-quantile fallback and **refuses to write** unless `--allow-fallback` is passed. Needs a tokenizer (`--tokenizer`, via `transformers`/`[llm]`/`[train-cuda]`). See [`src/sts_ai/CLAUDE.md`](../src/sts_ai/CLAUDE.md) "Training pipeline gotchas".
- `build_teacher_sft.py` — fail-closed search-label SFT builder for legacy action-index JSON, exact displayed-action text, or search-derived turn plans. Optional explicit passes use a deterministic identity-arm schedule; action-text passes can instead sample targets from aggregated root visits with content-addressed RNG seeds. Materialized multi-pass schedule order is only advisory to the default trainer iterator: the builder emits no manifest augmentation block, so `--mlx-preserve-row-order` rejects these datasets. Keep `eval_fraction=0` for materialized multi-pass datasets, or repetitions of one source state can leak across train/eval. Turn plans are anchored to the aggregated-root-visits consensus action: among stored non-abstaining queries whose this-turn line opens with that exact displayed description, prefer an eligible matching-vote query and otherwise record `plan_source="non_eligible_query"`; if none exists, omit the state and audit it under manifest `turn_plan_skips`. This matters because consensus and raw `best_sequence[0]` disagree on about 48% of the 150 frozen COMP-012 train states (130 are anchorable and 20 are skipped, yielding an approximately 130-state turn-plan set). `--state-selection first_per_turn` chooses the minimum `source_decision_index` for each `(window_id, turn)` from the raw dense collection **before** confidence/hidden-order filtering, public-state deduplication, and `--max-examples`; if that first state is rejected, the turn contributes no example rather than falling forward to a later micro-action. Conflicting duplicate `(window_id, turn, source_decision_index)` rows are rejected, while structurally identical decoded rows are harmlessly deduplicated. Read `source_state_selection` and the `n_source_rows_*` fields separately from `selection_method`, which describes the later stable public-hash prefix. The frozen COMP-010 manifest predates the split missing-action accounting: its two `public_alias_teacher_conflict` rows (`seed_122_r0_w0` turn 5/source 114 and `seed_150_r0_w0` turn 4/source 121) are singleton ambiguous references with no consensus action and were safely excluded; this is an accounting-label correction only, so do not regenerate the frozen artifact.
- `train_policy.py` — LoRA trainer dispatch: `--backend {mlx,trl}` over `src/sts_ai/train/train_mlx.py` (local mechanics) or `train_trl.py` (CUDA/TRL). Consumes the `build_sft_dataset.py` JSONL+manifest; `--loss-mask auto` (default) enforces its recorded objective and treats legacy manifests as completion loss. MLX outputs must be fresh/nonexistent. Current `search_teacher_sft` manifests are fail-closed (v3, content hash, public-v2 observation, aggregated-root selection, canonical/legal action-only target, exact tokenizer identity); `--expected-example-count N` additionally requires that contract and aborts unless token preparation retains exactly N rows. Every strict teacher run must also pass `--mlx-model-revision` with the lowercase 40-character cached Hugging Face commit: manifest validation continues to use the logical `--base-model` ID, while MLX loads the exact resolved snapshot and records its revision, file hashes, and aggregate content identity in both adapter configs. MLX seed/LoRA/accumulation settings are explicit via `--mlx-seed`, `--mlx-lora-rank`, `--mlx-lora-scale`, `--mlx-lora-dropout`, and `--mlx-grad-accum` (defaults `0/8/20/0/1`). `--mlx-action-token-only` is an additive strict-teacher-only objective override: it preserves the frozen `loss_mask_mode="action"` artifact and full causal sequence but narrows supervision from format+action tokens to tokens categorized as the `action_index` value. `--mlx-action-token-weight W` instead retains format+action supervision and changes their relative MLX loss weight; unit weight deliberately keeps the historical boolean-mask path, while non-unit weight stores float32 token weights and normalizes by total weight mass. The weighted trainer's console “trained tokens”/tokens-sec are effective weight mass, not raw tokens; use the preparation report for separate raw counts and masses. `--mlx-preserve-row-order` is a stricter paired-control path: batch 1, no validation split, accumulation 1, and a complete content-addressed schedule are required; preprocessing and training preserve that exact row order. The overrides and hyperparameters persist in `<out>/mlx_lora_config.json`; the exact realized mask/accounting/schedule persists in `<out>/native_mlx_data_report.json`, including zero/partial retained-count failures. The resulting adapter is evaluated with `run_until.py --adapter-path`. Extras: `.[train-mlx]` / `.[train-cuda]`.
- `compare_paired.py` — paired base-vs-trained eval over two `seed_*_r*.meta.json` dirs: averages K rollouts/seed, reports the **paired per-seed floor delta + sign test + bootstrap CI** beside per-arm hack signals (win/clear rate, `agent_invalid_rate`, decisions/run, budget-truncated rate). Dependency-free (`sts_ai.eval_stats`). Use after a K-per-seed `run_until --split eval` of each arm.
- `local_task_eval.py` / `local_task_compare.py` — local-curriculum evaluation and strict matched-arm comparison. Interface digests are validated per kind: `chat_template_probe_hash` is `sft_format`'s short 16-hex probe digest, the rest are full 64-hex SHA-256s (requiring 64 everywhere silently fails every real run's postflight). Completed episodes are resumed only when their stored command/cohort contract and trace match the current invocation; streaming metas missing derived task metrics are explicitly repaired from a non-empty trace and revalidated, and a postflight requires every expected episode. Comparison requires `--manifest` plus `--rollouts-per-window K`, verifies the complete manifest split (not merely an identical subset), exact window and `(world_seed, rollout_index, policy_seed)` identities, one task/source cohort, matching interface fingerprints, and identical generation settings. The only permitted—and required-to-actually-differ—arm interventions are explicit `--allow-intervention adapter_path` and/or `--allow-intervention combat_observation`; adapter identity includes content hashes, and the JSON report records declarations and observed values.
- `build_pg_dataset.py` — policy-gradient dataset builder: raw rollouts + per-trajectory `final_floor` → `{prompt, completion, advantage}` JSONL (+ manifest). `--mode offline` (`A = floor − baseline`) or `--mode group` (group-relative by world seed). Reuses `dataset_builder`'s discovery/skip-rules + `advantage.py`. **This is the right input for PG/GRPO** (the built SFT dataset is not — RWR already filtered/replicated it).
- `train_pg.py` — clipped+KL policy-gradient LoRA trainer (CUDA, `src/sts_ai/train/{pg_loss,train_pg_trl}.py`). Consumes a `build_pg_dataset.py` JSONL+manifest; mu=1 (TRL default). The offline signed-advantage stepping stone. `.[train-cuda]`.
- `score_teacher_actions.py` — teacher-action evaluator for search-teacher SFT data across all output contracts. `--contract {action_only,action_text,turn_plan}` (default action_only, byte-identical legacy behaviour) selects candidate construction: canonical `{"action_index":i}` completions, `{"action":"<description>"}` completions over the parsed LEGAL ACTIONS menu (terminator-bounded, fail-closed), or — for turn_plan rows — action_text candidates against an instruction-rerendered prompt (`scoring_contract` recorded). `--mode {candidates,generate}` picks likelihood scoring or greedy generation (`teacher_action_greedy_generation` report kind with valid_json/matched rates; `--max-tokens` caps generation). `--per-row-out` writes per-row results as JSONL for `rescore_teacher_report.py`. Scores run under a local MLX base model or adapter conditioned on the exact stored prompt; reports candidate-normalized probability/log-probability, teacher NLL, top-1 agreement (raw-sum primary, mean-token secondary for text candidates), per-window summaries, and provenance. `--out` and `--per-row-out` refuse to overwrite (checked before the model loads). The candidate probability is a softmax over exact candidate sequence log-probabilities; it deliberately excludes probability mass assigned to free-form or malformed outputs, so always interpret it beside rollout validity rather than as an unconditional policy probability.
- `rescore_teacher_report.py` — regret/turn-set rescoring of per-row teacher reports.
- `audit_state_sanity.py` — phantom-power state quarantine sidecar. Its findings are false positives: the flagged powers are burning-elite buffs and relic/potion effects (corrected 2026-10-15).
- `run_grpo.py` — on-policy GRPO loop (`src/sts_ai/train/grpo_loop.py`): in one process, regenerate K rollouts/train-seed at temp>0 (streaming orchestrator) → group-relative advantage → clipped+KL update → **vLLM LoRA hot-swap** (sleep during training), repeated for `--num-iterations`. **Dry-run with at least two iterations first.** Rollout sampling is salted per iteration (`salt = --policy-seed-salt + iteration`, default base 0; iteration 0 unsalted = historical stream); on a resume after a simulator-hang stall-kill, bump `--policy-seed-salt` (+1000/attempt) alongside `--start-iteration/--resume-adapter` so the redone iteration samples fresh trajectories. A launch into an out-dir that already has `run_config.json` writes `run_config.resume-<N>.json` instead of clobbering the original provenance. Also runs **locally on Apple Silicon** with `--backend mlx` (default `cuda`): the same loop with an MLX LoRA PG trainer (`train_pg_mlx`) + lockstep rollouts + adapter reload — no vLLM/CUDA, for fast cheap iteration in the project `.venv` (see `src/sts_ai/train/mlx_grpo.py` + `src/sts_ai/CLAUDE.md`). Add `--thinking` for native-thinking models (e.g. gemma-4-e4b). The `--backend cuda` path is unchanged.
- `rl_progress.py` — per-iteration progress table + PNG plots for a GRPO run dir (`--out-dir`). Tolerates in-flight, partially written iterations, so it is safe to run against a live run; re-reads everything every time. Reports mean/median final floor, decision retry/invalid rates, and stop-reason mix per iteration — always read the floor curve beside the invalid rate, since an unrecoverable invalid decision ends the episode.
- `rl_ops/` — operational scripts for long unattended local RL runs: self-healing supervisor (stall watchdog + salt-bumping restarts), per-attempt launcher, power/battery babysitter, and a read-only live wandb watcher. Paths derive from the script location and are env-overridable (`RL_OUT_DIR`, `RL_OPS_DIR`, …); ops state lands in `data/rl_ops/<run>/`. See [`rl_ops/README.md`](rl_ops/README.md) — it documents the pause/resume gestures and why each moving part exists.
- `visualize_rollout.py` — Streamlit viewer for rollout traces (click-through / 1-decision-per-second autoplay). Out-of-combat decisions show deck/relics/potions + chosen action + reasoning + `<think>`; **combat decisions render a board** (enemy panels with HP/block/intent, a player panel with HP/block/energy/powers, and the hand as neutral card tiles with the chosen card + targeted enemy highlighted). Tiles are colour-by-state (chosen / unaffordable), **not by card type** — type isn't in the records (the "records only" data scope). Needs the `viz` extra. Run: `PYTHONPATH=src .venv/bin/streamlit run scripts/visualize_rollout.py`. All parsing lives in the unit-tested `sts_ai.rollout_view` (no Streamlit import there); the script is a thin render shell.
- `interactive_app.py` — launches the **Interactive Rollout Studio** (FastAPI + offline browser UI): interactively drive agents, sample N decisions per turn from any method (`user`/`first`/`random`/`heuristic`/`model`), edit the LLM framing/prompt (with saved templates), branch to explore alternatives, and cache sessions as canonical rollout JSONL. Combats are steppable (`combat_control="llm"`, default) or auto-resolved (`search`). Needs the `app` extra (+ `llm` for the local MLX model backend); runs fully offline once a model is cached. Run: `PYTHONPATH=src .venv/bin/python scripts/interactive_app.py` then open `http://127.0.0.1:8000`. Backend + core logic live in the unit-tested `sts_ai.interactive` package (see [`src/sts_ai/interactive/CLAUDE.md`](../src/sts_ai/interactive/CLAUDE.md)); board HTML is the shared `sts_ai.rollout_view_html`.

## Area gotchas

- **`collect_search_bank.py` captures every combat decision.** Its native teacher
  replans with a fixed seed and 2,000 simulations on each decision; this differs
  from the hybrid resolver's multi-action plan commits and from the pilot's
  first-legal combat policy. Native search returns v2 descriptions, so render
  its selected action with v3 computed damage before mapping to the deduplicated
  menu (raw bits can identify a duplicate hand copy absent from that menu).
  The timed budget includes independent duplicate generation and exact replay.
  All raw files, including deadline-cutoff or failed worlds, are retained; only
  fully validated sources enter `collection.json`. Reproduce a recorded world
  with `worker --root <bank> --seed <recorded-seed> --output <fresh-path>`;
  a wall-clock budget alone cannot reproduce the same cohort on different hardware.

- **Whole-game paired reports need an external completeness/contract check.** `compare_paired.py` uses the world-seed intersection without enforcing episode identities or matching generation settings; `local_task_compare.py` is stricter. COMP023 eval30 also overlaps the shared combat adapter's training worlds (23/30), and its historical hybrid control uses a different OOC output contract/retry count/backend from OOC RL. The [2026-09-05 review](../docs/repository_review_2026-09-05.md) records the artifact checks; do not interpret those comparisons as unseen-world or weights-only effects without the appropriate controls.

- **A Hugging Face model ID is not a frozen base-model identity.** Strict teacher training resolves and records an exact cached revision plus config, tokenizer, index, and weight identities; do not load a mutable repository ID when reproducing a frozen run.
- **`--max-tokens` must be large for reasoning/thinking models (default 4096).** A small cap (e.g. 256) truncates the model mid-thought, so it never emits the closing JSON: `parse_json_action` fails, the decision is marked invalid, and after retries the rollout stops with `agent_invalid` before executing any fallback action. The eval's `invalid_rate` + `completion_tokens` (pinned at the cap) make this obvious. Raising the cap is **free for no-thinking models** (generation stops at EOS ~60-90 tokens, well under the cap), so the high default costs nothing there.
- **An unrecoverable invalid decision now *ends the rollout* (`stopped_reason=agent_invalid`), so outcome metrics are only interpretable beside the invalid rate.** Floor/win-rate from `summarize_rollouts.py` / `compare_models.py` partly reflect format-compliance, not just policy quality: an arm that truncates JSON more often will score lower for format reasons. Always read the `unexecuted` / `stopped_reason` columns (and `audit_rollouts.py`) next to outcome — never compare arms (esp. thinking vs no-thinking) on outcome alone.
- **`run_rollout.py` deletes the output file if it already exists** before writing. Pass a fresh `--output` path if you need to keep a prior trace.
- Records are appended per-decision during the run; a crash mid-rollout leaves a partial JSONL file.
- **Release each diagnostic checkpoint before loading the next one.** `MlxCandidateScorer._load()` returns `(model, tokenizer)`; binding that model to a loop-scope throwaway such as `_` keeps it alive even after the scorer fields are cleared, so the next checkpoint can momentarily require two full models and OOM. Use the tokenizer-only helper, clear the scorer in `finally`, and keep final JSON reports on the atomic same-directory-temp-plus-`os.replace` path.
- **Streamlit HTML rendering in `visualize_rollout.py` has two sharp edges.** (1) Markdown treats any line indented ≥4 spaces as a code block, so each HTML tile/panel must be emitted as a **single-line** string (no pretty-printed newlines) via `st.markdown(..., unsafe_allow_html=True)`. (2) A rerun (e.g. autoplay/`st.rerun`) re-renders the page from scratch, so the `<style>` block is **re-emitted every run** rather than once — don't guard it behind a session flag or the tiles lose their CSS.

## Run planning notes (distilled from removed decision docs; details in git history)

- **Throughput (Apple M5 Pro, 48 GB):** E4B batched decode ≈ 330 tok/s at batch 32; `mlx-lm` and `vllm-metal` are at raw parity, but vllm-metal's continuous batching is ~1.3–2× faster on variable-length (thinking) rollouts because it refills finished slots. H100 per-model decisions/sec tables live in git history (`docs/throughput_benchmarks.md`).
- **Local backend choice:** `--backend vllm` (vllm-metal) shares the exact `VllmJsonAgent`/`streaming_rollout` code path with the CUDA pod; it does not load LoRA adapters directly (fuse first), whereas `--backend mlx` takes `--adapter-path`. Generation orchestration is deliberately the synchronous vLLM `LLMEngine.step()` loop.
- **Pod dry-run discipline:** before a long GPU run, copy the exact production command and shrink only the numbers; run at least 2 iterations so the vLLM sleep↔trainer GPU hand-off and wake-after-train transition are exercised; set `VLLM_USE_FLASHINFER_SAMPLER=0`; `compare_paired.py` exits 2 on an empty arm — treat that as a failed launch, not a null result.
