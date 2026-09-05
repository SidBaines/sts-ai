# OOC-RL v1 (`ooc_grpo_v1`) — status and resume notes

**State: paused incomplete at iteration 7 of 12, 2026-08-20.** Nothing is
running; the original process stack is long gone. This file is the durable
record needed to either resume the run or close it out. Run outputs live under
`data/rl/ooc_grpo_v1/` (gitignored, local only).

## What the run is

Hybrid out-of-combat GRPO. The base model
(`mlx-community/gemma-4-e4b-it-bf16`, no adapter at iteration 0) makes
**out-of-combat decisions as action text** (`--output-contract action_text`)
while the built-in C++ search agent plays every combat
(`--combat-control search`, 50 simulations, deterministic). Because the search
agent's RNG is default-constructed per environment, fights resolve identically
given the same path prefix, so within-group floor variance is attributable to
the out-of-combat policy.

Reward is final floor, group-relative and std-normalized over K=6 rollouts per
world seed, on 16 fixed training seeds
(`configs/competence/ooc_rl_train16_v1.json`, split `ooc_rl_train16`);
KL-to-base 0.02, lr 1e-5, grad-accum 8, 1000 examples/iteration (≈125
optimizer updates), 12 iterations, MLX local backend. The exact argument list
is `scripts/rl_ops/ooc_rl_attempt.zsh` — treat it as the run definition.

This follows the COMP-022/023 competence work and is a rung on "train a model
to play the whole game well". The framing experiment remains explicitly on
hold.

Note on the loss: `logp_old = stop_gradient(logp_new)`, so the ratio is
identically 1 and clipping is inert. This is TRL-faithful at
`num_iterations=1` (TRL detaches the same way) — in practice the loss is
vanilla PG + KL-to-base. `mean_ratio=1.0` / `clip_fraction=0.0` are the
*correct* signatures here, not a bug. True multi-step GRPO with frozen
sampling logps would be a later upgrade.

## Results (iterations 0–6 complete)

| iter | mean floor | median | retry % | invalid deaths | notes |
|---|---|---|---|---|---|
| 0 | 16.8 | 16 | 4.3 | 22 | base policy; format-death-heavy |
| 1 | 16.8 | 16 | 0.2 | 1 | format fixed after one pass |
| 2 | **18.6** | 16 | 1.9 | 0 | reward peak |
| 3 | 17.8 | 16 | 0.8 | 0 | |
| 4 | 13.4 | 15 | 12.5 | 2 | first salted iteration; retry spike replicated on 2 stream sets |
| 5 | 15.1 | 16 | 3.2 | 0 | format re-fixed |
| 6 | 9.8 | 9 | 19.3 | **56** | format-death wave: 58% of episodes |

Iteration 7 has 29 of 96 episodes on disk and no adapter.

**Headline finding — format-drift instability.** The lenient `action_text`
parser (strips `N: ` menu prefixes, unique-prefix match, index fallback)
combined with on-policy *emitted-token* supervision (`sft_format` trains the
emitted variant whenever it resolves to the executed action) forms an
amplifying loop: leniently-resolved forms get reinforced, drift to the
parser's edge (`{"action": "1: next room Monster"}` — prefix plus paraphrase
that nothing resolves), then a death wave (floor ≈ 4, worst-in-group
advantage) slams format back. The invalid-death series 22→1→0→0→2→0→56 is a
**growing** oscillation, not a damping one. Reward is flat-to-down against the
iteration-2 peak, and by iteration 6 the curve mostly measures format dynamics
rather than play quality.

Read every number here beside the invalid rate: an unrecoverable invalid
decision ends the episode, so floors mix format compliance with policy
quality.

## v2 recipe (identified, deliberately not applied)

None of these were applied, to keep the v1 curve clean:

- Supervise the **canonical** action description rather than the emitted
  variant — or parse strictly at generation time so lenient forms never earn
  credit. This is the primary fix.
- Add a small retry penalty to the reward.
- Raise `kl_beta` (0.05+).
- Consider fewer optimizer updates per iteration.

## Resuming

Resume is stateless: `scripts/rl_ops/rl_supervisor.zsh` recomputes the resume
point from the adapters on disk every pass (latest
`iter_N/adapter/adapters.safetensors` → `--start-iteration N+1
--resume-adapter iter_N/adapter`), and `grpo_loop` skips episodes whose
`.meta.json` sidecar exists, so iteration 7's 29 finished episodes are kept.
Launch detached — harness background tasks get killed on this machine, and the
DEVNULL handles are load-bearing (without them the detached child holds the
command-substitution pipe open and `$()` blocks forever):

```zsh
cd /Users/sidbaines/Documents/SlayTheSpireAI
.venv/bin/python -c "import subprocess as sp; p=sp.Popen(['zsh','scripts/rl_ops/rl_supervisor.zsh'],start_new_session=True,stdin=sp.DEVNULL,stdout=sp.DEVNULL,stderr=sp.DEVNULL); print(p.pid)"
```

Ops state (status journal, run log, DONE file) lands in
`data/rl_ops/ooc_grpo_v1/`. Watch it with:

```zsh
tail -f -n 0 data/rl_ops/ooc_grpo_v1/run.status
PYTHONPATH=src .venv/bin/python scripts/rl_progress.py --out-dir data/rl/ooc_grpo_v1
```

wandb: project `sts-ooc-rl`, run name `ooc_grpo_v1` (one wandb run per
attempt, several share the name), plus `ooc_grpo_v1-live` watcher runs.

Pausing is `kill -STOP`/`-CONT` on the supervisor, the babysitter, and the
attempt's process group; SIGSTOP the babysitter too, or it will auto-resume
the run. See `scripts/rl_ops/README.md`.

## The outstanding deliverable

**Paired evaluation of the best adapter against the `base_hybrid` control
arm.** The best adapter is likely `iter_2/adapter` or `iter_3/adapter`
(pre-instability); iterations 4–6 are format noise, so this evaluation does
not require finishing iterations 7–12. The control arm already exists at
`data/rollouts/comp023_fullgame/base_hybrid` (60 games, mean floor 18.9,
temperature 0.7, 2 rollouts/seed over eval30). Generate the matching adapter
arm with `run_until.py --split eval` at the same K/temperature/contract plus
`--adapter-path <best>`, then `compare_paired.py` (≈1 hour total).

**eval30 is reserved for final reporting and this is that final use** — do not
burn it on intermediate checks. The embargoed 13-window Nob cohort stays
untouched.

## Incident log (2026-08-19, all resolved)

1. **Iteration 4 wedge (pre-salt).** A simulator UB infinite loop mid
   combat-search froze the single-threaded lockstep loop for 100 minutes. Seed
   screening does not transfer across policies (the hang is state-dependent).
   Wedged partials parked at `iter_4/rollouts.attempt0-wedged/`.
2. **rc=137 kernel memory kill** at the trainer-load boundary mid-iteration-5,
   likely aggravated by six hours of SIGSTOP swap accumulation. The supervisor
   auto-recovered in 43 seconds.
3. **Two stall-kills** on iteration 5 rollouts (30+ minute combat-search
   freezes; 2 of 3 salted passes affected under the iteration-4 policy).
4. **Append-concatenation corruption**, found via a 122-line floor-6 episode
   with `decision_index` resets. Iteration 5 was scrubbed and regenerated
   clean. This produced the three-layer fix now in the harness (slot unlink,
   per-episode resume, contiguity tripwire). Salvaged attempt-2 statistics
   before the scrub: iteration-4 policy second sample, retry 3.7%, invalid
   0.00%, n=64, mean floor 14.4.

## Backlog parked with this run

- The `adapter_fc` composite whole-game arm
  (`data/rollouts/comp023_fullgame`; stages are idempotent).
- Out-of-combat teacher routes B (playout-based OOC search teacher) and C
  (self-imitation format unification) — see the whole-game arms section of
  [`experiment_history.md`](experiment_history.md).
- Shared-adapter seeds 0 and 2 robustness; Lagavulin/Sentries state-sanity
  audit.
- The framing experiment (explicitly held).
- COMP-022 shared combat adapter, for later composite work:
  `data/competence/comp_022/adapters/dense3_action_text_seed1_step6000`
  (plus its fused model directory).
