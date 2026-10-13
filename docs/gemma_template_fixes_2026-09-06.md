# Gemma combat template repairs and paired rerun

This implements the ten findings in the [template audit](gemma_template_audit_2026-09-06.md) as **`combat_public_v4`**. The frozen v3 glossary, turn-math renderer, prompt framing, native binding, and simulator binary are unchanged. There is no schema migration or change to simulator seeds/action execution. New records identify their observation version and hash both the old and new presentation modules.

| Audit issue | Repair |
| --- | --- |
| Action versus turn ambiguity | Explicitly explain one action per response, repeated choices within a turn, and end-turn resolution |
| Base/upgraded number ambiguity | Explain `+`, base/upgraded parentheses, and already-modified inline damage |
| Intent punctuation | Parse the move identifier without consuming the following comma; retain existing effect definitions |
| Nob Skull Bash | Explain the 2 Vulnerable rider |
| Missing relic effects | Add the ten audited combat relic rules; explain referenced statuses even before activation; contextualize Blue Candle curse play and Paper Phrog Vulnerable |
| Active Duplication | Explain copying and remaining charges independently of potion ownership |
| Flame Barrier+ | Render instance-specific 12/16 base Block and 4/6 retaliation |
| False lethal verdicts | Remove both positive and negative verdicts; identify the restricted fixed-cost/current-damage arithmetic |
| Incomplete HP forecast | Replace the forecast with explicitly scoped incoming-attack arithmetic; name excluded effects and state that it does not predict HP loss |
| Selection warnings and GAMBLE label | Explain pending selections; suppress the misleading playability warning; consistently rename the exact no-discard action to `keep current hand (discard no cards)` |

The repairs describe the installed simulator. In particular, displayed battle turns start at zero (Captain's Wheel triggers at displayed turn 2; Stone Calendar at 6). The simulator removes one unused Duplication charge after the enemy phase. The readable GAMBLE label does not add any discard choices or change action bits.

## Implementation

- `src/sts_ai/glossary_v4.py`: pure combat renderer with nonmutating overlays on frozen glossary definitions; noncombat delegates unchanged.
- `src/sts_ai/turn_math_v4.py`: scoped arithmetic using the existing public input types, without outcome claims.
- `src/sts_ai/combat_public_v4.py`: adapter around a v3 environment; preserves native actions/state and changes public labels and observation identity.
- `rollout.prepare_decision`: routes v4 views to the new renderer.
- `fight_lab.evaluate_worker`: verifies the full original v3 source prefix first, then wraps that same live environment for v4 evaluation.
- `provenance.py`: records hashes for both layers, the routing code, and turn arithmetic.
- `scripts/sample_fight_bank.py`: supports `--baseline-run` and `--combat-observation`; freezes the attempted cohort, verifies baseline artifact hashes/settings/model/seeds, and snapshots implementation files.

## Validation and review

Validation artifacts are in `outputs/gemma_template_fixes_2026-09-06/`:

- `tests.log`: **847 tests passed, 11 skipped**, including regressions for all ten findings.
- `smoke.log`: bounded seed-3 random rollout, 50 search simulations, Act 1; completed in 55 decisions.
- `cohort_preflight.json`: **202/202** starts verified in isolated processes. Each full original prefix matches, then v4 preserves public game state and action indices/bits except the declared labels/version. No native rebuild was needed.
- `verify_cohort.py`: rerunnable source-prefix and presentation check, invoked with `PYTHONPATH=src:. .venv/bin/python outputs/gemma_template_fixes_2026-09-06/verify_cohort.py`.
- Native integration tests compare v3/v4 execution through Gambling Chip and following actions, and verify deterministic traces in independent processes.
- `model_preflight/comparison.json`: two fresh MLX processes produced identical three-decision traces (excluding latency) on the audited Gambling Chip start. All actions were valid, including the repaired no-discard choice. This bounded check intentionally stops after three decisions and is separate from the full fight rerun.
- `independent_review.md`: **approved** separate review of the combined changes, including 41 targeted tests and independent exact replay of 89 saved decisions across six audited fights. The reviewer found the inactive Incense Burner/Intangible definition gap; a new regression and status-reference expansion address it.

## Paired rerun protocol

Original arm:

```text
data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1
```

New arm:

```text
data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v4_v1
```

The cohort contains **202 starts**: all 201 original finished attempts (wins, deaths, invalid stops, and timeout) plus the interrupted fight. It excludes unattempted bank fights. The schedule retains the original order; each fight keeps its original sampling seed. The partial original fight is explicit in the cohort and must not enter a completed-pair comparison until handled under an appropriate missingness rule.

Model: `mlx-community/gemma-4-26b-a4b-it-4bit`, revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. Temperature **1.0**, top-p **0.95**, top-k **64**, **thinking off**, 4,096 maximum completion tokens, 1,000 maximum decisions, no formatting retries, no adapter. Identical seeds do not imply identical sampled actions after changing the prompt.

```bash
PYTHONPATH=src .venv/bin/python scripts/sample_fight_bank.py start \
  --root data/experiments/single_fight_v1/search_10min_2026-09-06 \
  --baseline-run data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1 \
  --combat-observation combat_public_v4 \
  --output data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v4_v1
```

Use a fresh directory for `start`. Each decision is appended as it arrives, each result is atomic, and the existing watchdogs/stop/resume protocol applies. The run's `progress.json` and `report.json` are the source of truth for completion, failures, and yield. The original arm remains paused and its traces are not rewritten. Keep runtime files unchanged during the new run: resume validates their hashes.

The paired run was launched after independent approval, the final 847-test suite,
all-start replay validation, and the fresh-process MLX repeat. It runs in the
background; use the new arm's `status.sh`, `stop.sh`, and `resume.sh` helpers.
Launch is not a claim of completed sampling: consult `progress.json` for yield.

This combined intervention tests whether these interface repairs help on the audited cohort. It does not isolate the effect of each repair or establish generalization to unseen fights/enemies.

### Interim residual failures (overnight comparison)

The first 153 finished v4 thinking-off attempts include four action-copying
failures on the unchanged literal menu option `{ EXHAUST_MANY none`: the model
returns `EXHAUST_MANY none`, omitting the opening brace. A fifth invalid response
combines one Green Louse target with the damage text from a different legal
option. These are residual failures in the current interface; the active arms
remain frozen for the matched comparison. Exact responses and legal menus are
saved in `outputs/gemma_overnight_monitor_2026-09-06/interim_failure_review.json`.
They do not establish how a further label repair would affect outcomes.

Token-limit diagnostics also need care: the saved completion-token count
re-tokenizes decoded text. The old arm has one saved count at 4,096 and seven
within eight tokens below that limit (including two invalid replies). Its frozen
sampler reports only the count at/above 4,096. Near-limit counts indicate possible
truncation but do not preserve the original generation stop reason; the final
comparison should state that measurement limitation explicitly.
