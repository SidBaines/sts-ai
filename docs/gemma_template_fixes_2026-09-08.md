# Gemma parser and combat-public-v5 repairs

Implements the fixes proposed in the [8 September analysis](../outputs/gemma_analysis_2026-09-08/REPORT.md).
Inference remains paused. No historical outcome, sampling seed, native binary,
v3/v4 renderer, or schema dataclass was rewritten. New fight-lab evaluations
default to `combat_public_v5`; v3/v4 remain explicit compatibility choices.

| Problem | Resulting behavior |
| --- | --- |
| Braces/quotes in reasoning hide valid JSON | Parser `final_channel_v2` locates the answer channel before scanning JSON; final spans retain their offsets into the original response. Raw response, extracted answer text, and parsed object are recorded separately in existing metadata. |
| JSON inside reasoning can be mistaken for an answer | Preserved, unclosed thought channels are rejected even when they contain legal-looking JSON. Closed channels with absent/malformed final answers never fall back into reasoning. Stripped-marker historical responses retain their compatibility fallback because their boundary is unavailable. |
| Training repeats the whole-response scanner bug | SFT action-span extraction shares the inference extractor, preserving action-mask alignment with the final answer. |
| Truncation missed by token-count proxy | Both analysis and new budget runtimes count saved truncation flags and unclosed preserved thought markers independently of re-tokenized length, alongside the existing limit/near-limit indicators. Derived tables retain the original flag separately. |
| Awkward multi-card selection labels | V5 renders `exhaust no cards` and `keep current hand (discard no cards)`. Nonempty masks name the actual hand indices and full card instances. Legal indices/bits, candidate lists, returned actions, and current-turn history stay consistent. |
| Missing slime mechanics | V5 explains the half-maximum-HP split trigger, replacement of the current intent, offspring types, and inherited current HP at splitting. |
| Missing pre-activation buff amounts | V5 states exact Fungi Beast Grow, Jaw Worm Bellow, and Nob Bellow amounts, using public ascension and the installed simulator's thresholds. |
| Ambiguous Rampage and energy notation | `Rampage=5` is explained as a +5 bonus and 13 unmodified damage. Energy `A/B` explicitly means current energy / normal per-turn recharge, with no cap implied. |
| Implicit round rules | V5 describes normal hand discard/draw, Ethereal, Block expiry and Barricade/Blur/Calipers exceptions, energy recharge/Ice Cream, and absent zero-valued Strength/Dexterity. |
| Base/upgraded values repeatedly misread | Card definitions retain each instance's name and resolve paired numbers: `Defend: Gain 5 Block`, `Defend+: Gain 8 Block`. Conditional upgraded effects have explicit overrides. Growing card values and Whirlwind's upgraded damage are resolved. |
| Block arithmetic left implicit | Supported fixed-Block cards in hand show a scoped calculation using current Dexterity, Frail and No Block. It excludes other play triggers/copies and later changes; variable effects such as Entrench are left as rules. |

The new wrapper still uses a v3 environment internally. Source-bank replay is
verified before wrapping. This is a presentation change with unchanged native
decisions, not a simulator migration. Unknown future outcomes, draw order,
RNG state, and search labels remain outside observations.

## Important correction to the preceding analysis

The full parser regression audit identified two previously accepted replies
that never closed their thought channel:

- `seed_4012_r0_d213`, JSONL line 12: a Strike action copied from unfinished reasoning.
- `seed_4101_r0_d141`, JSONL line 7: a Headbutt selection copied from unfinished reasoning.

Together with the already identified Nob truncation, there are **three unclosed
thinking responses out of 127**, not just the one flagged by old metadata.
The corrected diagnostic reports three. These are channel-completeness/budget
indicators; the exact generator stop reason was not saved.

The parser separately recovers the valid final answer at Slime Boss line 29.
Because that trajectory also passed through the earlier thought-only action,
the new parser would have stopped sooner in a literal replay. No historical
win or loss is retrospectively changed. Future outcomes require a fresh arm.

## Validation

Artifacts: `outputs/gemma_template_fixes_2026-09-08/`.

- **889 tests pass, 11 skipped**, including new parser, training-span, diagnostic,
  v5 rule/selection, native execution, and cross-process determinism regressions.
- The required bounded seed-3 random smoke rollout completes in 55 decisions.
- `cohort_preflight.json`: **202/202 source starts** pass exact original replay,
  then v5 preserves structured public state and native action identities.
- `trajectory_verification.json`: **212 saved v4 attempts / 4,130 decisions**
  replay successfully, preserving the previous v4 prompts and exposing valid v5
  views. Parser outcomes change only in the three documented cases above.
- Native fixtures check Defend, Defend+, Flame Barrier+, Iron Wave with Dexterity,
  and optional zero-card exhaust execution. The installed simulator applies
  its Block modifier twice for Iron Wave; v5 deliberately matches that behavior.
- The comparison gallery is regenerated with corrected truncation diagnostics;
  historical fight outcomes are unchanged.
- `pre_fix_runtime/` and `pre_fix_runtime_sha256.json` preserve the source before
  these repairs. The old manifests continue to identify the old implementation.

The tests establish renderer/parser correctness and replay consistency. They
do not establish that Gemma will play better with v5. No new model generation
was run during this fix, and the 16,384-token thinking budget was not raised.

## Using the fixes

The fight-lab GUI's next newly launched evaluation uses the new CLI default.
The old three-column trajectory viewer continues displaying saved trajectories.
To explicitly select v5 for a new evaluation:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.fight_lab eval \
  --root data/experiments/single_fight_v1/search_10min_2026-09-06 \
  --selection /absolute/path/to/selection.json \
  --output /absolute/path/to/a/new-run \
  --combat-observation combat_public_v5
```

To create a new matched no-thinking arm when sampling is requested again:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.sample_fight_bank start \
  --root data/experiments/single_fight_v1/search_10min_2026-09-06 \
  --baseline-run data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v4_v1 \
  --output data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_public_v5_v1 \
  --combat-observation combat_public_v5
```

These are launch commands, not commands executed by this repair. The sampler
keeps temperature 1.0, top-p 0.95, top-k 64 and the original per-fight seeds.
Use a fresh arm: the paused v4 manifest correctly rejects the changed runtime.
Do not rewrite its fingerprints to bypass this check. An exact old-runtime
resume would require the preserved source in an isolated compatible checkout.

No schema migration or dependencies are required. Derived analysis gains
`reported_thinking_truncated` and corrected `thinking_truncated` fields;
new agent metadata names the parser version and answer text. Frozen snapshots
retain their own calculated data and source files when re-rendered.
