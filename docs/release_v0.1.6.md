# Environment 0.1.6: higher ascensions

The 0.1.6 candidate supports Ironclad ascensions 0–20 through the shared
environment, with corrected shop prices, difficulty-aware combat text, explicit
fight boundaries and regression coverage. It is prepared for review; no release
tag has been published.

## Plan and implementation

The work follows the core-only scope Sid requested on 23 October 2026:

1. Repair Ascension 16 shop inflation in the tracked native patch, including
   Courier restocks and Membership Card interactions.
2. Exercise difficulty thresholds and the Ascension 20 ending through real
   native rules; use synthetic decks/frontiers only to reach bounded fixtures.
3. Expose consecutive fight boundaries, retain difficulty in metadata and
   numeric inputs, and correct enemy descriptions retaining base-level amounts.
4. Version the interface, rebuild independently, verify determinism/replay and
   document migration. Leave both workstreams and running studies untouched.

The native changes remain in `patches/sts_lightspeed_python_api.patch`; no
research overlay, training library, model or experiment enters this release.

## Changed behavior

**Goods cost 10% more at Ascension 16 and above.** The previous native code
applied a 20% discount instead. Initial cards, relics and potions now receive
inflation before relic discounts; Courier restocks also receive it. Removal is
a separate service and retains its existing prices. Regressions cover stock,
sale prices, both discount relics, replacements and the price actually charged.
The intended inflation is documented in
[Mega Crit's Weekly Patch 46](https://store.steampowered.com/news/posts/?appids=646570&enddate=1540508225&feed=steam_community_announcements).

**Enemy effect text follows ascension.** The pure `glossary_v6.py` renderer
corrects the known Acts 1–3 move clauses whose amounts change: buffs, debuffs,
healing, Block, summons and status cards. Examples are Sentry's three Dazed
cards at Ascension 18, Lagavulin's two points of Strength/Dexterity loss and
Time Eater's extra Slimed cards and Frail at Ascension 19. Ascender's Bane
explains Innate and Ethereal, including Blue Candle. Values follow the patched
native handlers, not a sampled future. Earlier renderers/tables are unchanged.
Ordinary Ascension 0 gameplay and text are unchanged.

**Fight identity is public.** `observation["state"]["combat_id"]` is 0 before
any combat and increases once per fight in that reset's run. It retains its
last value outside combat, counts automatic fights and resets on environment
reset. Repeated reads and native-action previews do not advance it. It contains
no RNG value, future encounter, draw order or private counter.

At Ascension 20, a step can end the first Act 3 boss and start the second with
phase `combat` on both sides. Their combat identifiers differ. The first boss
gives no terminal win reward. Both must be defeated for Act 3 victory, including
when all three keys carry the native run into Act 4. Losing the second boss is
a defeat. `max_act=4` remains an explicit experimental choice.

**Difficulty is retained in metadata and numeric inputs.** Reset/step `info`
includes `ascension`. CLI evaluation summaries include `env_config` and the
environment/interface identity; trace headers already record configuration.
Gymnasium appends raw ascension, 0–20, as feature sixteen. Its first fifteen
positions and action mask are unchanged.

## Migration

- Package/environment: `0.1.6`; public observation: `sts_env_public_v4`.
- Trace structure remains `sts_env_trace_v1`. Version/provenance checks still
  reject incompatible observations and builds before replay.
- Build a new native binary using this checkout's script. Never rebuild over
  a binary used by a running or frozen study.
- Preserve historical traces, fight banks and checkpoints with their original
  environment. Re-baseline on this release. Ascension 16+ trajectories change
  intentionally through shop prices.
- Numeric Gymnasium policies trained on fifteen inputs need an explicit
  sixteen-input migration or an adapter retaining the first fifteen features.
  Richer encoders choose their own input contract over the full observation.

Workstream adoption is separate: propagate ascension through task constructors,
collection, evaluation and selection gates. Within one build/protocol, identify
worlds by `(ascension, world seed)`; the same seed can generate different maps.
Keep related seeds in the same training/test partition when mixing difficulties.
Detect new fights by `combat_id`, including combat-to-combat transitions, and
deliberately update environment-version checks. Neither workstream changes here.

## Coverage and verification

| Rule | Core regressions |
|---|---|
| 0–20 | Starting HP, max HP, curse, potion capacity and public API difficulty |
| 1 | More elites, with the same rule at later ascensions |
| 2–4 | Normal, elite and boss damage thresholds |
| 5 | Reduced healing between acts |
| 6–11, 14 | Starting damage, enemy HP, curse and potion slots |
| 12 | Fewer upgraded rewards in both later acts on paired native draws |
| 13 | Reduced boss gold |
| 15 | Existing regressions for supported unfavorable-event cases |
| 16 | Initial and restock prices with both discount relics |
| 17–19 | Flight, Enrage, Curiosity and difficulty-aware effect text |
| 20 | Consecutive fight identity, keyed endings and second-boss defeat |

Native checks compare displayed text with executed Lagavulin debuffs, Slime Boss
status cards and Time Eater status cards. Renderer tests cover other thresholds
and newly referenced status definitions. Fresh CLI processes compare traces and
replay at ascensions 1, 10, 15 and 20, including all three modes at Ascension 20.

Validation on 23 October 2026:

| Check | Outcome |
|---|---|
| Full suite, simulator and native fixture required | 303 tests pass, no skips |
| Existing Ascension 0 trajectories | All eight full recorded games match after projecting out the documented v4 additions/version labels; historical fixture unchanged |
| Fresh-process determinism | All six ascension/mode combinations above match and replay |
| Native patch | Fresh apply, exact source match and reverse apply pass on a clean pinned checkout |
| Wheel | Core imports without optional dependencies; final renderer included; Ascension 20 twelve-step prefix truncates at its bound and replays |
| Effect-clause sweep | All 584 applicable ascension/rendering combinations include the revised clause |

The initial-price/restock regressions first failed against the original core
binary (14 failing subcases); all pass with the new patch. The random baseline
smokes below use seed 3, an Act 1 bound and 50 automatic battle simulations.
They test execution and replay, not player strength. Each mode ended in an
ordinary game defeat; none had a native fault or timeout.

| Ascension | Full control | Automatic combat | Automatic out-of-combat |
|---|---|---|---|
| 0 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 1 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 5 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 10 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 15 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 16 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 17 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 18 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 19 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |
| 20 | Defeat; replay passes | Defeat; replay passes | Defeat; replay passes |

Machine logs/manifests live in the gitignored `data/ascension-verification/`.
`verification.json` identifies the final suite log, traces, wheel and native
manifest. Build and test commands are in the
[environment contract](environment_base.md#release-verification).

## Limits

- Bounded checks do not prove retail-game equivalence for every enemy, event or
  reachable state.
- Ironclad, Acts 1–3 remain the supported study scope; Act 4 remains experimental.
  Unsupported character faults and native fault guards remain.
- Existing [environment limits](environment_base.md) still apply. Simulator
  faults/timeouts remain errors, never ordinary defeats.
- No higher-ascension player win rate or training improvement is claimed.
