# Environment 0.1.5 verification record

Release 0.1.5 fixes the known environment faults that remained after 0.1.4,
plus further simulator divergences found while checking them. Each item was
first checked against the real game, because some apparent faults are real
mechanics that were only displayed badly. The 0.1.4 "phantom powers", for
example, were the burning elite's buff.

The public observation label advances to `sts_env_public_v3`. Menus, observation
text, state fields, the keyed Act 3 reward and some game logic all change, so
replay older traces and banks with their original build, and re-baseline any
experiment before comparing it with 0.1.4 or earlier.

## Changes

Observation, menu and reward changes:

- **Card flags in combat text.** Play actions and hand lines show
  `(cost N, free once)` and `(cost N, retained)`, as pile labels already did. A
  free-to-play-once card (e.g. after Forethought) no longer shares a description
  with an ordinary copy, so menu de-duplication keeps both. The glossary KEY
  explains the flags when they appear.
- **Out-of-combat potions.** Drinkable potions (Blood Potion, Fruit Juice,
  Entropic Brew) and potion discards are appended after all ordinary
  out-of-combat choices, so ordinary indices are unchanged. Descriptions give
  nominal amounts. Native automated agents and the `first` agent are
  unaffected; random and LLM agents see larger menus whenever a potion is held.
- **Burning elite and keys.** `map_graph()` has a per-node `burning_elite`
  flag and a top-level `burning_elite` position. Map actions, route text and the
  glossary mark it `(burning: Emerald Key)`, and the room legend states its cost
  (one random extra buff) and its reward. The buff stays hidden until combat, as
  in the game. Run state has `keys: {ruby, emerald, sapphire}`.
- **Keyed Act 3 wins.** Holding all three keys, beating the Act 3 boss enters
  Act 4 instead of ending the game. At `max_act=3` this was reported as
  truncated with reward 0, so collecting keys turned a win into a loss. It now
  terminates as `victory` with reward 1. `info["outcome"]` is the episode
  outcome (`PLAYER_VICTORY`), while `observation["state"]["outcome"]` keeps the
  native value. `max_act` 1, 2 and 4 are unchanged.
- **Live combat run fields.** During combat, top-level `cur_hp`, `max_hp` and
  `gold` come from the live battle rather than the battle-entry snapshot.
- **Computed damage annotations** (opt-in, `combat_public_v3`). These now
  cover untargeted attacks with several living enemies (per target), Whirlwind
  (per hit × current X), Fiend Fire (per hit × cards exhausted) and Sword
  Boomerang (per random hit). Flag-off output is unchanged. Turn math keeps Fiend
  Fire out of its fixed-damage total.
- **Glossary corrections.** Flight, Byrd's Fly intent, Mode Shift, Angry, Time
  Warp, Thievery, Plated Armor and Curiosity now describe the simulator's actual
  rules.

Game-logic changes (frozen-seed trajectories that reach them differ):

- **Out-of-combat potion fall-through.** After a potion drink or discard,
  `GameAction::execute` fell through into the current screen's handler, using the
  slot index as a screen choice. It now returns immediately, and invalid potion
  actions (including on battle screens) throw before changing state.
  Out-of-combat Entropic Brew now uses the in-combat Brew's roll: it respects
  Sozu, fills the first empty slots and never rolls Fruit Juice.
- **Events.**
  - Hypnotizing Colored Mushrooms "Eat" heals 25% of max HP (rounded down) and
    adds Parasite; it previously awarded 99 gold.
  - Mysterious Sphere "Leave" no longer awards gold.
  - The Mausoleum always grants a random relic of a rolled tier, alongside its
    existing curse roll. It now consumes relicRng.
  - The Woman in Blue charges 20/30/40 gold.
  - Designer In-Spire's random-upgrade Adjustments option charges 40 gold (50 at
    A15), like its chosen-upgrade variant.
- **Curiosity.** The Awakened One gains Strength equal to its Curiosity whenever
  the player plays a Power; the trigger was commented out upstream. The buff
  checks the holder is still alive when it resolves.
- **Intangible.** Calculated damage against an Intangible enemy is capped at 1
  (upstream raised low damage to 1 instead). Annotations and search evaluation
  change, and sub-1 calculated damage is now 0.
- **Lagavulin.** Waking after its sleep turns now clears Asleep and removes its
  8 Metallicize, as waking from damage already did.
- **Flight.** Flight is removed at 0: a grounded Byrd takes full damage and gets
  no turn-start reset until Fly re-applies exactly 3 (4 at A17+).
- **Angry.** It now triggers only when attack damage gets through Block. This
  follows our reading of the game's `AngryPower` (moderate confidence; not
  checked against decompiled source).

## Verification

Built on 15 October 2026 from the unchanged upstream pin
`7476a81954020087da31d41d16fddf475746ec2d`, pybind11 v2.13.6, CMake 4.1.3,
Apple clang 21.0.0 and Python 3.12.13 on macOS arm64, in an isolated `.venv`.

- Patch SHA256 `d1428d0125497e12a0182b067a9b07870ecf419cb134e2fc0381466572ef37fd`.
  It applies to a fresh upstream checkout and reverse-applies to the build tree;
  no pybind11 submodule hunk and no one-space blank context lines.
- Native binary SHA256
  `a4dc593bd3c64c24e11e20dfdbbd183c752b25b615e49dedd84f2be446ae15ef`; a second
  build from a fresh upstream copy produced the identical binary.
- With the simulator and the test-only search fixture required
  (`STS_REQUIRE_SIMULATOR=1 STS_REQUIRE_SEARCH_TEST_FIXTURE=1`), the full suite
  passes all 275 tests with no skips. Each fix has regression tests driven by
  real simulator play, with the test fixture used only to set up scratch states.
- The bounded random smoke (seed 3, Act 1, 50 battle simulations) ends in an
  ordinary defeat after 21 decisions.
- Fresh-process determinism: each cohort was run twice in separate processes
  and gave identical traces once wall-clock `elapsed_seconds` fields are
  excluded, with no errors:
  - `sts-env evaluate --agent native --mode ooc --max-act 3 --seeds 0:24`;
  - `--agent native --mode full --max-act 1 --seeds 0:8`;
  - `--agent random --mode ooc --max-act 2 --seeds 0:16`, which executed 63
    out-of-combat potion actions.

  `sts-env replay` verified a saved trace step for step.

The fixes were implemented by Codex and reviewed by independent Claude
reviewers for spec compliance and code quality. Each review checked the
simulator source and, where relevant, the game's behaviour.

The immutable release reference is `env-v0.1.5`. Earlier tags and binaries are
unchanged.
