# Candidate public environment version 2

Local unreleased candidate `0.2.0` on `codex/nonllm-critique`, with observation identity `sts_env_public_v2` and native binding API 2. This deliberately changes the action/observation contract. Existing releases, source banks and checkpoints remain preserved in their original worktrees. No tag or environment-base branch has been moved.

## Public changes and migration

The core keeps all currently legal native combat actions. Card-instance labels identify the hand slot, free-once and retention state; numerical cost reflects a public free-play effect. Identical text is not an equivalence proof. Native teacher bits route to the exact displayed action, including two same-name cards with different current mechanics. The legacy low-level view is retained explicitly for historical menu/prefix tests.

Every observation adds typed `state.inventory` with deck entries, upgrade/special values, relic counters, potions and visible boss. In combat, current public potion slots/relic counters replace the stale GameContext inventory copies. Combat card dictionaries add opt-in intrinsic mechanics. Every legal candidate adds a `public` dictionary containing its current source/target/card/potion/shop/reward semantics. Unavailable damage remains unavailable; it is not assigned an invented zero. The binding computes only immediate public damage arithmetic on supported attacks.

No seed, native bits, unique card IDs, future search values or hidden draw order enter observations. Hand/visible deck and target indices support current public entity joins. Draw/exhaust selection positions remain internal routing and are excluded from typed action metadata. Unordered pile aggregation remains unchanged.

Old action indices and public hashes may change. Old TaskSpecs fail current native/interface/observation provenance validation. Recollect source banks under this candidate; never change old hashes or remove provenance checks to force replay. Legacy model loading for behavior transfer is separate from bank migration: any projection/initialization must be named, tested and rebaselined on the repaired interface before learning claims.

## Native repair and artifact boundaries

The isolated patch incorporates the existing Discovery correction from `54df6ef`: ordinary Ironclad Discovery no longer raises an unconditional false fault; unsupported class generation still fails closed. The candidate extends public binding views without changing the old views' default fields. Its source/build live at `external/sts_lightspeed_v020`; use this worktree's `.venv`. Patch/build SHA values are recorded in `src/sts_env/_backend/native_manifest.json` and the build manifest.

Regression coverage includes the reached TRAIN world 12022 free/normal Strike transition, every Discovery offer and upgrade, temporary-cost reset, X-cost semantics, overflow, unsupported-class guards, current typed public ownership, private-field exclusion and fresh-process deterministic traces/replay. The historical world-2005 fault fixture depended on the now-repaired Discovery flag; it is now checked as a healthy cache-parity case. Replay fault guards receive a deterministic reported fault after a real native action rather than relying on an obsolete RNG-dependent defect.

Before accepting this candidate for collection/training, run the complete host/native suite and bounded core smoke, verify scoped fresh/reverse patch application, and preserve validation artifacts. Passing these checks certifies the exercised contract, not complete retail-game fidelity or absence of all simulator defects.
