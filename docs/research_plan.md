# Environment base goals and status

`main` is the reusable environment release, extracted from the research
harness. Its source of truth is
[the environment contract and maintenance plan](environment_base.md). Research
goals, experiments, and artifacts live on workstream branches; see
[WORKSTREAMS.md](../WORKSTREAMS.md).

Release 0.1.0 passed all local gates on 11 October 2026: a dependency-free
stepping API, optional Gymnasium wrapper, three control modes, custom-agent CLI,
pinned native build, replayable public traces, and bounded native fault handling.
The immutable release reference is `env-v0.1.0`;
[the verification record](release_v0.1.0.md) contains the 191-test result and
CLI, replay, privacy, vectorization, and packaging evidence.

Subsequent work maintains tested environment fixes here with new immutable
release tags. Model, encoder, reward, and training experiments belong in adapters
or experiment branches started from a release tag. Known simulator and Act 4
public-state limitations remain documented in the contract.

Release 0.1.1 corrects the combat character-class initialization used by
Entropic Brew and rejects reported undefined behavior in automatic combat before
applying battle outcomes. Its new binary is built separately; 0.1.0 artifacts are
preserved. [The verification record](release_v0.1.1.md) describes the
regressions and intentional seeded-trajectory/provenance boundary. The immutable
release reference is `env-v0.1.1`, validated with 195 base tests, 307 experiment
integration tests, a bounded native smoke, and repeated affected-world checks.

Release 0.1.2 validates the supported Ironclad Discovery path instead of
rejecting every use with the upstream unsupported-behaviour marker. It preserves
explicit faults for unsupported classes and invalid offers. The
[repair and verification record](discovery_fix.md) documents exact development
replays, semantics regressions, determinism and the intentional provenance
boundary; [the release record](release_v0.1.2.md) adds the release-branch
rebuild. The immutable release reference is `env-v0.1.2`.

Release 0.1.3 repairs upstream shop pricing and ineffective potion purchases.
Shop menus change, so the observation label advances to `sts_env_public_v2`; see
[the migration record](shop_maintenance_013.md). The immutable release
reference is `env-v0.1.3`.

Release 0.1.4 canonicalizes hidden pile order before seeded native search
samples, so sampled search cannot depend on the true draw order. Ordinary
execution is unchanged; see [the search record](canonical_search_014.md). The
immutable release reference is `env-v0.1.4`.

Release 0.1.5 fixes the remaining known environment faults after checking each
against the real game (the 0.1.4 "phantom powers" were the burning elite's
buff). It changes menus, observation text and some game logic, so the
observation label advances to `sts_env_public_v3`; see [the release
record](release_v0.1.5.md). The immutable release reference is `env-v0.1.5`.

Release 0.1.6 corrects the out-of-combat potion line, which hid potions after
an earlier slot emptied, and tags episode errors as agent or environment
errors. The observation label advances to `sts_env_public_v4`; see [the
release record](release_v0.1.6.md). The immutable release reference is
`env-v0.1.6`.
