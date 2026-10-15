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
