# Environment release 0.1.5.2

This release reduces the cost of serving combat observations. Ordinary action
menus fetch the hand only when selection labels need card identities. The
history adapter reads the current turn's actions without building a full battle
summary. Backends that provide history through the summary still work.

## Compatibility

The immutable release tag and Python distribution are `env-v0.1.5.2` and
`0.1.5.2`. Gameplay version `0.1.5` and observation schema
`sts_env_public_v3` retain their meanings. Eight simulator source files now
include the standard algorithm header they use, allowing GCC 15 to compile
the pinned upstream source. The native patch and Python source fingerprints
change; use a build made for this release. Historical traces and research
builds keep their original identities.

## Verification on the M5 Pro

- Clean and reverse application of the versioned native patch, followed by a
  fresh core build and separate native search test fixture.
- All 281 environment tests passed with the native backend and search fixture
  required, including the step-by-step golden observations, rewards and menus.
- The bounded seed-3 random rollout replayed all 21 steps in a fresh process.
  Two independent rollouts matched across all 45 trace records apart from
  the terminal wall-clock field.
- An sdist and wheel were built, and the installed wheel located the verified
  native module from a fresh virtual environment. The Mac native module
  checksum matched the archived 0.1.5.1 core module.

The [workstream benchmark](https://github.com/SidBaines/sts-ai/blob/524766d/research/experiments/2026-11-18-mac-performance.md)
measures the environment change separately from the optional training changes.
