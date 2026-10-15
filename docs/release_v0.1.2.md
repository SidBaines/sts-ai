# Environment 0.1.2 verification record

Release 0.1.2 validates the supported Ironclad Discovery path instead of
rejecting every use as unsupported native behaviour. Unsupported class-card
generation and invalid Discovery offers still fail explicitly, and every
fault/timeout guard remains active. Public observation and trace schemas are
unchanged (`sts_env_public_v1`); native and interface identities change, so
seeded trajectories that reach Discovery intentionally continue past their
former rejection points. The repair, its semantics and its development
replays are documented in [the Discovery record](discovery_fix.md); it was
reviewed as draft PR #1 and carried to `main` in this release.

## Release-branch rebuild

On 15 October 2026 the release branch was rebuilt from a fresh clone of the
unchanged upstream pin `7476a81954020087da31d41d16fddf475746ec2d`, pybind11
v2.13.6, CMake 4.1.3, Apple clang 21.0.0 and Python 3.12.13 on macOS arm64,
using an isolated `.venv`.

- Patch SHA256 `21a564a810785d510c83faf5700e92559949810e323ab3282b3ba310fe632d5a`.
  Fresh and reverse application pass; the patch has no pybind11 submodule hunk
  and no one-space blank context lines.
- Native binary SHA256
  `645acbf2cfdf8055f59f379bdb0d2d2e3a06589809f6252f4bef3c50c3a0c198`,
  identical to the binary recorded in the Discovery verification, so the build
  is reproducible on this toolchain.
- `STS_REQUIRE_SIMULATOR=1 PYTHONPATH=src .venv/bin/python -m unittest discover
  -s tests -t .` passes all 202 tests with no skips, including the fresh-process
  Discovery determinism check.
- The bounded random smoke (seed 3, Act 1, 50 battle simulations) ends in an
  ordinary defeat after 21 decisions.

The immutable release reference is `env-v0.1.2`. Earlier tags and binaries are
unchanged.
