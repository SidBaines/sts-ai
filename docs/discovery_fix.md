# Discovery fault repair (0.1.2 candidate)

The native simulator's `undefinedBehaviorEvoked` flag is a broad upstream
unsupported-behaviour marker, not a C++ memory-safety diagnostic. In the pinned
source, playing **any Discovery card** sets it unconditionally. Its existing
Ironclad generation and selection path is implemented; the two confirmed Act 1
boss faults are therefore reproducible false rejections of that supported path.
This change does not establish a general repair of upstream memory corruption.

## Reproduction and scope

Development worlds 401491 and 400594, using the frozen 0.1.1 native binary,
fail after respectively 168 and 201 successful decisions at floor 16. In both,
the failing legal index is zero: play Discovery. World 401491 obtains it from
Colorless Potion; world 400594 obtains it from Jack of All Trades after a
Colorless Potion. These are development fixtures, not held-out evaluation worlds.

`tests/fixtures/discovery_boss_replays.json` stores action indices and their
public descriptions through the failing action. Replaying them requires no
model, Torch, or checkpoint. The successful prefixes match the original
semantic action hashes exactly:

| World | Original successful decisions | Prefix SHA256 |
| --- | --- | --- |
| 401491 | 168 | `87d3f0c2b66eae5b3f87a061b86f4b4e1678cecd7db21276302d3f3b803b6eb2` |
| 400594 | 201 | `d2bd3ddd9154aac8859b1156a24665aa1b8a235482f009536a80ae8f82d18d07` |

The queued Discovery action now explicitly validates Ironclad support and its
three distinct, non-invalid class-card offers before opening the menu. Unsupported
classes and invalid offers set the fault flag and raise a native error. The
Python sticky/live flag checks, automatic-combat guards, process isolation and
timeouts remain active. Failures remain errors and are never scored as losses.
The generator's pool order and RNG calls are unchanged.

The pinned simulator's [card definitions](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/include/constants/Cards.h)
make base Discovery exhaust and upgraded Discovery retain its ordinary discard
behaviour. Tests check both, all three selected offers, one temporary card at zero
cost for the current turn, base-cost restoration on the next turn, and duplicated
Discovery when the hand is full. They also check every unsupported character
class retains explicit failure. Card-generation pools for other classes remain
incomplete; no new claim of their fidelity is made.

## Verification

The isolated macOS build uses Python 3.12.13, Apple clang 21.0.0, CMake 4.1.3,
the unchanged upstream pin and pybind11 2.13.6. It was built with two workers at
nice level 15. Native binary SHA256:
`645acbf2cfdf8055f59f379bdb0d2d2e3a06589809f6252f4bef3c50c3a0c198`.
Patch SHA256:
`21a564a810785d510c83faf5700e92559949810e323ab3282b3ba310fe632d5a`.
Fresh and reverse patch application pass; the patch has no dependency submodule
pointer hunk or one-space blank context lines.

The required full suite passes **201 tests** with `STS_REQUIRE_SIMULATOR=1`.
The bounded random seed 3, Act 1, 50-simulation OOC smoke ends in ordinary defeat
after 21 decisions; replay verifies all 21 steps. Both confirmed prefixes now
open a valid Discovery menu. A deterministic first-action continuation clears
Act 1 after 216 and 231 total decisions. Two fresh processes produce identical
complete public-state/reward/boundary hashes:

| World | Continuation trace SHA256 |
| --- | --- |
| 401491 | `3060979da527253cabb84b100a3e03663cc9ee99ec0f2f39c9acec1a641e1f0f` |
| 400594 | `1538c45ef60b6bf552a8015e6d7b36966cc26c9e94a12bed48a8ec4fc47abe15` |

These continuations validate execution and determinism, not the score of the
original learned policy. The permanent integration test repeats the fresh-process
check with finite operation and subprocess deadlines.

## Version and experiment boundary

Package/environment version is 0.1.2; public observation and trace schemas remain
unchanged. The patch and interface identities change, and formerly rejected
Discovery trajectories now continue. This is an intentional seeded-trajectory
boundary. Rebuild into a separate directory for future experiments and record
its provenance; old traces, banks, checkpoints and frozen binaries retain their
original versions. The running long-curriculum-v2 study must continue using its
frozen 0.1.1 backend. This candidate has no release tag yet; no existing release
tag is moved and no study backend is hot-swapped.
