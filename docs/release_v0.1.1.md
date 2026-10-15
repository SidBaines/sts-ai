# Environment 0.1.1 verification record

Release 0.1.1 fixes an uninitialized combat character class. `Player.cc` now
has a safe default and `BattleContext::init` always copies `GameContext.cc`.
Native automatic combat value-initializes its local battle and throws when the
battle reports undefined behavior after initialization or playout, before
applying its outcome to the run. Faults therefore remain faults rather than
ordinary wins or losses.

The potion enum, normal-capacity potion acquisition methods, and read-only
battle character-class property are additive native setup/diagnostic APIs.
Policy observation schemas and reward definitions are unchanged. Seeded traces
which involve generated potions, or previously unreported automatic combat
faults, can intentionally change; old banks/binaries must retain their original
provenance and must not be resumed under this release.

Focused regression checks cover all four character classes with real Entropic
Brew use, repeatability after previous classes, Ironclad attack/skill/power potion
selection, and a real automatic-combat Discovery fault before `exitBattle`.
Upstream combat card-generation pools still always use Ironclad cards; this
release does not claim other-class card-generation fidelity or a general
repair of every native undefined-behavior path.

The isolated local binary is SHA256
`40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`,
with patch SHA256
`77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16`.
Upstream pin and pybind11 version are unchanged. Fresh apply and reverse apply
checks passed; the patch contains no pybind11 submodule-pointer change.

The full release suite, bounded CLI smoke, and affected-world replay validation
have passed. The immutable release reference is `env-v0.1.1`. The existing
`env-v0.1.0` tag and binary remain unchanged.

## Completed local base gates

On 12 October 2026, `STS_REQUIRE_SIMULATOR=1 PYTHONPATH=src
.venv/bin/python -m unittest discover -s tests -t .` passed all 195 tests in
2.9 seconds against the isolated binary. A bounded random OOC seed 3 smoke
(Act 1, 50 battle simulations, 1024 external-decision cap) ended in ordinary
defeat after 21 decisions with no simulator fault.

A frozen existing OOC warm policy on the same 32 development worlds (world
range starting at 10000, policy seed 777, horizon 1024, native 200 simulations,
CPU inference, four persistent workers) produced exactly equal per-world
semantic action hashes, step counts, outcomes, and fault traces in three fresh
runs at 10/10/5-second operation timeouts. Each run took about 2.5 seconds,
with 19 Act 1 clears among 31 valid worlds and one newly reported native fault
(world 10218, floor 16). This is a fault-exclusion check, not a new learner score.
The formerly unstable world 10121 cleared after 69 decisions every time.

For full control, native 200-simulation world 10121 produced 231 decisions and
465 JSONL records under both 0.1.0 and 0.1.1; complete traces were identical after
removing only release/binary/interface identity and elapsed-time fields. The
old binary checksum was rechecked and remained `8dba3d574ea3d4cd4c91b7031ffb38821b3e18c03358263407df3c05329c1724`.

The automatic-combat guard exposes faults previously hidden by the binding;
zero reported faults from old OOC artifacts therefore must not be interpreted
as native-state validity. The guard only checks the native live/sticky flag and
does not establish absence of every form of C++ undefined behavior.

## Experiment integration gate

The corrected experiment branch passed all 307 tests in 18.254 seconds against
this native build. Its generated fast-replay fixture uses newly collected 0.1.1
banks and panels; old 0.1.0 provenance remains correctly rejected. Fresh Act 1
collection yielded 49,944 labels, 2,245 battle starts, 252 valid worlds and four
faults. Fresh Act 3 collection yielded 560,620 labels, 24,276 battle starts,
2,025 valid worlds and 23 faults. These counts match the previous full-control
teacher collections while retaining the new binary and interface identities.

Both base and experiment verification gates passed before the release tag was
created. Experimental training, models and data remain outside this release.
