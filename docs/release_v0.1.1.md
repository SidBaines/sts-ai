# Environment 0.1.1 verification record (release candidate)

This candidate fixes an uninitialized combat character class. `Player.cc` now
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
provenance and must not be resumed under this candidate.

Focused regression checks cover all four character classes with real Entropic
Brew use, repeatability after previous classes, Ironclad attack/skill/power potion
selection, and a real automatic-combat Discovery fault before `exitBattle`.
Upstream combat card-generation pools still always use Ironclad cards; this
candidate does not claim other-class card-generation fidelity or a general
repair of every native undefined-behavior path.

The isolated local binary is SHA256
`40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`,
with patch SHA256
`77fddbd67c82bebcc4ff31eea6bb0f6d25e8216e44825d6459f184f47756ce16`.
Upstream pin and pybind11 version are unchanged. Fresh apply and reverse apply
checks passed; the patch contains no pybind11 submodule-pointer change.

The full release suite, bounded CLI smoke, and affected-world replay validation
are required before creating `env-v0.1.1`. The existing `env-v0.1.0` tag and binary
remain immutable. This record is deliberately pending those final gates.
