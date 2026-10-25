# Environment 0.1.6 verification record

Release 0.1.6 fixes two faults found by the LLM workstream on 25 October 2026:
- the out-of-combat potion line misreported potions;
- `run_episode` recorded agent errors as if the simulator had failed.

The out-of-combat observation text changes, so the public observation label
advances to `sts_env_public_v4`.
The release also restores the Linux build, which GCC 15 broke. Game logic, menus, rewards and the trace
schema are unchanged.

## Changes

- **Out-of-combat potion line.**
  - The fault: out-of-combat text looped over the number of potions held
    rather than the potion slots. Once an earlier slot had been emptied, the
    line showed `EMPTY_POTION_SLOT` and hid later potions that the menu still
    offered. For example, after discarding slot 0 of three potions, the line
    read `Potions: EMPTY_POTION_SLOT, Fire Potion` while the menu offered the
    Fruit Juice in slot 2. The glossary key also left out the hidden potions.
  - The fix: the line now lists every slot by index, in the same format as
    public combat text:
    `Potions (capacity 3): [0] empty, [1] Fire Potion, [2] Fruit Juice`. A run
    with no potions shows each slot as `empty`, not `none`.
  - Impact: in 48 out-of-combat LLM prompts sampled from 0.1.5.1, 7 disagreed
    with their own menus. In the step fixture's 457 out-of-combat states with
    the simple fixture policy, 3 were wrong.
- **Agent errors are distinguishable from environment faults.**
  - `run_episode` still records rather than raises errors, but each error
    record now carries `"source": "agent"` or `"source": "environment"`.
  - Agent errors are exceptions from loading the agent, its `reset`, its
    `choose_action`, or an invalid returned action. The native baseline's
    action query runs in the simulator, so it counts as environment.
  - `run_episode(raise_agent_errors=True)` writes the trace's end record and
    then re-raises the agent's original exception. Callers can use it to stop
    on, for example, a GPU out-of-memory error. Environment faults are never
    raised this way, and they are never losses.
  - `sts-env evaluate` reports `error_sources` and accepts
    `--stop-on-agent-error`.
- **Linux build with GCC 15.**
  - The fault: GitHub's `ubuntu-latest` runner moved to GCC 15.2, which
    rejects eight upstream simulator sources. They call `std::sort`,
    `std::stable_sort`, `std::find` or `std::find_if` without including
    `<algorithm>`. Linux CI had failed for this reason since 20 October,
    including for `env-v0.1.5.1`.
  - The fix: the patch adds `#include <algorithm>` to those files.
  - Impact: none on behaviour. The macOS binary is byte-identical with and
    without these lines.
- **Tooling.** `scripts/benchmark_env_step.py --record-fixture PATH` records
  the step-regression fixture. The fixture is now
  `tests/fixtures/env_step_reference.json`, recorded from this release.

## Migration notes

- Replay traces from 0.1.5 and 0.1.5.1 with their original build. Their
  out-of-combat observation hashes differ under 0.1.6.
- Re-baseline any experiment that reads out-of-combat text, including LLM
  prompts, before comparing it with 0.1.5.1. Structured state, menus and game
  logic are unchanged.
- Error records gained a `source` key. Readers that use only `kind` and
  `message` are unaffected, and the trace schema stays `sts_env_trace_v1`.

## Verification

Built on 25 October 2026 from the unchanged upstream pin
`7476a81954020087da31d41d16fddf475746ec2d`, with pybind11 v2.13.6, CMake 4.1.3,
Apple clang 21.0.0 and Python 3.12.13 on macOS arm64, in an isolated `.venv`.

- **Patch.** SHA256
  `71a779eab2991c2a1829a5d23c1f8d3dc320afd941d581bb3ec03816bdf3185c`. It applies
  to a fresh upstream checkout and reverse-applies to the build tree. Apart
  from shifted hunk line numbers, the patch diff is the potion-line change
  plus eight `#include <algorithm>` lines.
- **GCC 15.** Under GCC 15.3 (the `gcc:15` Docker image), all 29 simulator
  sources and both binding files pass a full syntax check.
- **Native binary.** SHA256
  `b279577747e408fe8d939814217f58aa2d06ef9aa5e0c2ffa59052b64f831155`. A second
  build from a fresh copy of the pinned source produced the identical binary.
  Adding the GCC 15 includes left it byte-identical, so the checks below
  apply to the final patch.
- **Only the potion line changed.** The 0.1.5.1 step fixture covers 8 worlds
  and 1,883 decisions. Replayed on the new build, with the new potion line
  converted back to the old count-based format, every observation, info,
  reward and termination hash matched. The fixture was then re-recorded for
  0.1.6.
- **Regression tests.**
  - On the unmodified 0.1.5.1 patch, built for comparison, the new potion test
    state reads `Potions: EMPTY_POTION_SLOT, Fire Potion`; this release reads
    all three slots.
  - New agent-error tests cover errors from a raising agent, an illegal
    action, a failing `reset`, a failing load and the native baseline, plus
    re-raising and `evaluate` stopping.
- **Full suite.** With the simulator and the test-only search fixture required
  (`STS_REQUIRE_SIMULATOR=1 STS_REQUIRE_SEARCH_TEST_FIXTURE=1`), all 286 tests
  pass with no skips.
- **CLI checks.**
  - `sts-env doctor` reports the new patch identity.
  - The bounded random smoke (seed 3, Act 1, 50 battle simulations) ends in an
    ordinary defeat after 21 decisions under `0.1.6` and
    `sts_env_public_v4`, and `sts-env replay` verifies it.
  - A custom agent raising a simulated out-of-memory error was recorded with
    `source: agent`, and `evaluate --stop-on-agent-error` stopped after that
    seed.
  - The wheel holds 27 files, with no native binary, tests or upstream source,
    and reports version 0.1.6.
- **Fresh-process determinism.** Each cohort below was run twice in separate
  processes. The traces were identical once wall-clock `elapsed_seconds`
  fields are excluded, with no errors:
  - `sts-env evaluate --agent native --mode ooc --max-act 3 --seeds 0:12`;
  - `--agent random --mode ooc --max-act 2 --seeds 0:16`. It executed 63
    potion actions, the same number as the identical cohort in the 0.1.5
    record, consistent with unchanged game logic;
  - `--agent native --mode full --max-act 1 --seeds 0:8`.

  Bounded full-mode and combat-mode random episodes were also identical
  across fresh processes and replayed step for step.

The immutable release reference will be `env-v0.1.6`, tagged after merge.
Earlier tags and binaries are unchanged.
