# Slay the Spire Environment

A maintained environment base for custom agents, LLM adapters, and reinforcement
learning experiments. The Python core has no runtime dependencies. Game logic
comes from a pinned, patched [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed)
C++ simulator, built separately for your Python interpreter.

This release supports Ironclad, with Acts 1–3 as the standard experiment scope.
Act 4 is optional and experimental: its key ownership and burning-elite flags
are not yet a complete public observation surface. Experiments, model clients, training algorithms,
and experiment-specific reward shaping belong in adapters or branches created
from a release tag. The `environment-base` branch receives environment fixes;
`env-v0.1.0` identifies the first tested release.

## Install and build

Use Python 3.10–3.13, Git, CMake, and a C++ compiler. On macOS install the Xcode
command-line tools; on Linux install your distribution's C++ build tools. From a
source checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
scripts/build_lightspeed.sh
.venv/bin/sts-env doctor
```

The build script pins upstream and pybind11, applies the versioned binding and
bug-fix patch, and builds the native module in `external/sts_lightspeed/build`.
Use the same interpreter to build and run. The build script may install its pinned
CMake package into the interpreter selected by `PYTHON`. Use this checkout's own
isolated `.venv`; any `PYTHON` override should point there. Do not borrow an
active or frozen experiment runtime for a build. The Python package does not bundle a
native binary. If you install a wheel or keep the simulator elsewhere, set
`STS_LIGHTSPEED_BUILD` to its build directory.

## Choose who controls each decision

| Mode | Your agent controls | Automatic policy controls |
| --- | --- | --- |
| `full` | Combat and out-of-combat decisions | Internal transitions only |
| `ooc` | Out-of-combat decisions | Combat through native search |
| `combat` | Combat decisions | Out-of-combat through the native policy |

The flipped hybrid is supported by `combat`. The native out-of-combat policy
uses scripted choices and seeded random map pathing; it does not perform an
out-of-combat lookahead search. This distinction matters when comparing results.

## CLI

Run a bounded baseline and save its replayable trace:

```bash
.venv/bin/sts-env run --agent random --seed 3 --mode ooc \
  --max-act 1 --battle-simulations 50 --output data/smoke.jsonl
.venv/bin/sts-env inspect data/smoke.jsonl
.venv/bin/sts-env replay data/smoke.jsonl
```

Evaluate a seed set with one trace per seed and a combined `summary.json`:

```bash
.venv/bin/sts-env evaluate --agent first --seeds 3,4,5 --mode combat \
  --max-act 1 --output data/baseline
```

`--seeds 3:6` means seeds 3, 4, and 5. Alternatively use `--seed-file` with a JSON
list or one integer per line. `--policy-seed` controls the agent RNG independently
of the world seed. `--max-steps` limits exposed decisions, `--max-act` sets the
act boundary, and `--step-timeout` bounds native reset/search/step operations.
Your own model/API agent should implement deadlines for its calls. `--ascension`
sets difficulty (0–20); `--ooc-seed` separately controls the automatic OOC policy. Existing output
files are rejected, so evaluations cannot silently overwrite earlier results.

Play manually with `sts-env play --seed 3 --mode full`, or inspect all available
flags using `sts-env <command> --help`. Built-in agents are `first`, `random`, and
`native`. `doctor` reports installation and simulator identity without modifying
the checkout. `replay` re-runs stored actions and checks the recorded public
observations, rewards, and episode boundaries.

## Plug in an agent

Put `my_agent.py` in an importable directory (for example the working directory):

```python
class MyAgent:
    def __init__(self, preference=0):
        self.preference = preference

    def reset(self, seed):
        # Optional: reset your own model state or RNG for this episode.
        pass

    def choose_action(self, observation):
        # Return the index of an action in this observation's legal menu.
        return self.preference


def make_agent(preference=0):
    return MyAgent(preference=preference)
```

```bash
.venv/bin/sts-env run --agent my_agent:make_agent \
  --agent-config '{"preference": 0}' --seed 3 --mode full \
  --max-act 1 --output data/custom.jsonl
```

Factories receive keyword arguments from `--agent-config`. Agents implement
`choose_action(observation) -> int` and may implement `reset(seed)`. Evaluation
reuses the loaded agent across seeds when it provides `reset`: that method must
clear per-episode state and seed your RNG while retaining expensive model
weights. Without `reset`, evaluation creates a fresh agent for each episode.
Action
indices refer to the current legal-action menu; they have no stable meaning
across different observations. Invalid indices fail explicitly. Wrap your own
LLM client or learned policy behind this interface; no particular model SDK is
required by the environment.

## Python API

```python
from sts_env import SlayTheSpireEnv

def main():
    with SlayTheSpireEnv(mode="full", max_act=1) as env:
        observation, info = env.reset(seed=3)
        terminated, truncated = info["terminated"], info["truncated"]
        while not (terminated or truncated):
            action = 0  # Choose an index from observation["legal_actions"].
            observation, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    main()
```

Observations contain `state` (structured public state), `text` (public text),
`phase`, and `legal_actions` (public menu index and description).
`reset` returns `(observation, info)`; `step` returns
`(observation, reward, terminated, truncated, info)`. The default reward is 1
on a game victory and 0 otherwise. Defeat terminates the game; the configured
act/decision limits truncate it. Supply `reward_fn` in your own adapter for
other rewards, and record that choice with experiment results.

Use the observation as policy input. Diagnostic `info` can contain privileged
simulator details such as raw executed-action bits, which can encode hidden draw
positions; do not feed that diagnostic metadata to your model. The Gymnasium
wrapper's `info["observation"]` contains the safe public observation.

For Gymnasium integration, install `.venv/bin/python -m pip install -e '.[gym]'`
and import `GymnasiumEnv` from `sts_env.gymnasium`. Its fixed `Discrete(2048)` action space uses
an `action_mask` to mark the current valid menu indices (configure `max_actions`
for larger menus). Observations are dictionaries with `features` (15 numeric
values) and `action_mask`; the full public observation is retained
in `info["observation"]`. `GymnasiumEnv.FEATURE_NAMES` gives the feature order.

The compact encoder summarizes HP, gold, progress, and basic combat counts. It
omits card identities, enemy powers/intents, event offers, and map structure; it
is a baseline input rather than a complete game-state encoding. Supply a richer
encoder in your experiment wrapper for serious learning. Algorithms should respect
the mask, and action meanings vary with the current menu. Selecting a padded,
masked slot inside the fixed Gymnasium action space produces an explicit no-op
with reward zero (`info["invalid_action"] = True`); `invalid_action_reward` can
be configured. The Gymnasium decision limit counts all attempted steps so these
no-ops cannot create an unbounded episode. The dependency-free core rejects
invalid action indices. Raw native action bits are excluded from policy
observations because they can reveal hidden simulator positions. Python scripts
using worker isolation need the
`if __name__ == "__main__"` guard shown above.

## Reproducibility and faults

Record the release tag, world and policy seeds, mode, search settings, simulator
identity, and agent/model identity for every experiment. The environment records
a hash of its Python interface files plus the exact runtime Python version;
replay rejects changed environment source even when its version label stays the
same. Record your own agent/model code and checkpoint identity separately. CLI traces retain the
configuration and public decision sequence for replay. Reproducibility is
validated within the same native build; identical traces across different
platforms or compiler builds are not promised.

Native calls run in an isolated worker with a timeout. A stalled or crashed
simulator raises `SimulatorTimeout` or `SimulatorError`; it must not be counted as
a normal loss or silently retried under a different seed. The simulator patch
contains defensive fixes, but timeout isolation remains necessary for upstream
native faults. Known upstream undefined-memory and phantom-power failures are
not certified as eliminated in every reachable state. Timeouts, native errors,
and reported undefined-behaviour flags are contained; unflagged state corruption
can still escape those checks. The [environment contract and release checks](docs/environment_base.md)
explain the interface boundaries and verification procedure.
