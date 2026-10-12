# Train and use a non-LLM agent

Use the `codex/nonllm-curriculum` experiment checkout for `sts_policy`, its
training scripts and PyTorch plugin. Its environment core is the released
`env-v0.1.1`. The clean `environment-base` release provides the public API and
CLI; it does not include this experiment's models or training adapters.
Ironclad and Acts 1–3 are the supported experiment scope.

## Setup

Run commands from the experiment repository root, using the same Python
interpreter used to build the simulator:

```sh
.venv/bin/python -m pip install -e '.[policy]'
export PYTHONPATH=src
export STS_LIGHTSPEED_BUILD="$PWD/external/sts_lightspeed_v011/build"
.venv/bin/python -m sts_env doctor
```

The `policy` extra installs the pinned optional Torch and NumPy dependencies;
the environment core remains dependency-free. The build above already exists
in this worktree and is the corrected 0.1.1 binary, SHA256
`40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`.
The old `8dba…` binary and its auto-combat results are a different condition.
Native binaries must match the Python ABI and versioned patch. On a fresh
checkout, run `scripts/build_lightspeed.sh` and point `STS_LIGHTSPEED_BUILD` at
its actual build directory. To preserve an older patched source/build, choose
new directories through `STS_LIGHTSPEED_SOURCE` and `STS_LIGHTSPEED_BUILD`.
Do not point a new build at another source tree's existing CMake cache.

## Start a separate training experiment

These bounded examples create **new** checkpoints under `data/custom_nonllm`;
they do not resume or change the campaign. The teacher bank and task panel
referenced here are existing corrected-backend artifacts in this worktree.

```sh
.venv/bin/python scripts/train_nonllm.py bc \
  --teacher data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl \
  --split train --phase all --size large --context none \
  --device mps --batch-size 1024 --learning-rate 0.0001 \
  --updates 100 --seconds 120 --output data/custom_nonllm/bc.pt

.venv/bin/python scripts/train_nonllm.py ppo \
  --checkpoint data/custom_nonllm/bc.pt \
  --tasks data/nonllm_curriculum/panels_native011/train_act1.jsonl \
  --split train --task-kind battle --max-act 1 --horizon 256 \
  --device mps --inference-device cpu --workers 8 --threads 2 \
  --batch-size 1024 --learning-rate 0.0001 --clip-ratio 0.1 --gamma 1 \
  --kl-guard-before-update --fast-observations --fast-replay --deferred-bootstrap \
  --simulations 200 --reward-mode potential --terminal-hp-weight 0 \
  --updates 100 --seconds 180 --output data/custom_nonllm/combat.pt
```

Use `--device cpu` when MPS is unavailable. The measured Mac execution route is
CPU inference with eight persistent native workers and MPS optimizer updates.
The current joint stage uses batch 1024, LR 1e-4, clip 0.1, gamma 1 and the
before-update KL guard. Its full-run reward is `progress` with floor weight
0.05; the combat example above uses `potential`. These are recorded experiment
settings, not environment defaults or a universal optimum.

`--task-kind ooc` trains only OOC choices with native combat;
`--task-kind run` exposes both phases. Training is restricted to `train` worlds.
Use separate `dev` evaluations for selection and preserve the final-world
embargo. Teacher labels may use privileged search; policy inputs are public
observations only. Faults invalidate whole episodes and are reported separately.
See [training methods](nonllm_training_methods.md) for BC/DAgger/PPO and reward
semantics. Run `scripts/train_nonllm.py --help` for all options.

## Load the frozen composite through the public CLI

During training, `campaign.json` selection is mutable. After final selection,
`data/nonllm_native_revision/campaign/frozen_identities.json` records immutable
checkpoint paths and hashes. Its `checkpoints.shared.path` selects the final
whole-run checkpoint, including a composite when that is the selected model.
That freeze file is created at the final gate; do not guess a checkpoint name
or treat a current joint-stage filename as the final release artifact.

After that file exists, derive the real checkpoint path from it:

```sh
AGENT_CONFIG="$(.venv/bin/python -c 'import json; from pathlib import Path; f=json.loads(Path("data/nonllm_native_revision/campaign/frozen_identities.json").read_text()); print(json.dumps({"checkpoint": f["checkpoints"]["shared"]["path"], "device": "cpu", "deterministic": True}))')"
.venv/bin/python -m sts_env run \
  --agent sts_policy.model_agent:make_agent --agent-config "$AGENT_CONFIG" \
  --mode full --seed 10121 --policy-seed 777 --max-act 1 --max-steps 2048 \
  --battle-simulations 200 --step-timeout 10 \
  --output data/custom_nonllm/frozen_dev10121.jsonl
.venv/bin/python -m sts_env inspect data/custom_nonllm/frozen_dev10121.jsonl
.venv/bin/python -m sts_env replay data/custom_nonllm/frozen_dev10121.jsonl
```

The factory accepts a composite `.pt` through `checkpoint` and dispatches its
heads using the public decision phase. It also accepts separate
`combat_checkpoint` and `ooc_checkpoint` keys instead of `checkpoint`; see
[composite checkpoints](nonllm_composite_checkpoints.md). The `sts-env` console
command is equivalent to `.venv/bin/python -m sts_env`. Existing trace files
are rejected; choose a new output path for another run.

| Public CLI mode | Model controls | Automatic controller |
|---|---|---|
| `full` | Combat and OOC | Internal transitions only |
| `ooc` | OOC | Native combat search |
| `combat` | Combat | Scripted native OOC choices with seeded pathing |

The flipped hybrid is `combat`; its OOC controller is not a lookahead search.
For composite performance, use `full`. The public environment reports act
limits as truncations and gives reward 1 only for game victory. Curriculum
Act-clear rewards and shaping belong to the training adapter; they do not
change public CLI reward or termination semantics.

## Verify before sharing

The helper runs the actual public CLI twice in fresh processes per seed, then
replays each first trace. It checks the frozen backend/checkpoints, public legal
actions, both decision phases, zero automatic native decisions, and identical
complete traces except elapsed time. Seeds 10121 and 10011 are already verified
development worlds; the helper rejects seed 3, unknown worlds and final worlds.

```sh
.venv/bin/python scripts/verify_nonllm_cli.py \
  --freeze data/nonllm_native_revision/campaign/frozen_identities.json \
  --checkpoint-key shared \
  --dev-evidence data/nonllm_native_revision/campaign/joint_start_composite_dev.metrics.json \
  --seeds 10121,10011 --output-directory data/nonllm_native_revision/cli_verification \
  --plan-only
```

Review the printed commands and identities, then remove `--plan-only` to execute.
Use a new verification directory if a previous one exists. This helper expects
a frozen composite checkpoint and does not run final-test worlds.

Checkpoints, teacher banks, metrics and native build products are gitignored
local artifacts; committing code does not preserve them. Before moving or
archiving this worktree, copy the needed `.pt` files, evidence, freeze manifests
and dataset provenance to a durable location and verify their hashes. Keep the
adapter source/version and native build identity with them. Freeze records use
absolute paths: relocation needs a separately recorded path identity while
preserving the original freeze record. A native build can be recreated from its
pinned release patch for the destination interpreter and platform.
