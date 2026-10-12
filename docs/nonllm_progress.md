# Non-LLM curriculum experiment progress log

Updated 2026-10-12 04:32:54 UTC. Campaign status: **running**; phase: **joint_act3**.

Allocation: 2026-10-11 20:08:46 UTC → 2026-10-12 06:30:00 UTC. Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.

This page is generated from the atomic campaign journal and raw metrics. It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. Development results are not final-test results. Simulator faults are separate from ordinary game defeats. [Short latest-first summary](nonllm_summary.md) records the major decisions and supporting evidence.

## Current selection

```json
{
  "combat": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt",
  "joint_act1": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt",
  "joint_act3": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.pt",
  "joint_pilot": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_initial_composite_5a0cc4ad74cb6b2c.pt",
  "ooc": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt",
  "ooc_pilot": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.pt",
  "shared_bc": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt"
}
```

## Experiment timeline

### 2026-10-12 01:12:48 UTC — ooc_specialist_bc

Status: **complete**. Finished 2026-10-12 01:14:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.metrics.json --seconds 120 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl --updates 100000 --phase ooc --fast-replay
```

Training: 2,973 updates, 2,831,172 examples, 115.56s; 24,499 examples/s; mean loss 0.2247.

Development imitation: {"accuracy": 0.8119601328903654, "examples": 1505, "loss": 2.397033955013237, "menu_accuracy": {"choice": 0.7289272030651341, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.8119601328903654}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.7289272030651341, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 122.85s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.log`.

### 2026-10-12 01:14:54 UTC — ooc_warmstart_reference_dev

Status: **complete**. Finished 2026-10-12 01:14:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_warmstart_reference_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_warmstart_reference_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 20, 'act_clear': 12}; 2,018 valid transitions.

Mean terminal/cutoff floor 15.97; mean fight HP cost 48.72.

Total job wall time 2.25s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_warmstart_reference_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_warmstart_reference_dev.log`.

### 2026-10-12 01:14:59 UTC — ooc_lr0.0001_progress0.05

Status: **complete**. Finished 2026-10-12 01:16:34 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 20 rounds, 1262 non-fault episodes, 18 surfaced faults, 0 truncations; outcomes {'defeat': 541, 'act_clear': 721}; 78,359 valid transitions.

Mean terminal/cutoff floor 15.97; mean fight HP cost 33.03.

PPO: 240 optimizer steps; last update {"approx_kl": 0.0014803492813371122, "clip_fraction": 0.0087890625, "early_stop": true, "elapsed_seconds": 1.0015198329929262, "entropy": 0.10295930504798889, "policy_loss": -0.005998647461334865, "preparation_seconds": 0.6977342078462243, "ratio_mean": 0.9997431635856628, "samples": 3873, "steps": 3, "value_loss": 0.09579917788505554}.

Fault evidence: [{"action_trace_sha256": "c2e0f42516f17aadb5abd8fc3f3a9fac78e9ae9e3c93d19cb84d395468dc16c2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 17, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22059 floor=5 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "5e023f3329cae70b05f706f15b917c2c6ebabf51c82966b8fea49a5f8518705f", "task_kind": "ooc", "wait_seconds": 0.004890041891485453, "world_seed": 22059}, {"action_trace_sha256": "35a983f5ac1b00b9c49b114438bdd120d5c2bf9f0f48efb320eb8b3d1ce1153b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22069 floor=7 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "6c2750e649a2a910c0ece1032750740ec0f103426082ea5b79d64dae23d26f90", "task_kind": "ooc", "wait_seconds": 0.002913333009928465, "world_seed": 22069}, {"action_trace_sha256": "0e4eb038a9e6783a200751ff225567d4aa5c0e545fa3a34282a0d2d82f5eaceb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 26, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22069 floor=7 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "6c2750e649a2a910c0ece1032750740ec0f103426082ea5b79d64dae23d26f90", "task_kind": "ooc", "wait_seconds": 0.005891249980777502, "world_seed": 22069}, {"action_trace_sha256": "0cdcd15fa761410555e31e8c59cee93cb8e69ad2b1771c8426831b40614ce752", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=11 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.004276582971215248, "world_seed": 22058}, {"action_trace_sha256": "cf35ac451b70b552315da37e6372fd726e7474d4df1c09b5bb3f5390601bf324", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22063 floor=13 encounter=8 error=native battle playout evoked undefined behavior", "task_id": "7b24cff62c4d22c1b73d270dfbafe8ccb959afe6dccfaf1d28d0c9e35984ccb9", "task_kind": "ooc", "wait_seconds": 0.005802250001579523, "world_seed": 22063}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.11s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05.log`.

### 2026-10-12 01:16:34 UTC — ooc_lr0.0001_progress0.05_dev

Status: **complete**. Finished 2026-10-12 01:16:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 13, 'act_clear': 19}; 2,069 valid transitions.

Mean terminal/cutoff floor 16.59; mean fight HP cost 31.12.

Total job wall time 2.27s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.05_dev.log`.

### 2026-10-12 01:16:39 UTC — ooc_lr0.0003_progress0.05

Status: **complete**. Finished 2026-10-12 01:18:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 25 rounds, 1578 non-fault episodes, 5 surfaced faults, 8 truncations; outcomes {'defeat': 525, 'act_clear': 1045, 'GameOutcome.UNDECIDED': 8}; 96,577 valid transitions.

Mean terminal/cutoff floor 16.05; mean fight HP cost 25.05.

PPO: 108 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5559154581278563, "preparation_seconds": 0.4539873329922557, "samples": 2560, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "8ac8f3b61347171df08b7f38bc6e2249e9bb84233a215fb00f94ce6b150eecfc", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 3, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22047 floor=1 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "40d5a1a2977eb8c174c077ea58f818c296597134e63984028871802773ad7074", "task_kind": "ooc", "wait_seconds": 0.00526533299125731, "world_seed": 22047}, {"action_trace_sha256": "8ac8f3b61347171df08b7f38bc6e2249e9bb84233a215fb00f94ce6b150eecfc", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 3, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22047 floor=1 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "40d5a1a2977eb8c174c077ea58f818c296597134e63984028871802773ad7074", "task_kind": "ooc", "wait_seconds": 0.0033280421048402786, "world_seed": 22047}, {"action_trace_sha256": "0c524d577a179de66808c56379f8a8638ddc9fb700a6e880b4c79957fd037542", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 51, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22026 floor=11 encounter=7 error=native battle playout evoked undefined behavior", "task_id": "77871c8c0f05b93328a010b93caa09bafa9d14a9668e46e8a4e0183e6660a0b8", "task_kind": "ooc", "wait_seconds": 0.0008465831633657217, "world_seed": 22026}, {"action_trace_sha256": "6f22733d27d3b48c450020c4f8876068beba1b58b8a574a4903555fcb5ceb21b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 16, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22059 floor=5 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "5e023f3329cae70b05f706f15b917c2c6ebabf51c82966b8fea49a5f8518705f", "task_kind": "ooc", "wait_seconds": 0.0041963751427829266, "world_seed": 22059}, {"action_trace_sha256": "8be5de84a89a9d86ccf0513badecff600ca43202059a2cb4ff18ddc7882729af", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 60, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22011 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "24736c0781c945a5cc516e4a22a0fc5d0d141a17d737a9e364413ece127e20e1", "task_kind": "ooc", "wait_seconds": 0.014405791880562901, "world_seed": 22011}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.68s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.log`.

### 2026-10-12 01:18:14 UTC — ooc_lr0.0003_progress0.05_dev

Status: **complete**. Finished 2026-10-12 01:18:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 8, 'act_clear': 23}; 1,881 valid transitions.

Mean terminal/cutoff floor 15.97; mean fight HP cost 20.29.

Fault evidence: [{"action_trace_sha256": "85debad0993a5f0b27dc4182d1fee041787f8bbcfd9d06ca053d7a2c5e30deb8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10152 floor=7 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "4b366888d6184231e49210e82bf33eb0fd9b974b29d5a46c1bd5858171948e1a", "task_kind": "ooc", "wait_seconds": 0.004327415954321623, "world_seed": 10152}].

Total job wall time 2.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05_dev.log`.

### 2026-10-12 01:18:19 UTC — ooc_lr0.001_progress0.05

Status: **complete**. Finished 2026-10-12 01:19:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 26 rounds, 1615 non-fault episodes, 11 surfaced faults, 8 truncations; outcomes {'defeat': 767, 'act_clear': 840, 'GameOutcome.UNDECIDED': 8}; 95,556 valid transitions.

Mean terminal/cutoff floor 15.54; mean fight HP cost 36.98.

PPO: 50 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.3075572911184281, "preparation_seconds": 0.20385674992576241, "samples": 1074, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "5c190240b86234a1e5a3e306fa911b0ce39e04e5de7677ade73b9ee1d52cf1cb", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=6 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.006711834110319614, "world_seed": 22058}, {"action_trace_sha256": "f0b1e2fdae75ea76640a41cf76d6ff7c2ab39b971a7316c47cdb4931a2df3fcb", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=6 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.006749332882463932, "world_seed": 22058}, {"action_trace_sha256": "af6dc76cded852f22432afa1ac8ab0096181a70c850bbcdead04a0ff90272fa7", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=6 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.006099333055317402, "world_seed": 22058}, {"action_trace_sha256": "498d5ee1e0a5d33ec95b7e6b53af142cc74d67f7cec3ddafd1d1071d3210815a", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=6 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.006049915915355086, "world_seed": 22058}, {"action_trace_sha256": "d68fa09c482f287a560204ef7b0471a72af905f40f39d93bc3bd8096d615aee8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22069 floor=7 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "6c2750e649a2a910c0ece1032750740ec0f103426082ea5b79d64dae23d26f90", "task_kind": "ooc", "wait_seconds": 0.005460083018988371, "world_seed": 22069}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.67s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05.log`.

### 2026-10-12 01:19:54 UTC — ooc_lr0.001_progress0.05_dev

Status: **complete**. Finished 2026-10-12 01:19:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 24, 'act_clear': 8}; 1,815 valid transitions.

Mean terminal/cutoff floor 14.94; mean fight HP cost 60.47.

Total job wall time 2.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.001_progress0.05_dev.log`.

### 2026-10-12 01:19:59 UTC — ooc_lr0.0001_sparse

Status: **complete**. Finished 2026-10-12 01:21:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode sparse --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 21 rounds, 1330 non-fault episodes, 14 surfaced faults, 0 truncations; outcomes {'defeat': 582, 'act_clear': 748}; 81,817 valid transitions.

Mean terminal/cutoff floor 15.87; mean fight HP cost 33.54.

PPO: 248 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.7834962089546025, "preparation_seconds": 0.6775955418124795, "samples": 3836, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "5fc1684e0472dc9d565190c9b5455a9f99a32921ab4d2b1598c450da9f08ce89", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=11 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.005185625050216913, "world_seed": 22058}, {"action_trace_sha256": "36d227bd9380d1bdadcb3eef04300924c4a8b5a14ce4786e1441ae4afca67343", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 43, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22052 floor=12 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "2b7870ec7c87c922884506d89c29c3376838e3df37a84874d254cda3b841b783", "task_kind": "ooc", "wait_seconds": 0.004935875069350004, "world_seed": 22052}, {"action_trace_sha256": "c2e0f42516f17aadb5abd8fc3f3a9fac78e9ae9e3c93d19cb84d395468dc16c2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 17, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22059 floor=5 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "5e023f3329cae70b05f706f15b917c2c6ebabf51c82966b8fea49a5f8518705f", "task_kind": "ooc", "wait_seconds": 0.005386499920859933, "world_seed": 22059}, {"action_trace_sha256": "99353d6b7a4b579a8ede67134613cfab49f1a5d1c301579101b1fbec34ecb145", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 11, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22024 floor=3 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "13f1f20ead1ecd7b25c0ad5ab43358f11957d7c7f50626d474921fa29d1c7320", "task_kind": "ooc", "wait_seconds": 0.0046153750736266375, "world_seed": 22024}, {"action_trace_sha256": "99353d6b7a4b579a8ede67134613cfab49f1a5d1c301579101b1fbec34ecb145", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 11, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22024 floor=3 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "13f1f20ead1ecd7b25c0ad5ab43358f11957d7c7f50626d474921fa29d1c7320", "task_kind": "ooc", "wait_seconds": 0.004656041972339153, "world_seed": 22024}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse.log`.

### 2026-10-12 01:21:35 UTC — ooc_lr0.0001_sparse_dev

Status: **complete**. Finished 2026-10-12 01:21:40 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 14, 'act_clear': 17}; 1,955 valid transitions.

Mean terminal/cutoff floor 16.23; mean fight HP cost 34.77.

Fault evidence: [{"action_trace_sha256": "a000a4a61978e3090269a182c65056af2f5b77e823f1b65495516be77564721d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 12, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10070 floor=3 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "32e836d57878321fa33742873738a4dedcf1a627fb08986a33811963dc5ecc67", "task_kind": "ooc", "wait_seconds": 0.00678083417005837, "world_seed": 10070}].

Total job wall time 2.61s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_sparse_dev.log`.

### 2026-10-12 01:21:40 UTC — ooc_lr0.0001_progress0.02

Status: **complete**. Finished 2026-10-12 01:23:15 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_specialist_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.02 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 22 rounds, 1380 non-fault episodes, 8 surfaced faults, 8 truncations; outcomes {'defeat': 571, 'act_clear': 801, 'GameOutcome.UNDECIDED': 8}; 85,405 valid transitions.

Mean terminal/cutoff floor 15.94; mean fight HP cost 31.70.

PPO: 237 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.449068833142519, "preparation_seconds": 0.4373358751181513, "samples": 2480, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "0e4eb038a9e6783a200751ff225567d4aa5c0e545fa3a34282a0d2d82f5eaceb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 26, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22069 floor=7 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "6c2750e649a2a910c0ece1032750740ec0f103426082ea5b79d64dae23d26f90", "task_kind": "ooc", "wait_seconds": 0.005461832974106073, "world_seed": 22069}, {"action_trace_sha256": "dd47cb940c0f4af2423d92097977e00df22801257389bb179c0e91ef2a07f316", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22052 floor=12 encounter=17 error=native battle playout evoked undefined behavior", "task_id": "2b7870ec7c87c922884506d89c29c3376838e3df37a84874d254cda3b841b783", "task_kind": "ooc", "wait_seconds": 0.0038801247719675303, "world_seed": 22052}, {"action_trace_sha256": "45fd9e4d114bd521b74ac7a2d0d3becbfe52929a87163b408e548fd06bba2d75", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22052 floor=12 encounter=8 error=native battle playout evoked undefined behavior", "task_id": "2b7870ec7c87c922884506d89c29c3376838e3df37a84874d254cda3b841b783", "task_kind": "ooc", "wait_seconds": 0.0060102499555796385, "world_seed": 22052}, {"action_trace_sha256": "f21b545f7251071cf9ea3587f26e06975ac7022e9a4240105be3e48c665d7bd2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 15, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22027 floor=4 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "6b49b59fcfe19e3b9626a7c7fe4279d2de538a54a42b8dbb820b2300f0aa4507", "task_kind": "ooc", "wait_seconds": 0.003532417118549347, "world_seed": 22027}, {"action_trace_sha256": "9d704a4b21035a910d28cc6c0f2cc1ab0155b5bfab828dd5c6076b9e15de0d58", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 53, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=22058 floor=14 encounter=10 error=native battle playout evoked undefined behavior", "task_id": "67f7b7c2d6c90d5e3a0c891e5a9730a49f58d38fe7c40cf2da10348805ec0c42", "task_kind": "ooc", "wait_seconds": 0.004044499946758151, "world_seed": 22058}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02.log`.

### 2026-10-12 01:23:15 UTC — ooc_lr0.0001_progress0.02_dev

Status: **complete**. Finished 2026-10-12 01:23:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 14, 'act_clear': 17}; 1,954 valid transitions.

Mean terminal/cutoff floor 16.16; mean fight HP cost 34.52.

Fault evidence: [{"action_trace_sha256": "332c9b3f39a71f2577fcd813c038e8cd48dde9e9e0d37abc6bc76899c7351415", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 17, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10049 floor=5 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "69b616abaec698e16cb89d12105b1174a04e0f4bc502c19c78347e4a701c5d84", "task_kind": "ooc", "wait_seconds": 0.007468750001862645, "world_seed": 10049}].

Total job wall time 2.56s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0001_progress0.02_dev.log`.

### 2026-10-12 01:23:20 UTC — ooc_initial_dev

Status: **complete**. Finished 2026-10-12 01:23:25 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_initial_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_initial_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 8, 'act_clear': 23}; 1,881 valid transitions.

Mean terminal/cutoff floor 15.97; mean fight HP cost 20.29.

Fault evidence: [{"action_trace_sha256": "85debad0993a5f0b27dc4182d1fee041787f8bbcfd9d06ca053d7a2c5e30deb8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10152 floor=7 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "4b366888d6184231e49210e82bf33eb0fd9b974b29d5a46c1bd5858171948e1a", "task_kind": "ooc", "wait_seconds": 0.004483500029891729, "world_seed": 10152}].

Total job wall time 2.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_initial_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_initial_dev.log`.

### 2026-10-12 01:23:25 UTC — ooc_0000

Status: **complete**. Finished 2026-10-12 01:26:31 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_lr0.0003_progress0.05.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 11 --world-start 20000 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3126 non-fault episodes, 37 surfaced faults, 8 truncations; outcomes {'act_clear': 2346, 'defeat': 772, 'GameOutcome.UNDECIDED': 8}; 192,177 valid transitions.

Mean terminal/cutoff floor 16.35; mean fight HP cost 18.13.

PPO: 92 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 1.0631125420331955, "preparation_seconds": 0.9604140841402113, "samples": 5544, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "1b2718e6000dd40f84213fae2758987784637e8a8a9a2799d4e1c9ee59a4637d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 12, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20033 floor=3 encounter=1 error=native battle playout evoked undefined behavior", "task_id": "b5c3162709c75f0cb057fa02e95f40270e3584156a99dede8ec0cb39c84b13af", "task_kind": "ooc", "wait_seconds": 0.004330958006903529, "world_seed": 20033}, {"action_trace_sha256": "9c3c8a9b3c395185a168290d180b92f55e3fec8e61f4d50496eb6a323b38e708", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 43, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20059 floor=13 encounter=11 error=native battle playout evoked undefined behavior", "task_id": "1d5ba29f8e1f37746595036a12f7a2b8e7fce1f24dc322ad54aaddf4ed2b4c36", "task_kind": "ooc", "wait_seconds": 0.004072416806593537, "world_seed": 20059}, {"action_trace_sha256": "f821749f353c9c735f320293af5feb4d30b34cdab161b11dde52e2db189f6245", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20003 floor=16 encounter=19 error=native battle playout evoked undefined behavior", "task_id": "21d4db66c2cb405d31dcdc2d03462fb3c57b561b7c13c7e8a35212d2bbb85fef", "task_kind": "ooc", "wait_seconds": 0.010448042070493102, "world_seed": 20003}, {"action_trace_sha256": "1b2718e6000dd40f84213fae2758987784637e8a8a9a2799d4e1c9ee59a4637d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 12, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20033 floor=3 encounter=1 error=native battle playout evoked undefined behavior", "task_id": "b5c3162709c75f0cb057fa02e95f40270e3584156a99dede8ec0cb39c84b13af", "task_kind": "ooc", "wait_seconds": 0.004216458182781935, "world_seed": 20033}, {"action_trace_sha256": "1af364bc62a263bfe4a8454db39412f69b5aebe0ab65cc2a01912505e5ab94fe", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 27, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20042 floor=7 encounter=7 error=native battle playout evoked undefined behavior", "task_id": "58dad565a46795a5c3a91ad4f899043c0fe84f809d137de6b16377ce00b15763", "task_kind": "ooc", "wait_seconds": 0.0023772921413183212, "world_seed": 20042}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.25s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.log`.

### 2026-10-12 01:26:31 UTC — ooc_0000_dev

Status: **complete**. Finished 2026-10-12 01:26:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 8, 'act_clear': 23}; 1,912 valid transitions.

Mean terminal/cutoff floor 16.61; mean fight HP cost 18.23.

Fault evidence: [{"action_trace_sha256": "b14348423c20c9e847c4f76ab70b0c52693ebf44946dbaa033a50d20d8fc3069", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 13, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10152 floor=4 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "4b366888d6184231e49210e82bf33eb0fd9b974b29d5a46c1bd5858171948e1a", "task_kind": "ooc", "wait_seconds": 0.006094082957133651, "world_seed": 10152}].

Total job wall time 2.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000_dev.log`.

### 2026-10-12 01:26:36 UTC — ooc_0001

Status: **complete**. Finished 2026-10-12 01:29:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0000.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 12 --world-start 20128 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3116 non-fault episodes, 41 surfaced faults, 7 truncations; outcomes {'defeat': 617, 'act_clear': 2492, 'GameOutcome.UNDECIDED': 7}; 187,541 valid transitions.

Mean terminal/cutoff floor 16.52; mean fight HP cost 13.30.

PPO: 85 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.8606785000301898, "preparation_seconds": 0.7583623339887708, "samples": 4667, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "1cd5b746b554b0c2199eee6ada5e1d3c43376890eb085da25868450aae188c67", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 23, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20265 floor=7 encounter=7 error=native battle playout evoked undefined behavior", "task_id": "6997e6f99eda0ad677d3aad28ec89a546b6450fe0ccc00bc547aa1e73e5322df", "task_kind": "ooc", "wait_seconds": 0.003501832950860262, "world_seed": 20265}, {"action_trace_sha256": "65507562bfe74e1343a118a9030e867af8cc8910796e847fbc0affedc3f3e802", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20142 floor=11 encounter=12 error=native battle playout evoked undefined behavior", "task_id": "2eae28c396a546025286cce7224303bf2374f24519addc1499083ed42b548c4d", "task_kind": "ooc", "wait_seconds": 0.003940457943826914, "world_seed": 20142}, {"action_trace_sha256": "21002d9e36e8f5fa5ac573bb9fa992372fc3a529213991bfb76c4a8aef52731e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 56, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20270 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "24cfbb0fcb8af7f47a5a35994fb8b5d22de5aa14f9b2cedd78bfdcb973a7f606", "task_kind": "ooc", "wait_seconds": 0.010644125053659081, "world_seed": 20270}, {"action_trace_sha256": "721fd52402dc01b5920c7a9eb283025a1fd3273491e380d81fcab56b34105712", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 57, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20253 floor=16 encounter=18 error=native battle playout evoked undefined behavior", "task_id": "a9fb9fb92d3d7f30199c7376131dba6d6bc75403a2e729057026871721af2174", "task_kind": "ooc", "wait_seconds": 0.008696750039234757, "world_seed": 20253}, {"action_trace_sha256": "e36959cce5a7bd266786cbc7e8152c87e20b265a59edc338a0acff42e7b5bf8b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 55, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20270 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "24cfbb0fcb8af7f47a5a35994fb8b5d22de5aa14f9b2cedd78bfdcb973a7f606", "task_kind": "ooc", "wait_seconds": 0.008784792153164744, "world_seed": 20270}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.06s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.log`.

### 2026-10-12 01:29:41 UTC — ooc_0001_dev

Status: **complete**. Finished 2026-10-12 01:29:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 6, 'act_clear': 25}; 1,863 valid transitions.

Mean terminal/cutoff floor 16.74; mean fight HP cost 12.39.

Fault evidence: [{"action_trace_sha256": "e5d727cc5157919fd8386697c9c136ae488aa7341ce1ea182cd93ddbb1a75eb4", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10011 floor=14 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "5856fc3f45a5da8007637f9ab5e9dce1e2154d9a7aed8bde5688392af3d4818a", "task_kind": "ooc", "wait_seconds": 0.0046536249574273825, "world_seed": 10011}].

Total job wall time 2.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001_dev.log`.

### 2026-10-12 01:29:46 UTC — ooc_0002

Status: **complete**. Finished 2026-10-12 01:32:52 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 13 --world-start 20256 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3108 non-fault episodes, 48 surfaced faults, 8 truncations; outcomes {'act_clear': 2376, 'defeat': 724, 'GameOutcome.UNDECIDED': 8}; 182,820 valid transitions.

Mean terminal/cutoff floor 16.48; mean fight HP cost 16.87.

PPO: 87 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 1.0157689589541405, "preparation_seconds": 0.9103037500753999, "samples": 4683, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "4bac6a59fb1ccac2eead6ed7e1e1e8f8fa7692b7a528d3770487f687a0fc9572", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 20, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20265 floor=7 encounter=7 error=native battle playout evoked undefined behavior", "task_id": "6997e6f99eda0ad677d3aad28ec89a546b6450fe0ccc00bc547aa1e73e5322df", "task_kind": "ooc", "wait_seconds": 0.0038416660390794277, "world_seed": 20265}, {"action_trace_sha256": "3dc25ae546a026764d45751fa02bb6472307b505d830537473f1441e6e702c7a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 12, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20350 floor=5 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "40ff110335ea0f41e904bcdd6da7d77e840512747e0dc9b8356633f2020e2d26", "task_kind": "ooc", "wait_seconds": 0.0038321667816489935, "world_seed": 20350}, {"action_trace_sha256": "cfcad156d31ed40bd03bdf21f67f7c27b9ba928b879be001d4b0d01ce35520da", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 16, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20350 floor=5 encounter=1 error=native battle playout evoked undefined behavior", "task_id": "40ff110335ea0f41e904bcdd6da7d77e840512747e0dc9b8356633f2020e2d26", "task_kind": "ooc", "wait_seconds": 3.0838418751955032e-06, "world_seed": 20350}, {"action_trace_sha256": "08380b6b11b641c3b69f4d95bc2d73961d94b7e34b47fdc55fc63c77f7ea2706", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 47, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20417 floor=14 encounter=12 error=native battle playout evoked undefined behavior", "task_id": "af341f71d437d0e0055e0c7dda96f3717c6940878f8eba4a44e17457e58d4061", "task_kind": "ooc", "wait_seconds": 0.004202874843031168, "world_seed": 20417}, {"action_trace_sha256": "b058d78d7c91b98c60c02c1b5cd02ac21eb4cfa93360ef60ac10192d0ff1cde0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 56, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20270 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "24cfbb0fcb8af7f47a5a35994fb8b5d22de5aa14f9b2cedd78bfdcb973a7f606", "task_kind": "ooc", "wait_seconds": 0.016350624850019813, "world_seed": 20270}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.23s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.log`.

### 2026-10-12 01:32:52 UTC — ooc_0002_dev

Status: **complete**. Finished 2026-10-12 01:32:57 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 7, 'act_clear': 25}; 1,885 valid transitions.

Mean terminal/cutoff floor 16.53; mean fight HP cost 15.94.

Total job wall time 2.50s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002_dev.log`.

### 2026-10-12 01:32:57 UTC — ooc_0003

Status: **complete**. Finished 2026-10-12 01:36:02 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0002.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 14 --world-start 20384 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 26 rounds, 3264 non-fault episodes, 33 surfaced faults, 7 truncations; outcomes {'defeat': 619, 'act_clear': 2638, 'GameOutcome.UNDECIDED': 7}; 194,600 valid transitions.

Mean terminal/cutoff floor 16.67; mean fight HP cost 13.63.

PPO: 101 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 1.0673969169147313, "preparation_seconds": 0.9619425418786705, "samples": 5832, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "47b2625296af9be4d7a17dfffd1ebb9b0d71e49f1af4c48749cd39ee0bca6906", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 33, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20442 floor=10 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "e040afb1e8aa40e9d21d72a19d9955f05f53c2b2208ece6e5a7110dcaaf8d093", "task_kind": "ooc", "wait_seconds": 0.003792916890233755, "world_seed": 20442}, {"action_trace_sha256": "ffa5fbd61b22e2218f90e76521faadebd228b0b0631db7916e753a588784c0aa", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 20, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20497 floor=6 encounter=16 error=native battle playout evoked undefined behavior", "task_id": "d7170c6167d1bf4a72aa847fe09a1c307beab08f8486f59132e4a45e4ce5f353", "task_kind": "ooc", "wait_seconds": 0.006603709189221263, "world_seed": 20497}, {"action_trace_sha256": "bab4af11fa43639dfb25812b926cc66b33e3eed261f2d30c8470ff63b661e183", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 21, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20497 floor=6 encounter=16 error=native battle playout evoked undefined behavior", "task_id": "d7170c6167d1bf4a72aa847fe09a1c307beab08f8486f59132e4a45e4ce5f353", "task_kind": "ooc", "wait_seconds": 0.00017741695046424866, "world_seed": 20497}, {"action_trace_sha256": "91fdafca7607887affcc5d598e3fd7e1920e460467ead15d933b8c728402581f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 33, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20442 floor=10 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "e040afb1e8aa40e9d21d72a19d9955f05f53c2b2208ece6e5a7110dcaaf8d093", "task_kind": "ooc", "wait_seconds": 0.0039010420441627502, "world_seed": 20442}, {"action_trace_sha256": "d77ad0ac254b229149b65620c51e1a33427301813977650466c8fcf80e41064a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 33, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20442 floor=10 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "e040afb1e8aa40e9d21d72a19d9955f05f53c2b2208ece6e5a7110dcaaf8d093", "task_kind": "ooc", "wait_seconds": 0.0038972918409854174, "world_seed": 20442}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.27s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.log`.

### 2026-10-12 01:36:02 UTC — ooc_0003_dev

Status: **complete**. Finished 2026-10-12 01:36:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 6, 'act_clear': 26}; 1,913 valid transitions.

Mean terminal/cutoff floor 16.09; mean fight HP cost 13.16.

Total job wall time 2.53s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003_dev.log`.

### 2026-10-12 01:36:07 UTC — ooc_0004

Status: **complete**. Finished 2026-10-12 01:39:13 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0003.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 15 --world-start 20512 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 27 rounds, 3452 non-fault episodes, 4 surfaced faults, 7 truncations; outcomes {'defeat': 542, 'act_clear': 2903, 'GameOutcome.UNDECIDED': 7}; 207,801 valid transitions.

Mean terminal/cutoff floor 16.67; mean fight HP cost 10.96.

PPO: 132 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 1.3413155828602612, "preparation_seconds": 1.2370544578880072, "samples": 7562, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "c6306de59d51f4933bf8aba78ca1315c9804783a1b7dd4c7e9a6dc443dece8ce", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20599 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "bcae26e7e63fec8e09aa2ab44ea8c7e09f93f355c6d033a76a3014a21d495b42", "task_kind": "ooc", "wait_seconds": 0.010379124898463488, "world_seed": 20599}, {"action_trace_sha256": "c8ff384dd253711ebd204a1e76f57c34364e5831a84c5988d779f58acfbb53c8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 36, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20547 floor=10 encounter=11 error=native battle playout evoked undefined behavior", "task_id": "d5c88c64ff47f7c798b3ae30a0e2f2fd615c16faba4edc4c0cd2a4e0c582d228", "task_kind": "ooc", "wait_seconds": 0.003985583782196045, "world_seed": 20547}, {"action_trace_sha256": "d9f5c84255e8eb9fad272781a64480b62fbdb9e99a41114d2171ebf884fd992f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 36, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20547 floor=10 encounter=11 error=native battle playout evoked undefined behavior", "task_id": "d5c88c64ff47f7c798b3ae30a0e2f2fd615c16faba4edc4c0cd2a4e0c582d228", "task_kind": "ooc", "wait_seconds": 0.00440125004388392, "world_seed": 20547}, {"action_trace_sha256": "378cfa93a39f5b9efbad2037219749468eb938ecb518c92e6a876d3fb3dd913e", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20634 floor=12 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "e51945c81f4865f94cfac0e825ce88d4e79e07080e5b6d08564bf3ec9fa5e2bf", "task_kind": "ooc", "wait_seconds": 0.004601250169798732, "world_seed": 20634}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.55s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.log`.

### 2026-10-12 01:39:13 UTC — ooc_0004_dev

Status: **complete**. Finished 2026-10-12 01:39:18 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 3, 'act_clear': 29}; 1,982 valid transitions.

Mean terminal/cutoff floor 16.91; mean fight HP cost 5.16.

Total job wall time 2.58s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004_dev.log`.

### 2026-10-12 01:39:18 UTC — ooc_0005

Status: **complete**. Finished 2026-10-12 01:42:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 16 --world-start 20640 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3181 non-fault episodes, 19 surfaced faults, 0 truncations; outcomes {'act_clear': 2818, 'defeat': 363}; 197,012 valid transitions.

Mean terminal/cutoff floor 16.83; mean fight HP cost 7.09.

PPO: 239 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 1.3810100418049842, "preparation_seconds": 1.2685579168610275, "samples": 7844, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "a8d3d8dc8f4d432208d0b4f43d548f8103bb6ef176abda6548295327cb75d9be", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 51, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20764 floor=13 encounter=11 error=native battle playout evoked undefined behavior", "task_id": "039fae097bf0ae5c0a7e8e3fa54737614b837242d4360c8397e75f8b36d6657e", "task_kind": "ooc", "wait_seconds": 0.007016249932348728, "world_seed": 20764}, {"action_trace_sha256": "c90a54f5a495a3bca961bebc9a52b1def13f939745947d60ee2152e9394f9f82", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 36, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20692 floor=10 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "3e909c11e9144e7336944fad84c71e352832469239f204f1440951abca060a81", "task_kind": "ooc", "wait_seconds": 0.003481999970972538, "world_seed": 20692}, {"action_trace_sha256": "89ccdc3e6f22aeade188564e6b817f3e70167eb9c111b5dca1aac3c4314f2fd3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 35, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20684 floor=10 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "f4253df39626800a0639adb1cb8c126324a36558f4903d1a9a3e73ab8721ba45", "task_kind": "ooc", "wait_seconds": 0.00597937498241663, "world_seed": 20684}, {"action_trace_sha256": "2ec352b2e4763b917bc739b5f89aa956eedf6e0b4985181daa195285c597340e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 46, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20703 floor=14 encounter=10 error=native battle playout evoked undefined behavior", "task_id": "98ebfac71d0e015de127364c1c3b534fee8e76368529b889e1466e36c2b31dbe", "task_kind": "ooc", "wait_seconds": 0.004932542098686099, "world_seed": 20703}, {"action_trace_sha256": "d68f0a520dc1e2349188a3f432920d8b22fe2cb712d829ebaf77d90853a95a4a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 54, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20741 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "67befa0a5d31caddebd6141a91a1d19fdb34cf95b683c3546f8aa23f02bfea4a", "task_kind": "ooc", "wait_seconds": 0.010331666097044945, "world_seed": 20741}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.log`.

### 2026-10-12 01:42:24 UTC — ooc_0005_dev

Status: **complete**. Finished 2026-10-12 01:42:29 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 3, 'act_clear': 28}; 1,922 valid transitions.

Mean terminal/cutoff floor 16.90; mean fight HP cost 6.48.

Fault evidence: [{"action_trace_sha256": "6e610fefffd05ffc01a8b4de20831b84410ca72769913ad66aac89330f8474ce", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10011 floor=14 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "5856fc3f45a5da8007637f9ab5e9dce1e2154d9a7aed8bde5688392af3d4818a", "task_kind": "ooc", "wait_seconds": 0.005313416942954063, "world_seed": 10011}].

Total job wall time 2.56s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005_dev.log`.

### 2026-10-12 01:42:29 UTC — ooc_0006

Status: **complete**. Finished 2026-10-12 01:45:34 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0005.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 17 --world-start 20768 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3139 non-fault episodes, 14 surfaced faults, 8 truncations; outcomes {'defeat': 222, 'act_clear': 2909, 'GameOutcome.UNDECIDED': 8}; 197,013 valid transitions.

Mean terminal/cutoff floor 16.81; mean fight HP cost 3.63.

PPO: 254 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.9013066671323031, "preparation_seconds": 0.7914315420202911, "samples": 4710, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "f909a020b5e99fb897344faad59584660e0e6b574dc34005daaa3f9f42c2490e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20823 floor=14 encounter=13 error=native battle playout evoked undefined behavior", "task_id": "1379db62ac109b151eb215d1244f64c5a54b7b7776fc505848d17caa1dbb6e9a", "task_kind": "ooc", "wait_seconds": 0.004459833027794957, "world_seed": 20823}, {"action_trace_sha256": "db0086c23df29b4a5512dd8e26873aeb02894e05ca4a138649ba1f1434dabee5", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20823 floor=14 encounter=13 error=native battle playout evoked undefined behavior", "task_id": "1379db62ac109b151eb215d1244f64c5a54b7b7776fc505848d17caa1dbb6e9a", "task_kind": "ooc", "wait_seconds": 0.003773583797737956, "world_seed": 20823}, {"action_trace_sha256": "b6f9ae56545793274701b9913ce7ae926e4114e51f90e231d715794a15383224", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 53, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20823 floor=14 encounter=10 error=native battle playout evoked undefined behavior", "task_id": "1379db62ac109b151eb215d1244f64c5a54b7b7776fc505848d17caa1dbb6e9a", "task_kind": "ooc", "wait_seconds": 0.0037599578499794006, "world_seed": 20823}, {"action_trace_sha256": "c2232540f35ca8c077cb66de44324ba3e3274f0c5af54e10d951b49555cf1703", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20823 floor=14 encounter=13 error=native battle playout evoked undefined behavior", "task_id": "1379db62ac109b151eb215d1244f64c5a54b7b7776fc505848d17caa1dbb6e9a", "task_kind": "ooc", "wait_seconds": 0.004215250024572015, "world_seed": 20823}, {"action_trace_sha256": "910f7fd9f840331646bcd6022690e995018d101005e373a1777ea97951208f83", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 53, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=20823 floor=14 encounter=10 error=native battle playout evoked undefined behavior", "task_id": "1379db62ac109b151eb215d1244f64c5a54b7b7776fc505848d17caa1dbb6e9a", "task_kind": "ooc", "wait_seconds": 3.7909485399723053e-06, "world_seed": 20823}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.30s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.log`.

### 2026-10-12 01:45:34 UTC — ooc_0006_dev

Status: **complete**. Finished 2026-10-12 01:45:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 30 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'act_clear': 26, 'defeat': 4}; 1,866 valid transitions.

Mean terminal/cutoff floor 16.87; mean fight HP cost 8.23.

Fault evidence: [{"action_trace_sha256": "e5f0171595af2c45cddfe0a9f7c43f96b59e2770b5ff1e4d4010a5b5c504af59", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 54, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10218 floor=16 encounter=19 error=native battle playout evoked undefined behavior", "task_id": "d073a34ba2a731cbcef2a90e3a423f7edc55009b7f0225cfdd7ba118b9128d46", "task_kind": "ooc", "wait_seconds": 0.012662458000704646, "world_seed": 10218}, {"action_trace_sha256": "d51a1711e277e8d52bd66e53c0072c65b27c46d25097c4c09819c894f0085f3a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 51, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10011 floor=14 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "5856fc3f45a5da8007637f9ab5e9dce1e2154d9a7aed8bde5688392af3d4818a", "task_kind": "ooc", "wait_seconds": 0.0046388329938054085, "world_seed": 10011}].

Total job wall time 2.99s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006_dev.log`.

### 2026-10-12 01:45:39 UTC — ooc_0007

Status: **complete**. Finished 2026-10-12 01:48:45 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0006.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 18 --world-start 20896 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 26 rounds, 3304 non-fault episodes, 1 surfaced faults, 8 truncations; outcomes {'act_clear': 2991, 'defeat': 305, 'GameOutcome.UNDECIDED': 8}; 206,807 valid transitions.

Mean terminal/cutoff floor 16.79; mean fight HP cost 5.60.

PPO: 272 optimizer steps; last update {"algorithm_version": "masked_ppo_v1_after_update_kl", "early_stop": true, "elapsed_seconds": 1.1641339580528438, "preparation_seconds": 1.0592617909424007, "rejected_kl": null, "rejected_minibatches": 0, "samples": 6432, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "66e87eb6d37480d0ec84eea86f298e8fa410dfebbdf154f073b862bd166ea869", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 45, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21012 floor=12 encounter=12 error=native battle playout evoked undefined behavior", "task_id": "abe4dcdc509d460bb0e3f7b416dad260373d35caae41de2bce5bb711c54c7aa1", "task_kind": "ooc", "wait_seconds": 2.959044650197029e-06, "world_seed": 21012}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.35s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.log`.

### 2026-10-12 01:48:45 UTC — ooc_0007_dev

Status: **complete**. Finished 2026-10-12 01:48:50 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 3, 'act_clear': 28}; 1,941 valid transitions.

Mean terminal/cutoff floor 16.90; mean fight HP cost 6.06.

Fault evidence: [{"action_trace_sha256": "039a06d260ce0e8ed9d03b554ab4cf7ce13cbbe63abae85e3d4d783664095866", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 13, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10152 floor=4 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "4b366888d6184231e49210e82bf33eb0fd9b974b29d5a46c1bd5858171948e1a", "task_kind": "ooc", "wait_seconds": 0.005145875038579106, "world_seed": 10152}].

Total job wall time 2.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007_dev.log`.

### 2026-10-12 01:48:50 UTC — ooc_0008

Status: **complete**. Finished 2026-10-12 01:51:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 19 --world-start 21024 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 26 rounds, 3209 non-fault episodes, 15 surfaced faults, 8 truncations; outcomes {'act_clear': 2851, 'defeat': 350, 'GameOutcome.UNDECIDED': 8}; 197,359 valid transitions.

Mean terminal/cutoff floor 16.82; mean fight HP cost 7.13.

PPO: 314 optimizer steps; last update {"algorithm_version": "masked_ppo_v1_after_update_kl", "early_stop": true, "elapsed_seconds": 0.21148975007236004, "preparation_seconds": 0.20227262494154274, "rejected_kl": null, "rejected_minibatches": 0, "samples": 1096, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "d2e8630c7b6acd7757757d5747e6ee7499c2b02a4ed02fbb6780500a37f89bba", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 61, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21116 floor=16 encounter=19 error=native battle playout evoked undefined behavior", "task_id": "8820d83f18973122535548ee5b391e7c7dc663fcc288ae8659e1d0e508c18bdd", "task_kind": "ooc", "wait_seconds": 0.0193637500051409, "world_seed": 21116}, {"action_trace_sha256": "1ec9e90f299066387c296fcd856da76b632fd6c2a0d313052b3d32ecf763ea6b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 6, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21130 floor=2 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "b5cc63e41cd57d0de8b9a18796adf1c6f642b4d7e4505e874adcb0206817aa21", "task_kind": "ooc", "wait_seconds": 3.1669624149799347e-06, "world_seed": 21130}, {"action_trace_sha256": "673de70f96c56a7565bcd580a8d31a77ba0fa2c92521fd7db40078fe938dc22a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 6, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21130 floor=2 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "b5cc63e41cd57d0de8b9a18796adf1c6f642b4d7e4505e874adcb0206817aa21", "task_kind": "ooc", "wait_seconds": 0.005810917122289538, "world_seed": 21130}, {"action_trace_sha256": "1ec9e90f299066387c296fcd856da76b632fd6c2a0d313052b3d32ecf763ea6b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 6, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21130 floor=2 encounter=4 error=native battle playout evoked undefined behavior", "task_id": "b5cc63e41cd57d0de8b9a18796adf1c6f642b4d7e4505e874adcb0206817aa21", "task_kind": "ooc", "wait_seconds": 0.004555625142529607, "world_seed": 21130}, {"action_trace_sha256": "784832c89586fb87b08b70a9b258689750ea37b2c83a98f38a0f9d541d56626a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 5, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21097 floor=1 encounter=2 error=native battle playout evoked undefined behavior", "task_id": "1e9220cb7160134526602bf09d6404a8b59d0cfef7d38166675f780a92472e0f", "task_kind": "ooc", "wait_seconds": 0.004023207817226648, "world_seed": 21097}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.39s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.log`.

### 2026-10-12 01:51:55 UTC — ooc_0008_dev

Status: **complete**. Finished 2026-10-12 01:52:00 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 6, 'act_clear': 25}; 1,886 valid transitions.

Mean terminal/cutoff floor 16.74; mean fight HP cost 15.23.

Fault evidence: [{"action_trace_sha256": "5753251ace3658086d7a57e1759d1a0d815d567d0f30032e2386ff86ff09375c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10011 floor=14 encounter=6 error=native battle playout evoked undefined behavior", "task_id": "5856fc3f45a5da8007637f9ab5e9dce1e2154d9a7aed8bde5688392af3d4818a", "task_kind": "ooc", "wait_seconds": 0.0051167500205338, "world_seed": 10011}].

Total job wall time 2.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008_dev.log`.

### 2026-10-12 01:52:01 UTC — ooc_0009

Status: **complete**. Finished 2026-10-12 01:55:06 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0008.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 20 --world-start 21152 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3170 non-fault episodes, 14 surfaced faults, 7 truncations; outcomes {'defeat': 320, 'act_clear': 2843, 'GameOutcome.UNDECIDED': 7}; 197,525 valid transitions.

Mean terminal/cutoff floor 16.82; mean fight HP cost 6.99.

PPO: 250 optimizer steps; last update {"algorithm_version": "masked_ppo_v1_after_update_kl", "early_stop": true, "elapsed_seconds": 1.2300352919846773, "preparation_seconds": 1.1224184590391815, "rejected_kl": null, "rejected_minibatches": 0, "samples": 6720, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "36e9e35bfeee0561845281159979103df1e18a975f1584e8f7ce184727b0605f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 20, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21292 floor=5 encounter=5 error=native battle playout evoked undefined behavior", "task_id": "414d71fca1ebad044a2440f3d71f3d60b5cd1b4b703b7265773fd12e6c2045e0", "task_kind": "ooc", "wait_seconds": 0.006579125067219138, "world_seed": 21292}, {"action_trace_sha256": "54b753ba1adae02b6f4c28154d70e47fc66af305cfeffc7b01333cb193245904", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 57, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21283 floor=14 encounter=11 error=native battle playout evoked undefined behavior", "task_id": "bbf0e40558416c765c2a265d5197f705656b5323fc1fe7b4c2a73f0e5afa94ed", "task_kind": "ooc", "wait_seconds": 0.006164374994114041, "world_seed": 21283}, {"action_trace_sha256": "b859a8e159241d43b2c4ae8989600713cddcb0490d72a24686094da0d81566b7", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21237 floor=14 encounter=9 error=native battle playout evoked undefined behavior", "task_id": "a2177b80dd502d9ace524473e93c3419679e5bc85131fe223c1f60e6d20bff16", "task_kind": "ooc", "wait_seconds": 0.004783625015988946, "world_seed": 21237}, {"action_trace_sha256": "5f520bbfe38de866e6ff0f29f5330218c8d44c37cd191dd9aeb68897211bdebb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 22, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21189 floor=7 encounter=12 error=native battle playout evoked undefined behavior", "task_id": "b0c7daf35f809627731846cd397f981348085386ff30c689a6bd5dc833d36c0e", "task_kind": "ooc", "wait_seconds": 0.004845625022426248, "world_seed": 21189}, {"action_trace_sha256": "5037939c0866e59fed4825dbdef4953b4af6dc2e5399635239edace37473c545", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 22, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21189 floor=7 encounter=12 error=native battle playout evoked undefined behavior", "task_id": "b0c7daf35f809627731846cd397f981348085386ff30c689a6bd5dc833d36c0e", "task_kind": "ooc", "wait_seconds": 0.005855625029653311, "world_seed": 21189}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.43s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.log`.

### 2026-10-12 01:55:06 UTC — ooc_0009_dev

Status: **complete**. Finished 2026-10-12 01:55:11 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 6, 'act_clear': 26}; 2,036 valid transitions.

Mean terminal/cutoff floor 16.81; mean fight HP cost 14.69.

Total job wall time 2.64s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009_dev.log`.

### 2026-10-12 01:55:11 UTC — ooc_0010

Status: **complete**. Finished 2026-10-12 01:58:13 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0010.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0010.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0009.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 21 --world-start 21280 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 25 rounds, 3175 non-fault episodes, 10 surfaced faults, 8 truncations; outcomes {'defeat': 274, 'act_clear': 2893, 'GameOutcome.UNDECIDED': 8}; 201,188 valid transitions.

Mean terminal/cutoff floor 16.85; mean fight HP cost 5.46.

PPO: 304 optimizer steps; last update {"algorithm_version": "masked_ppo_v1_after_update_kl", "early_stop": true, "elapsed_seconds": 1.224218624876812, "preparation_seconds": 1.1150840828195214, "rejected_kl": null, "rejected_minibatches": 0, "samples": 6825, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "2b56371e17d481b44a1ab8796235d1b6fd739898d7d0cb3cb8ba429c6a331183", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 14, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21433 floor=4 encounter=1 error=native battle playout evoked undefined behavior", "task_id": "62e84395df34ca7c8c2af30a58fa2f57c7f0b7c26f86de7a9aec594f293a1b82", "task_kind": "ooc", "wait_seconds": 0.0046444998588413, "world_seed": 21433}, {"action_trace_sha256": "bfdbba032c9b81bee8d57ee729e86ae00e51611b7bdd037c21417cc9980ca5b0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 39, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21378 floor=11 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "b8777961923c50bdb723d4d181733eaafe64ee3c228b0400d6096939d82b67bd", "task_kind": "ooc", "wait_seconds": 0.001071541104465723, "world_seed": 21378}, {"action_trace_sha256": "bfdbba032c9b81bee8d57ee729e86ae00e51611b7bdd037c21417cc9980ca5b0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 39, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21378 floor=11 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "b8777961923c50bdb723d4d181733eaafe64ee3c228b0400d6096939d82b67bd", "task_kind": "ooc", "wait_seconds": 0.00503670796751976, "world_seed": 21378}, {"action_trace_sha256": "bfdbba032c9b81bee8d57ee729e86ae00e51611b7bdd037c21417cc9980ca5b0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 39, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21378 floor=11 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "b8777961923c50bdb723d4d181733eaafe64ee3c228b0400d6096939d82b67bd", "task_kind": "ooc", "wait_seconds": 0.005417749984189868, "world_seed": 21378}, {"action_trace_sha256": "6f4bb0540d61ae5dec6a2f1d1f04026178a84c44ff24f02d3ffb8201370c5a28", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 39, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=21378 floor=11 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "b8777961923c50bdb723d4d181733eaafe64ee3c228b0400d6096939d82b67bd", "task_kind": "ooc", "wait_seconds": 0.0037709998432546854, "world_seed": 21378}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.74s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0010.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0010.log`.

### 2026-10-12 01:59:51 UTC — kl_legacy_after_update

Status: **complete**. Finished 2026-10-12 02:00:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update.metrics.json --task-kind ooc --split train --world-start 23000 --episodes 64 --seed 11 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --learning-rate 3e-4 --reward-mode progress --progress-floor-weight .05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 60 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 17 rounds, 1086 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'act_clear': 957, 'defeat': 129}; 65,954 valid transitions.

Mean terminal/cutoff floor 16.82; mean fight HP cost 8.43.

PPO: 91 optimizer steps; last update {"algorithm_version": "masked_ppo_v1_after_update_kl", "approx_kl": 0.01821324661723338, "clip_fraction": 0.026781249791383743, "early_stop": true, "elapsed_seconds": 1.036618584068492, "entropy": 0.06798295974731446, "policy_loss": -0.005039682146161795, "preparation_seconds": 0.6420440000947565, "ratio_mean": 1.000363576412201, "rejected_kl": null, "rejected_minibatches": 0, "samples": 3872, "steps": 5, "value_loss": 0.039534016698598864}.

Fault evidence: [{"action_trace_sha256": "8fa24012271a5654ec3718dace16546e6e662717ded07110074801605b9fb5ad", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.015626666136085987, "world_seed": 23010}, {"action_trace_sha256": "98b0463f16e66df4af840be143b2175303a7b7313266f02b34f858f00d228c7a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 49, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23053 floor=13 encounter=14 error=native battle playout evoked undefined behavior", "task_id": "5cbe4805d1024581ec54e3bf420097519cf4852eee5a9819ef628d81638089c4", "task_kind": "ooc", "wait_seconds": 0.0042563751339912415, "world_seed": 23053}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 60.05s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/kl_legacy_after_update.log`.

### 2026-10-12 02:00:51 UTC — kl_legacy_after_update_dev

Status: **complete**. Finished 2026-10-12 02:00:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'act_clear': 29, 'defeat': 2}; 1,891 valid transitions.

Mean terminal/cutoff floor 16.87; mean fight HP cost 3.26.

Fault evidence: [{"action_trace_sha256": "25019c31b02b0d12864273a35125bcf0cd3aeaeb11db227a1fecaeee6fd835cc", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 55, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10218 floor=16 encounter=19 error=native battle playout evoked undefined behavior", "task_id": "d073a34ba2a731cbcef2a90e3a423f7edc55009b7f0225cfdd7ba118b9128d46", "task_kind": "ooc", "wait_seconds": 0.005695500178262591, "world_seed": 10218}].

Total job wall time 2.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/legacy_after_update_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/kl_legacy_after_update_dev.log`.

### 2026-10-12 02:00:55 UTC — kl_before_update

Status: **complete**. Finished 2026-10-12 02:01:56 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.metrics.json --task-kind ooc --split train --world-start 23000 --episodes 64 --seed 11 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --learning-rate 3e-4 --reward-mode progress --progress-floor-weight .05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 60 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer --kl-guard-before-update
```

Live collection: 17 rounds, 1080 non-fault episodes, 8 surfaced faults, 1 truncations; outcomes {'act_clear': 944, 'defeat': 135, 'GameOutcome.UNDECIDED': 1}; 66,027 valid transitions.

Mean terminal/cutoff floor 16.81; mean fight HP cost 8.83.

PPO: 75 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.7339596250094473, "preparation_seconds": 0.6321917909663171, "rejected_kl": null, "rejected_minibatches": 0, "samples": 3840, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "b618aea89834977b1195c48629ba65e3a17f0d9ea29f24668817e730b28d309e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.010980458930134773, "world_seed": 23010}, {"action_trace_sha256": "ba89a272628670ee1e860b61622eda70a4a584102ebcf3c5e6ebd5130baeb2af", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.010077625047415495, "world_seed": 23010}, {"action_trace_sha256": "127d73aca239b004828e27f4e5fe15ddb634088b5eb9e7c35003fe4e4e72fcc1", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.010423542000353336, "world_seed": 23010}, {"action_trace_sha256": "91a4421649dea67703a939d108cc77cfeef27ea486cfd83b3bf3f498c1c1f16b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.010484833037480712, "world_seed": 23010}, {"action_trace_sha256": "91a4421649dea67703a939d108cc77cfeef27ea486cfd83b3bf3f498c1c1f16b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=23010 floor=16 encounter=20 error=native battle playout evoked undefined behavior", "task_id": "3925b926478e28d0ed439786c70c0cce90fa03f67ba4e233935c0afe436ce0fe", "task_kind": "ooc", "wait_seconds": 0.010613500140607357, "world_seed": 23010}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 60.84s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/kl_before_update.log`.

### 2026-10-12 02:01:56 UTC — kl_before_update_dev

Status: **complete**. Finished 2026-10-12 02:01:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 1, 'act_clear': 31}; 1,983 valid transitions.

Mean terminal/cutoff floor 16.97; mean fight HP cost 1.84.

Total job wall time 2.09s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/kl_before_update_dev.log`.

### 2026-10-12 02:01:59 UTC — corrected_teacher_scripted

Status: **complete**. Finished 2026-10-12 02:02:03 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python scripts/train_nonllm.py bc --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/ooc_lookahead_native011/scripted_teacher.jsonl --dev-teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl --phase ooc --device mps --inference-device cpu --threads 2 --batch-size 256 --learning-rate 0.0003 --seed 11 --updates 20 --seconds 30 --metric-interval 1 --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_scripted.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_scripted.metrics.json
```

Training: 20 updates, 5,120 examples, 0.41s; 12,407 examples/s; mean loss 2.3170.

Development imitation: {"accuracy": 0.7840531561461794, "examples": 1505, "loss": 1.1251314790541944, "menu_accuracy": {"choice": 0.6886973180076629, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.7840531561461794}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.6886973180076629, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 3.36s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_scripted.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_scripted.log`.

### 2026-10-12 02:02:03 UTC — corrected_teacher_scripted_dev

Status: **complete**. Finished 2026-10-12 02:02:05 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_scripted.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_scripted.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_scripted_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'act_clear': 19, 'defeat': 12}; 1,908 valid transitions.

Mean terminal/cutoff floor 16.42; mean fight HP cost 30.55.

Fault evidence: [{"action_trace_sha256": "9e61f82b4dc6547673bf202df7e42a12b93571ee04cc952d0f1aafb9188d49ac", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 13, "fault_stage": "step", "message": "RuntimeError: battle resolution failed seed=10152 floor=4 encounter=3 error=native battle playout evoked undefined behavior", "task_id": "4b366888d6184231e49210e82bf33eb0fd9b974b29d5a46c1bd5858171948e1a", "task_kind": "ooc", "wait_seconds": 0.004083500010892749, "world_seed": 10152}].

Total job wall time 2.13s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_scripted_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_scripted_dev.log`.

### 2026-10-12 02:02:05 UTC — corrected_teacher_lookahead

Status: **complete**. Finished 2026-10-12 02:02:09 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python scripts/train_nonllm.py bc --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_0004.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/ooc_lookahead_native011/teacher.jsonl --dev-teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl --phase ooc --device mps --inference-device cpu --threads 2 --batch-size 256 --learning-rate 0.0003 --seed 11 --updates 20 --seconds 30 --metric-interval 1 --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_lookahead.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_lookahead.metrics.json
```

Training: 20 updates, 5,120 examples, 0.42s; 12,149 examples/s; mean loss 6.9067.

Development imitation: {"accuracy": 0.6857142857142857, "examples": 1505, "loss": 1.183816873037142, "menu_accuracy": {"choice": 0.5469348659003831, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.6857142857142857}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.5469348659003831, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 3.37s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_lookahead.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_lookahead.log`.

### 2026-10-12 02:02:09 UTC — corrected_teacher_lookahead_dev

Status: **complete**. Finished 2026-10-12 02:02:12 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/ooc_teacher20_lookahead.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_lookahead.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_lookahead_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 16, 'act_clear': 16}; 1,856 valid transitions.

Mean terminal/cutoff floor 16.19; mean fight HP cost 40.12.

Total job wall time 2.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_lookahead_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/corrected_teacher_lookahead_dev.log`.

### 2026-10-12 02:12:54 UTC — joint_start_shared_bc_dev

Status: **complete**. Finished 2026-10-12 02:13:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_shared_bc_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_shared_bc_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 25, 'act_clear': 7}; 6,554 valid transitions.

Mean terminal/cutoff floor 14.69; mean fight HP cost 62.72.

Total job wall time 5.90s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_shared_bc_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_shared_bc_dev.log`.

### 2026-10-12 02:13:04 UTC — joint_start_combat_dev

Status: **complete**. Finished 2026-10-12 02:13:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_combat_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_combat_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 31}; 5,168 valid transitions.

Mean terminal/cutoff floor 13.06; mean fight HP cost 80.00.

Fault evidence: [{"action_trace_sha256": "f852bc81b6a3c298fc70c63d1fb6dbb3f1bbe449e2e56d9b8ac41966b41c1757", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 91, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "bb464b83be4b18258530d9a59ecde99129927fd4ea5e7784c78cd29c77eda65a", "task_kind": "run", "wait_seconds": 0.0037908749654889107, "world_seed": 10062}].

Total job wall time 5.10s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_combat_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_combat_dev.log`.

### 2026-10-12 02:13:14 UTC — joint_start_ooc_dev

Status: **complete**. Finished 2026-10-12 02:13:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_ooc_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_ooc_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 29, 'act_clear': 3}; 3,237 valid transitions.

Mean terminal/cutoff floor 8.28; mean fight HP cost 72.50.

Total job wall time 3.25s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_ooc_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_ooc_dev.log`.

### 2026-10-12 02:13:20 UTC — joint_start_composite_dev

Status: **complete**. Finished 2026-10-12 02:13:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_composite_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_composite_dev.metrics.json --combat-checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --ooc-checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/kl_guard_paired/before_update.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --world-start 10000 --split dev --task-kind run --max-act 1 --horizon 2048 --simulations 200 --seconds 120 --fast-observations --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 21, 'defeat': 11}; 5,763 valid transitions.

Mean terminal/cutoff floor 16.38; mean fight HP cost 26.50.

Total job wall time 5.67s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_composite_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_start_composite_dev.log`.

### 2026-10-12 02:13:30 UTC — joint_pilot_single

Status: **complete**. Finished 2026-10-12 02:15:05 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --world-start 30000 --episodes 64 --horizon 2048 --max-act 1 --seed 11 --deferred-bootstrap --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --fast-replay --warm-optimizer
```

Live collection: 11 rounds, 699 non-fault episodes, 5 surfaced faults, 0 truncations; outcomes {'defeat': 665, 'act_clear': 34}; 103,263 valid transitions.

Mean terminal/cutoff floor 11.16; mean fight HP cost 75.84.

PPO: 384 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 1.8644818749744445, "preparation_seconds": 1.754834207938984, "rejected_kl": null, "rejected_minibatches": 0, "samples": 9799, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "6c45b842e6f9946cddfdcddc1b5890c506920815146a1ad9e27bfe37eb979eb8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 78, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "29836c9a42a21087668942c2e4a7667b07c4b340df40169429a5162ccdf8a35f", "task_kind": "run", "wait_seconds": 0.00394237507134676, "world_seed": 30003}, {"action_trace_sha256": "4b05802aea735ca575433cfbbd5a56c21b89e5e0c3af81e4b01237dca2e46936", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 12, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "a135eca485175fc061351b99f26cd5790eed48b48a0d858e2e8af2657a22e837", "task_kind": "run", "wait_seconds": 0.0037112499121576548, "world_seed": 30007}, {"action_trace_sha256": "5175bbf77094a3afaf44c721f3c470a407f90e7e667a59dd6032b317c7d75388", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 53, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "7587f69a03c2e0ebaaefcf4505d13c6dbb35eb9074477445cee7956eb93018dd", "task_kind": "run", "wait_seconds": 0.002916709054261446, "world_seed": 30023}, {"action_trace_sha256": "42814ebb8ee6095af08884f551c7d7adf73e4b5f2b58fd39f2b4c67ed03fe149", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 11, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "a135eca485175fc061351b99f26cd5790eed48b48a0d858e2e8af2657a22e837", "task_kind": "run", "wait_seconds": 0.0028282920829951763, "world_seed": 30007}, {"action_trace_sha256": "ac66f129fd6ad9e876a16d1606e9b0fd42f9f63e6486989760ec8d2d53bb47ac", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 59, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "7587f69a03c2e0ebaaefcf4505d13c6dbb35eb9074477445cee7956eb93018dd", "task_kind": "run", "wait_seconds": 0.0025120831560343504, "world_seed": 30023}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single.log`.

### 2026-10-12 02:15:05 UTC — joint_pilot_single_dev

Status: **complete**. Finished 2026-10-12 02:15:15 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 2048 --max-act 1 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 26, 'act_clear': 6}; 5,519 valid transitions.

Mean terminal/cutoff floor 14.50; mean fight HP cost 64.38.

Total job wall time 5.17s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_single_dev.log`.

### 2026-10-12 02:15:16 UTC — joint_pilot_composite

Status: **complete**. Finished 2026-10-12 02:16:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_initial_composite_5a0cc4ad74cb6b2c.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --world-start 30000 --episodes 64 --horizon 2048 --max-act 1 --seed 11 --deferred-bootstrap --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 613 non-fault episodes, 1 surfaced faults, 8 truncations; outcomes {'defeat': 339, 'act_clear': 266, 'GameOutcome.UNDECIDED': 8}; 102,045 valid transitions.

Mean terminal/cutoff floor 15.11; mean fight HP cost 43.46.

PPO: 203 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 1.2764558750204742, "preparation_seconds": 1.1682410000357777, "rejected_kl": null, "rejected_minibatches": 0, "samples": 5872, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "52417000c5692bd0a4115070509b8fa2ca807c25d4c8ac8fe06345e6c836e103", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 109, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3fb7284b4450ea2abd6a13e656fea019d7ad58f1463531ce5e6c114175867e1a", "task_kind": "run", "wait_seconds": 0.004564874805510044, "world_seed": 30047}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 91.51s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite.log`.

### 2026-10-12 02:16:51 UTC — joint_pilot_composite_dev

Status: **complete**. Finished 2026-10-12 02:17:01 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 2048 --max-act 1 --fast-replay
```

Live collection: 1 rounds, 30 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'act_clear': 20, 'defeat': 10}; 4,983 valid transitions.

Mean terminal/cutoff floor 15.87; mean fight HP cost 26.17.

Fault evidence: [{"action_trace_sha256": "7c8567c7a724221feebe3f3c8c3772c83d4f511795d272644bb191ab9a7013b8", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 122, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "27a7d0e8fbcf1de7c2e5979511bc771fbb364b4700910e4e73756cefb5cf1582", "task_kind": "run", "wait_seconds": 0.002512624952942133, "world_seed": 10221}, {"action_trace_sha256": "d77d847389d9ef712812c6107afd2530796bb4824b7c76b7fa2ed7c2e50e4b3a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 155, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c75da13f24c2f463915fa5553f8b2695be5833478a2af6a38daf84936ce8adc6", "task_kind": "run", "wait_seconds": 0.0032306669745594263, "world_seed": 10218}].

Total job wall time 6.19s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_pilot_composite_dev.log`.

### 2026-10-12 02:17:01 UTC — joint_act1_initial_dev

Status: **complete**. Finished 2026-10-12 02:17:11 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_initial_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_initial_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_initial_composite_5a0cc4ad74cb6b2c.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 19, 'act_clear': 29}; 8,451 valid transitions.

Mean terminal/cutoff floor 16.08; mean fight HP cost 30.46.

Total job wall time 8.11s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_initial_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_initial_dev.log`.

### 2026-10-12 02:17:11 UTC — joint_act1_0000

Status: **complete**. Finished 2026-10-12 02:20:17 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_initial_composite_5a0cc4ad74cb6b2c.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 11 --world-start 30000 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 11 rounds, 1372 non-fault episodes, 2 surfaced faults, 8 truncations; outcomes {'defeat': 768, 'act_clear': 596, 'GameOutcome.UNDECIDED': 8}; 222,901 valid transitions.

Mean terminal/cutoff floor 14.92; mean fight HP cost 44.16.

PPO: 252 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.940184541977942, "preparation_seconds": 2.8283359589986503, "rejected_kl": null, "rejected_minibatches": 0, "samples": 15176, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "43b3231776a0a8e9b0d010341feace423e83218974751c69a50dde4ad8457e0f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 87, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "a4d7f397719f8fc7a70a00bcab43bf751ae1549c1c4bbeaa7572029221cb7abd", "task_kind": "run", "wait_seconds": 0.0030772918835282326, "world_seed": 30055}, {"action_trace_sha256": "e5251f64b30649934b722932eb41f01a5831ebdafe964224ab8688d629d93472", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 73, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "f2c158e51913245c7174118a42abf83fe86343bf491d25a238f62e4a5c5955fb", "task_kind": "run", "wait_seconds": 0.005190500058233738, "world_seed": 30035}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.20s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.log`.

### 2026-10-12 02:20:17 UTC — joint_act1_0000_dev

Status: **complete**. Finished 2026-10-12 02:20:27 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 47 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 21, 'act_clear': 26}; 7,940 valid transitions.

Mean terminal/cutoff floor 15.87; mean fight HP cost 35.30.

Fault evidence: [{"action_trace_sha256": "4efaaadf74be1120764b32fed82303cb75999ffb5a8913085cd63855721e74da", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 165, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c75da13f24c2f463915fa5553f8b2695be5833478a2af6a38daf84936ce8adc6", "task_kind": "run", "wait_seconds": 0.0036680828779935837, "world_seed": 10218}].

Total job wall time 8.45s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000_dev.log`.

### 2026-10-12 02:20:27 UTC — joint_act1_0001

Status: **complete**. Finished 2026-10-12 02:23:32 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0000.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 12 --world-start 30128 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1274 non-fault episodes, 6 surfaced faults, 0 truncations; outcomes {'act_clear': 650, 'defeat': 624}; 216,031 valid transitions.

Mean terminal/cutoff floor 15.75; mean fight HP cost 37.75.

PPO: 282 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 4.2701794588938355, "preparation_seconds": 4.1521719170268625, "rejected_kl": null, "rejected_minibatches": 0, "samples": 22086, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "b355949a38a427ae52a984cffea62f300fba4a8420a00e7f39075f9c0e8eba22", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 139, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "af29779ba9601a74140ef7fa7981a6acf9fb0d13730303567303e904f0c56bad", "task_kind": "run", "wait_seconds": 0.0035754998680204153, "world_seed": 30176}, {"action_trace_sha256": "2d2dfa35dca60c190eb3c8277ab41ffcc1e8fd2d9b1a5d4c9c4b0a291a41bcd0", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 99, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "af29779ba9601a74140ef7fa7981a6acf9fb0d13730303567303e904f0c56bad", "task_kind": "run", "wait_seconds": 0.0028210829477757215, "world_seed": 30176}, {"action_trace_sha256": "d95ff5e1d388498025288d5d7721c5c705d54a45dbf79b8976b5f47a304f9628", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 101, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "af29779ba9601a74140ef7fa7981a6acf9fb0d13730303567303e904f0c56bad", "task_kind": "run", "wait_seconds": 0.0033757081255316734, "world_seed": 30176}, {"action_trace_sha256": "3493747efdbf3542398d5204fdd19b45527ca6413fb6fad3575c4f17a974ff1c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 138, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "af29779ba9601a74140ef7fa7981a6acf9fb0d13730303567303e904f0c56bad", "task_kind": "run", "wait_seconds": 0.0038454579189419746, "world_seed": 30176}, {"action_trace_sha256": "068b532c9f153e851c50c99a6105f4a6db64d603cefd7a202a2dd7274c2df673", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 101, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "af29779ba9601a74140ef7fa7981a6acf9fb0d13730303567303e904f0c56bad", "task_kind": "run", "wait_seconds": 0.004104250110685825, "world_seed": 30176}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.31s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.log`.

### 2026-10-12 02:23:32 UTC — joint_act1_0001_dev

Status: **complete**. Finished 2026-10-12 02:23:42 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 34, 'defeat': 14}; 8,043 valid transitions.

Mean terminal/cutoff floor 16.35; mean fight HP cost 22.35.

Total job wall time 7.97s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001_dev.log`.

### 2026-10-12 02:23:42 UTC — joint_act1_0002

Status: **complete**. Finished 2026-10-12 02:26:48 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 13 --world-start 30256 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1239 non-fault episodes, 4 surfaced faults, 8 truncations; outcomes {'defeat': 487, 'act_clear': 744, 'GameOutcome.UNDECIDED': 8}; 207,730 valid transitions.

Mean terminal/cutoff floor 15.90; mean fight HP cost 30.40.

PPO: 364 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.893255334114656, "preparation_seconds": 2.780637416988611, "rejected_kl": null, "rejected_minibatches": 0, "samples": 14640, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "f56b8a5aa12c2d2159cfb2eeafbd2f163a0ad8fe0b76254966ad4075b14be313", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b3160ba50f445fdb9f82ed5f4790671e62018dcd52aeb1b94e1877c594d0b2bb", "task_kind": "run", "wait_seconds": 0.0025542499497532845, "world_seed": 30318}, {"action_trace_sha256": "a846b7b0efc081762db69d6c0c38ef1089d5465ce4e4dabf11c889d4dc46afaf", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 31, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e8cb201eb2f0916487d79bdd3f8e44e33d7a4eec34d44a8d66eee56c8f65be0d", "task_kind": "run", "wait_seconds": 0.0025129159912467003, "world_seed": 30364}, {"action_trace_sha256": "090627ce368494646adb077e844123c792908739c3b9a96289dd9760a86e6866", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 163, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "da98e7242bee6e7fbd5261211cb3c35e58beead0c6f7284163e5d39d22c61038", "task_kind": "run", "wait_seconds": 0.004527916898950934, "world_seed": 30275}, {"action_trace_sha256": "b15e82b748ae607a290ac0f41ea0fcdb19b9a6072549581006a43934218bf3ff", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 55, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b3160ba50f445fdb9f82ed5f4790671e62018dcd52aeb1b94e1877c594d0b2bb", "task_kind": "run", "wait_seconds": 0.0033137090504169464, "world_seed": 30318}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.14s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.log`.

### 2026-10-12 02:26:48 UTC — joint_act1_0002_dev

Status: **complete**. Finished 2026-10-12 02:26:58 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 11, 'act_clear': 37}; 7,863 valid transitions.

Mean terminal/cutoff floor 16.52; mean fight HP cost 17.54.

Total job wall time 7.92s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002_dev.log`.

### 2026-10-12 02:26:58 UTC — joint_act1_0003

Status: **complete**. Finished 2026-10-12 02:30:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0002.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 14 --world-start 30384 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1277 non-fault episodes, 3 surfaced faults, 0 truncations; outcomes {'act_clear': 797, 'defeat': 480}; 211,678 valid transitions.

Mean terminal/cutoff floor 16.29; mean fight HP cost 28.68.

PPO: 298 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "approx_kl": 0.004178633389528841, "clip_fraction": 0.08955199333528678, "early_stop": true, "elapsed_seconds": 5.539216874865815, "entropy": 0.4918726359804471, "policy_loss": -0.0005160573831138512, "preparation_seconds": 3.8697148750070482, "ratio_mean": 1.000245417157809, "rejected_kl": null, "rejected_minibatches": 0, "samples": 20856, "steps": 24, "value_loss": 0.033494796603918076}.

Fault evidence: [{"action_trace_sha256": "8f4d7241dc7447781fcfce150b74185a1f175471b054aee4d587e689c5ca966b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 87, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40357e71c2e573f7f2b339ed5a1a8ae02c07ee0bbad99c986db42df46a284ec0", "task_kind": "run", "wait_seconds": 0.0024658329784870148, "world_seed": 30481}, {"action_trace_sha256": "4141e3cff0bc6c286b5b68e3e34b5b32713844d80cb33a8f86b96295ec7150be", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 78, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40357e71c2e573f7f2b339ed5a1a8ae02c07ee0bbad99c986db42df46a284ec0", "task_kind": "run", "wait_seconds": 0.003485166933387518, "world_seed": 30481}, {"action_trace_sha256": "ea2ec84b9f94d16144047073cdf4debb6068e85349a5d35466d2478ba5a0d1d3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 132, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "6cafdb5659855693760db9652712a536dcd62e61ccfcc39507ee546f7b42545c", "task_kind": "run", "wait_seconds": 0.0039604168850928545, "world_seed": 30407}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.log`.

### 2026-10-12 02:30:04 UTC — joint_act1_0003_dev

Status: **complete**. Finished 2026-10-12 02:30:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 47 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 11, 'act_clear': 36}; 7,837 valid transitions.

Mean terminal/cutoff floor 16.60; mean fight HP cost 17.53.

Fault evidence: [{"action_trace_sha256": "8545ba7c69ccb4120b00696567dac5ec898270b332398393164a85ea8cd7306d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 99, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c75da13f24c2f463915fa5553f8b2695be5833478a2af6a38daf84936ce8adc6", "task_kind": "run", "wait_seconds": 0.00299970805644989, "world_seed": 10218}].

Total job wall time 8.19s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003_dev.log`.

### 2026-10-12 02:30:14 UTC — joint_act1_0004

Status: **complete**. Finished 2026-10-12 02:33:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0003.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 15 --world-start 30512 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1222 non-fault episodes, 7 surfaced faults, 8 truncations; outcomes {'act_clear': 764, 'defeat': 450, 'GameOutcome.UNDECIDED': 8}; 197,642 valid transitions.

Mean terminal/cutoff floor 16.31; mean fight HP cost 28.39.

PPO: 472 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.3693902909290045, "preparation_seconds": 2.2564555830322206, "rejected_kl": null, "rejected_minibatches": 0, "samples": 11675, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3dff709fbaea6d5672ea3f145fa15711d10c72da9fb4063e6062b71ea8a1fcd8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 128, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e0d62a2d49ce4a6be803e6fd12dd2dec084ccf2d0885e501457375e18a8b3284", "task_kind": "run", "wait_seconds": 0.003249750006943941, "world_seed": 30675}, {"action_trace_sha256": "56a13beae07d88e4a2f30b04726f34d7f68636477a11ae387bad71ad6d3d3b89", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 88, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "845310267a3f2b10231649f36a915b82adbbeb7f55e02619b86adfd57de85f7f", "task_kind": "run", "wait_seconds": 0.0029030409641563892, "world_seed": 30548}, {"action_trace_sha256": "df031e4fcc75a938f1a6c51a75302d08fdcea75a624043c2ec96d41ca10408aa", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 51, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b4fd8a235e4fc7ebbe19ed8f874b9c43203d1dde5d41901a79b65866ce21f253", "task_kind": "run", "wait_seconds": 0.0022071669809520245, "world_seed": 30666}, {"action_trace_sha256": "92ec1496cab9313ce8f06bec7776a93996101ce661fd4bfe741409bbb7ad5b29", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 134, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e0d62a2d49ce4a6be803e6fd12dd2dec084ccf2d0885e501457375e18a8b3284", "task_kind": "run", "wait_seconds": 0.0030473750084638596, "world_seed": 30675}, {"action_trace_sha256": "d5bacb7fd8c005a3fba103e42905b1b1c37076f25e62813136257545ec9216c2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 67, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "6dba5568e704adc0cca57c41b4838726e6d4064bec716fd73ccfab5b0e3f154a", "task_kind": "run", "wait_seconds": 0.0036574159748852253, "world_seed": 30674}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 182.64s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.log`.

### 2026-10-12 02:33:19 UTC — joint_act1_0004_dev

Status: **complete**. Finished 2026-10-12 02:33:29 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 37, 'defeat': 11}; 7,846 valid transitions.

Mean terminal/cutoff floor 16.56; mean fight HP cost 17.77.

Total job wall time 7.92s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004_dev.log`.

### 2026-10-12 02:33:29 UTC — joint_act1_0005

Status: **complete**. Finished 2026-10-12 02:36:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0004.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 16 --world-start 30640 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1272 non-fault episodes, 8 surfaced faults, 0 truncations; outcomes {'defeat': 483, 'act_clear': 789}; 208,822 valid transitions.

Mean terminal/cutoff floor 16.16; mean fight HP cost 28.99.

PPO: 341 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 4.105166542110965, "preparation_seconds": 3.9952366249635816, "rejected_kl": null, "rejected_minibatches": 0, "samples": 21252, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3ab9c8388f05412005f405f2bb39eb06426b7f35b9ce908b1e2dac26cc722b37", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 66, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "6dba5568e704adc0cca57c41b4838726e6d4064bec716fd73ccfab5b0e3f154a", "task_kind": "run", "wait_seconds": 0.0029228751081973314, "world_seed": 30674}, {"action_trace_sha256": "863f332953ed7ad14455b7cf2985be2e3610d99feeafaef512d8303e089eb644", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 119, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b2c818299245cb2f93894abb81f51067f7675ba5b374b9c63a2b8781fde17bd4", "task_kind": "run", "wait_seconds": 0.004048749804496765, "world_seed": 30793}, {"action_trace_sha256": "4d9c17ffa5674bf620b3c376b7ecefdb12354e732f70dad4d5a0d44fafa33f3b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 44, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8327ace355329cbd69ecaff7d6b5ba86ee38c9c8b324a70db1cec0dd762318a3", "task_kind": "run", "wait_seconds": 0.002573166973888874, "world_seed": 30772}, {"action_trace_sha256": "42cc178894acc79e1f140f3721970f436624f908f4034d396a477334b81ffa13", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 134, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e0d62a2d49ce4a6be803e6fd12dd2dec084ccf2d0885e501457375e18a8b3284", "task_kind": "run", "wait_seconds": 0.004502458032220602, "world_seed": 30675}, {"action_trace_sha256": "6bd59ae73504c15aff377bbdd69b37ed94bb8e1887943882d6d34e433aa280db", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 79, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c1ceffd0080cb237f91fe2773167d5d50eb19bcee316bf2acad81158335051f9", "task_kind": "run", "wait_seconds": 0.003559958888217807, "world_seed": 30678}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.94s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.log`.

### 2026-10-12 02:36:35 UTC — joint_act1_0005_dev

Status: **complete**. Finished 2026-10-12 02:36:45 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 47 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 11, 'act_clear': 36}; 7,931 valid transitions.

Mean terminal/cutoff floor 16.47; mean fight HP cost 17.91.

Fault evidence: [{"action_trace_sha256": "fdf756e794f419026cf35df5676d96d1cbb19c7b33df74d739dc3fc6a895d166", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 90, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c75da13f24c2f463915fa5553f8b2695be5833478a2af6a38daf84936ce8adc6", "task_kind": "run", "wait_seconds": 0.002541415859013796, "world_seed": 10218}].

Total job wall time 8.22s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005_dev.log`.

### 2026-10-12 02:36:45 UTC — joint_act1_0006

Status: **complete**. Finished 2026-10-12 02:39:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0005.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 17 --world-start 30768 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1265 non-fault episodes, 15 surfaced faults, 0 truncations; outcomes {'act_clear': 894, 'defeat': 371}; 210,650 valid transitions.

Mean terminal/cutoff floor 16.53; mean fight HP cost 22.52.

PPO: 245 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 3.978477083146572, "preparation_seconds": 3.869106125086546, "rejected_kl": null, "rejected_minibatches": 0, "samples": 20807, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "dc418844f06378a5e3dd80bf6528ec64c17bf605fedb6b556b5b078639d7d993", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 103, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "12a0b0f937c68abc8bdb4062c902478e39f4d60f06c33956fb8f05fa0a0eadc4", "task_kind": "run", "wait_seconds": 0.004597584018483758, "world_seed": 30831}, {"action_trace_sha256": "7302cf8407ee3f5f9482dce49ef60fad7ddd31d8ef3056e82b2cfe534b1825c1", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 162, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "277ac1978e425e065e36814677253d830df90f60f2c90248f88668b42af3fc12", "task_kind": "run", "wait_seconds": 0.003167124930769205, "world_seed": 30854}, {"action_trace_sha256": "79dbbcac51a3d3d1eeff6b095b98db74b9a95f8318c5c92f5919ee30ab85b4ba", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 120, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2c7c9b4328967549bfd716c80db6cc3bc33325eae6ed3f833e9425f075f7b62a", "task_kind": "run", "wait_seconds": 0.003104166127741337, "world_seed": 30787}, {"action_trace_sha256": "f9593750ff18caf7ef1f98603dbd8bf42f5773d23e90d1013ab1d43fcea1ac68", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 120, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "f7bfb7e581452e62398a9e2b8f2138c688b4f1e91d1ea9985cc1a031b42cfa75", "task_kind": "run", "wait_seconds": 0.004068874986842275, "world_seed": 30770}, {"action_trace_sha256": "dff70ceacb41b0c0ae01a539784700054c36cb5c73d4af9d678108412fc130eb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 127, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "7cfa02661a24ae3bdd5b4a9f31328e8117f81b945026ff5806c298ac10ae8072", "task_kind": "run", "wait_seconds": 0.002541291993111372, "world_seed": 30855}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.10s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.log`.

### 2026-10-12 02:39:51 UTC — joint_act1_0006_dev

Status: **complete**. Finished 2026-10-12 02:40:01 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 13, 'act_clear': 35}; 8,096 valid transitions.

Mean terminal/cutoff floor 16.40; mean fight HP cost 21.42.

Total job wall time 7.88s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006_dev.log`.

### 2026-10-12 02:40:01 UTC — joint_act1_0007

Status: **complete**. Finished 2026-10-12 02:43:06 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0006.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 18 --world-start 30896 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 9 rounds, 1149 non-fault episodes, 3 surfaced faults, 0 truncations; outcomes {'act_clear': 797, 'defeat': 352}; 192,588 valid transitions.

Mean terminal/cutoff floor 16.48; mean fight HP cost 23.30.

PPO: 584 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 4.0179597090464085, "preparation_seconds": 3.909207792021334, "rejected_kl": null, "rejected_minibatches": 0, "samples": 21253, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "ac33466f6c73c7610e9666b9a6a20db60758a762c78cca74200a873e29f0dc87", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 141, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "425342f2606dc2ee16322d49bb3d509c8a46fdcf5a1d8dba2ec14548972d9d84", "task_kind": "run", "wait_seconds": 0.0038063330575823784, "world_seed": 31009}, {"action_trace_sha256": "713a4b77d54b87d609caa50ef1a864ece51ee7d76ea21c39f61443ef2c2bdee6", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "81f912e7a731e31de28a61c67a4c16727e9c4842636c9ea4bf853ff9e9769bb4", "task_kind": "run", "wait_seconds": 0.003981791902333498, "world_seed": 30964}, {"action_trace_sha256": "87f8f7c0be52362d440fb6da2199e15ecb2cd2083a37adae532cd8e4ac7f0353", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "81f912e7a731e31de28a61c67a4c16727e9c4842636c9ea4bf853ff9e9769bb4", "task_kind": "run", "wait_seconds": 0.0033839999232441187, "world_seed": 30964}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.93s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.log`.

### 2026-10-12 02:43:06 UTC — joint_act1_0007_dev

Status: **complete**. Finished 2026-10-12 02:43:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 38, 'defeat': 10}; 7,924 valid transitions.

Mean terminal/cutoff floor 16.65; mean fight HP cost 16.31.

Total job wall time 7.68s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007_dev.log`.

### 2026-10-12 02:43:16 UTC — joint_act1_0008

Status: **complete**. Finished 2026-10-12 02:46:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 19 --world-start 31024 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-12 02:51:43 UTC — joint_act1_0008

Status: **complete**. Finished 2026-10-12 02:54:48 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 19 --world-start 31024 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 9 rounds, 1073 non-fault episodes, 20 surfaced faults, 8 truncations; outcomes {'defeat': 304, 'act_clear': 761, 'GameOutcome.UNDECIDED': 8}; 181,551 valid transitions.

Mean terminal/cutoff floor 16.57; mean fight HP cost 21.80.

PPO: 569 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.2207554581109434, "preparation_seconds": 2.0746130000334233, "rejected_kl": null, "rejected_minibatches": 0, "samples": 11112, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3c7b6e4f37919027759a199aad07deb49d0a555540d28238c48e52e917e110c0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "902bf60326ed201544b12356ea3083f48382ed038ac054827f4f1fff87b3f824", "task_kind": "run", "wait_seconds": 0.0020057091023772955, "world_seed": 31133}, {"action_trace_sha256": "ec7161916c0b4c233a8c8a39515521f77f1740300dbe25d11ef0e7018f21d65b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 159, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1b262244c39391ed1f71be2edd21317e0e51455d3b99b7de0576e33faac4ca05", "task_kind": "run", "wait_seconds": 0.0028225418645888567, "world_seed": 31140}, {"action_trace_sha256": "3f38436647fd7b94022acbdffac4ea66babb2b12af3ed20049aa3ca846cab233", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 138, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "dd586096d82805d610210b1e5e78d6a64dd406f3d1dd1415b86d40bac6920ac2", "task_kind": "run", "wait_seconds": 0.0025201248936355114, "world_seed": 31100}, {"action_trace_sha256": "beb0db4d30984a294eba63983cb260263daf3a5479c7b2293f5ed57370baec34", "attempted_action_index": 2, "error": "SimulatorError", "excluded_transitions": 142, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "0dba94db85d23c98ebbbf455e645564a652322e991f963e1e0c747652aee985f", "task_kind": "run", "wait_seconds": 0.0032450410071760416, "world_seed": 31132}, {"action_trace_sha256": "8843587f31509046e51dc88e9ed3817023cbff369dcee30677a4daa81c65f87e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 7, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "d37901b0de028470f614fe8c3180656a3523c4b3b4f3c37a2eac8d6e72f40cdb", "task_kind": "run", "wait_seconds": 0.002229624893516302, "world_seed": 31064}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 182.48s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.log`.

### 2026-10-12 02:54:49 UTC — joint_act1_0008_dev

Status: **complete**. Finished 2026-10-12 02:54:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 37, 'defeat': 11}; 8,300 valid transitions.

Mean terminal/cutoff floor 16.54; mean fight HP cost 17.73.

Total job wall time 8.15s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008_dev.log`.

### 2026-10-12 02:54:59 UTC — joint_act1_0009

Status: **complete**. Finished 2026-10-12 02:58:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 20 --world-start 31152 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1220 non-fault episodes, 5 surfaced faults, 7 truncations; outcomes {'defeat': 329, 'act_clear': 884, 'GameOutcome.UNDECIDED': 7}; 207,748 valid transitions.

Mean terminal/cutoff floor 16.51; mean fight HP cost 21.11.

PPO: 443 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.3323920420370996, "preparation_seconds": 2.2209918750450015, "rejected_kl": null, "rejected_minibatches": 0, "samples": 11908, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "2058217b649d3f4b1cb7b7a5d7d33cf2bb6a2f721961e86674a1e90fbac8dc9d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 58, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "0ac4971ed1772f3cba18cc4e3a50744af96feb50cb3754f5e663c02566e423c6", "task_kind": "run", "wait_seconds": 0.002323665888980031, "world_seed": 31213}, {"action_trace_sha256": "9b05e08ca5347cf62e84e384c740f8e184816363a5825226b0aa4e01750ffb2d", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 193, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "88fd7049d21cae376347f2a6014ed5d5edf80893f2bc9480b2ac07dcba1cc751", "task_kind": "run", "wait_seconds": 0.003509000176563859, "world_seed": 31157}, {"action_trace_sha256": "6c95d15b51725b51eafa834d366ba391fa997e62e6e2c019c49a37e7e21f7b5b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 94, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "797d2b60ff1c5dfed3a60e9993757a12ae673405ed6049b095eaea9ce90d2063", "task_kind": "run", "wait_seconds": 0.002900124993175268, "world_seed": 31282}, {"action_trace_sha256": "7cfdd57b614d1f4e03c28a553d1994a8af75b0288359a4c6792520d6f9c8b0ae", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 90, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "797d2b60ff1c5dfed3a60e9993757a12ae673405ed6049b095eaea9ce90d2063", "task_kind": "run", "wait_seconds": 0.00214966619387269, "world_seed": 31282}, {"action_trace_sha256": "90b5bee78483e15620de748257be9ae2f3352adf9d7eee0eb940827b87caa936", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 99, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "797d2b60ff1c5dfed3a60e9993757a12ae673405ed6049b095eaea9ce90d2063", "task_kind": "run", "wait_seconds": 0.0025195840280503035, "world_seed": 31282}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 182.63s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.log`.

### 2026-10-12 02:58:04 UTC — joint_act1_0009_dev

Status: **complete**. Finished 2026-10-12 02:58:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 13, 'act_clear': 35}; 8,235 valid transitions.

Mean terminal/cutoff floor 16.62; mean fight HP cost 21.56.

Total job wall time 7.81s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009_dev.log`.

### 2026-10-12 02:58:14 UTC — joint_act1_0010

Status: **complete**. Finished 2026-10-12 03:01:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0009.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 21 --world-start 31280 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1239 non-fault episodes, 8 surfaced faults, 8 truncations; outcomes {'act_clear': 859, 'defeat': 372, 'GameOutcome.UNDECIDED': 8}; 211,994 valid transitions.

Mean terminal/cutoff floor 16.46; mean fight HP cost 23.61.

PPO: 341 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 2.9903121672105044, "preparation_seconds": 2.882057708222419, "rejected_kl": null, "rejected_minibatches": 0, "samples": 15380, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "bd8b343efe9a54bde1eb60255361ea8450e393a7a8e79785612f51875cf8fc5f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "51525736614bb78f7c050d1d4a288265371cede83a4a34d2f6b556e19262e665", "task_kind": "run", "wait_seconds": 0.002277666935697198, "world_seed": 31418}, {"action_trace_sha256": "0420938c1bbfa0ce94f50de72cb517d245b220588c4e21b1d08fd062d0bf0bd2", "attempted_action_index": 2, "error": "SimulatorError", "excluded_transitions": 141, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b021cc661d9045f72f238a083fa7d38d6e35c4ccdd8067eec2da71fe10d7d10f", "task_kind": "run", "wait_seconds": 0.004015625221654773, "world_seed": 31396}, {"action_trace_sha256": "1abeea4d79f42a02d38ab02109429200828a495692687cee7c437b2d24312ae8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 97, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "56e6d08357c9c9af61d42d605d57ff55c629dd3e9ab0fdf4710890e61bc57a7a", "task_kind": "run", "wait_seconds": 0.002587374998256564, "world_seed": 31311}, {"action_trace_sha256": "9b5ec2a9590d40379c7a8275a9517f5374b7cedb3a20e93efd01f7d5af895bf0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 119, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "797d2b60ff1c5dfed3a60e9993757a12ae673405ed6049b095eaea9ce90d2063", "task_kind": "run", "wait_seconds": 0.003801416140049696, "world_seed": 31282}, {"action_trace_sha256": "2033d37ac879e4429cbedc888c8ee63d09749b531704348a0ff4b68c0c4a2ea5", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 36, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2e820905659cefa6aa41a9c6f3a07ab72fddf7135ecf41d38e98e7dbd3c6620f", "task_kind": "run", "wait_seconds": 0.0028909160755574703, "world_seed": 31412}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.30s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.log`.

### 2026-10-12 03:01:20 UTC — joint_act1_0010_dev

Status: **complete**. Finished 2026-10-12 03:01:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 37, 'defeat': 11}; 8,142 valid transitions.

Mean terminal/cutoff floor 16.77; mean fight HP cost 18.12.

Total job wall time 7.79s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010_dev.log`.

### 2026-10-12 03:01:30 UTC — joint_act1_0011

Status: **complete**. Finished 2026-10-12 03:04:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0010.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 22 --world-start 31408 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1275 non-fault episodes, 5 surfaced faults, 4 truncations; outcomes {'act_clear': 1030, 'defeat': 241, 'GameOutcome.UNDECIDED': 4}; 217,663 valid transitions.

Mean terminal/cutoff floor 16.66; mean fight HP cost 14.82.

PPO: 314 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 3.9329542501363903, "preparation_seconds": 3.82280550012365, "rejected_kl": null, "rejected_minibatches": 0, "samples": 21317, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "ac4ae6ca493604656335103324926421f1afafa32740cda3c523a5122353824b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "51525736614bb78f7c050d1d4a288265371cede83a4a34d2f6b556e19262e665", "task_kind": "run", "wait_seconds": 0.0033681250642985106, "world_seed": 31418}, {"action_trace_sha256": "ab79cf23467b1fe403bab0221f24262c5710220d84ef77366ebbd722aaf08861", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 113, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4adc51ab36a9a8345768a4a270442c177b247f6faae41cb87aded90323d1433d", "task_kind": "run", "wait_seconds": 0.003094749990850687, "world_seed": 31554}, {"action_trace_sha256": "d453ca0026e1dc7dab08376d68b8afe246e17fb081c94ca152b5632a22de27e8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2e820905659cefa6aa41a9c6f3a07ab72fddf7135ecf41d38e98e7dbd3c6620f", "task_kind": "run", "wait_seconds": 0.003271042136475444, "world_seed": 31412}, {"action_trace_sha256": "51ade0d845df14cf2453815e12a2c62dc8bcfdde53046d08d1cf094f676c1c7c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 56, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "51525736614bb78f7c050d1d4a288265371cede83a4a34d2f6b556e19262e665", "task_kind": "run", "wait_seconds": 0.003368500154465437, "world_seed": 31418}, {"action_trace_sha256": "82d5e9259be371bb8d1f3032b06da63f4c8f83b8b4ffcc14a508961ae7811eab", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 52, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "51525736614bb78f7c050d1d4a288265371cede83a4a34d2f6b556e19262e665", "task_kind": "run", "wait_seconds": 0.0034204169642180204, "world_seed": 31418}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 184.17s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.log`.

### 2026-10-12 03:04:36 UTC — joint_act1_0011_dev

Status: **complete**. Finished 2026-10-12 03:04:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 38, 'defeat': 10}; 8,248 valid transitions.

Mean terminal/cutoff floor 16.67; mean fight HP cost 16.62.

Total job wall time 7.85s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011_dev.log`.

### 2026-10-12 03:04:46 UTC — joint_act1_0012

Status: **complete**. Finished 2026-10-12 03:07:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0011.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 23 --world-start 31536 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 11 rounds, 1281 non-fault episodes, 7 surfaced faults, 8 truncations; outcomes {'act_clear': 980, 'defeat': 293, 'GameOutcome.UNDECIDED': 8}; 221,186 valid transitions.

Mean terminal/cutoff floor 16.68; mean fight HP cost 17.89.

PPO: 226 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.3001125000882894, "preparation_seconds": 0.19695283309556544, "rejected_kl": null, "rejected_minibatches": 0, "samples": 920, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "98bbde453f0741dd1d0aa4077052c661cb23c539e068139b5067a58dbf267b30", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 116, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4adc51ab36a9a8345768a4a270442c177b247f6faae41cb87aded90323d1433d", "task_kind": "run", "wait_seconds": 0.0027637919411063194, "world_seed": 31554}, {"action_trace_sha256": "94b9e3768677fc40af31f859a41ae96a3b6005f50bf130ab88a82a7fede97571", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 148, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4adc51ab36a9a8345768a4a270442c177b247f6faae41cb87aded90323d1433d", "task_kind": "run", "wait_seconds": 0.0032432088628411293, "world_seed": 31554}, {"action_trace_sha256": "d36a240a229e96f32dfdfa8eebd5c2402c16ec54020ff230fd97ed544c6810a3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 171, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "27dc01f0b490713f341b1992015c81fa3841b9dc1a6622edd15a0547b19177c1", "task_kind": "run", "wait_seconds": 0.004048791015520692, "world_seed": 31644}, {"action_trace_sha256": "a6c89936655897a0665ff6f9bb1f3439e95f0e8c02e447d62519ec50c526caf2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 34, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1f4f8d3fd81ea4862ac4431d0a79767b340641a7cdbdf7fcd214fa9668b9d40c", "task_kind": "run", "wait_seconds": 0.003763499902561307, "world_seed": 31651}, {"action_trace_sha256": "b46c9ecaef16f12b6b9ab0e76cb94413af94dc3e2ef30c4cd16b969fe4497bc1", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 65, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "53cb8ce1eee6170f2ed7d85360ae41e1f9771d61314efe3608cb0af19b9f10b2", "task_kind": "run", "wait_seconds": 0.0032535840291529894, "world_seed": 31605}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.58s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.log`.

### 2026-10-12 03:07:51 UTC — joint_act1_0012_dev

Status: **complete**. Finished 2026-10-12 03:08:01 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 40, 'defeat': 8}; 8,346 valid transitions.

Mean terminal/cutoff floor 16.73; mean fight HP cost 13.33.

Total job wall time 7.89s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012_dev.log`.

### 2026-10-12 03:08:02 UTC — joint_act1_0013

Status: **complete**. Finished 2026-10-12 03:11:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0012.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 24 --world-start 31664 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1249 non-fault episodes, 1 surfaced faults, 8 truncations; outcomes {'defeat': 243, 'act_clear': 998, 'GameOutcome.UNDECIDED': 8}; 216,515 valid transitions.

Mean terminal/cutoff floor 16.65; mean fight HP cost 14.52.

PPO: 327 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 3.088636958040297, "preparation_seconds": 2.9807411248330027, "rejected_kl": null, "rejected_minibatches": 0, "samples": 16408, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "91d8ae7733b79d9c8c81df91c293b40ce9aae3ae7c5f125dec634c9a16068783", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 32, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3ee03378a8c713d56fd901da3a6081955c6e74e69a1668dbe3ee969bdfe2b27c", "task_kind": "run", "wait_seconds": 0.00321095809340477, "world_seed": 31675}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.33s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.log`.

### 2026-10-12 03:11:07 UTC — joint_act1_0013_dev

Status: **complete**. Finished 2026-10-12 03:11:17 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 37, 'defeat': 11}; 8,209 valid transitions.

Mean terminal/cutoff floor 16.67; mean fight HP cost 17.96.

Total job wall time 7.81s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013_dev.log`.

### 2026-10-12 03:11:17 UTC — joint_act1_0014

Status: **complete**. Finished 2026-10-12 03:14:23 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0013.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 25 --world-start 31792 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1167 non-fault episodes, 9 surfaced faults, 8 truncations; outcomes {'defeat': 264, 'act_clear': 895, 'GameOutcome.UNDECIDED': 8}; 198,443 valid transitions.

Mean terminal/cutoff floor 16.59; mean fight HP cost 17.40.

PPO: 478 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.6797589999623597, "preparation_seconds": 0.5745164591353387, "rejected_kl": null, "rejected_minibatches": 0, "samples": 2911, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "d631768b30ef0dd49557c061f98afa93e88c64a71485885b8021374a028298e9", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 24, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3577388a863d9e7c1e5257c7faa2fbb59897486275ffc48aed365c4c0bbd3025", "task_kind": "run", "wait_seconds": 0.0029331669211387634, "world_seed": 31957}, {"action_trace_sha256": "c59beb26a392038f0aabb20a080679690c91c7e792763e109db3c7f345eda5fe", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 156, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3f2dfe428065349a6e6715b929c6562c999edebf8fc4cb8cc1268c952a5aa1b6", "task_kind": "run", "wait_seconds": 0.0032552501652389765, "world_seed": 31886}, {"action_trace_sha256": "dcea01fd831b80268cd95c56d0427769ee2f01f878adb4f1faacbbc94e3d2795", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 102, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "df5788297f50997bef5fd945dacc886df88bdfbb20bef47483c500cf2017ae14", "task_kind": "run", "wait_seconds": 0.003698958083987236, "world_seed": 31857}, {"action_trace_sha256": "1dec954cf834f952ff5fe58db372254af3cc6a22d69ccfb6ceeae628dfa47023", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 56, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8d721ffd98ec57505e8d341c9af54aedfe17b4ab06b4dad242ef164ca4883bf3", "task_kind": "run", "wait_seconds": 0.0037111248821020126, "world_seed": 31892}, {"action_trace_sha256": "29c6a8552e34bce365a4d4ac7e448914820ffb9d75f3b7cbeb5f0f51e087c970", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 131, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "6e5db3d3c61409de4b83efdbb12fb1af506fd2c6f7f4698827387510e31dd0e9", "task_kind": "run", "wait_seconds": 0.004208833910524845, "world_seed": 31931}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.99s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.log`.

### 2026-10-12 03:14:23 UTC — joint_act1_0014_dev

Status: **complete**. Finished 2026-10-12 03:14:33 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 36, 'defeat': 12}; 8,151 valid transitions.

Mean terminal/cutoff floor 16.71; mean fight HP cost 19.77.

Total job wall time 7.71s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014_dev.log`.

### 2026-10-12 03:14:33 UTC — joint_act1_0015

Status: **complete**. Finished 2026-10-12 03:17:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0014.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 26 --world-start 31920 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1201 non-fault episodes, 14 surfaced faults, 8 truncations; outcomes {'defeat': 290, 'act_clear': 903, 'GameOutcome.UNDECIDED': 8}; 206,273 valid transitions.

Mean terminal/cutoff floor 16.64; mean fight HP cost 18.93.

PPO: 393 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 1.9860903329681605, "preparation_seconds": 1.877601542044431, "rejected_kl": null, "rejected_minibatches": 0, "samples": 10240, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "182b3cbdf7f49f36679e6691c1395b6702442401c1df4861e843d608d7ee717c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 78, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4c4a0d4aa8f9d7af1b2438f764c44f10a25d4d1d54b0a7e2ca9b3305be6222db", "task_kind": "run", "wait_seconds": 0.003272667061537504, "world_seed": 31991}, {"action_trace_sha256": "76367a24506baf42b1dc4714954258ec81d5211e6b5a5bd8f2c72bb79baaa078", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 161, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8977fefb7f8dcc2b17c9ed93c8dd04189cdbd9921ee367d188e5bdf2c6928cc2", "task_kind": "run", "wait_seconds": 0.002876457991078496, "world_seed": 31960}, {"action_trace_sha256": "b542827a8fe1bf9835027b3e15f88ca8ad07d87242b2de5db79f72a19a143f63", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 139, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "6e5db3d3c61409de4b83efdbb12fb1af506fd2c6f7f4698827387510e31dd0e9", "task_kind": "run", "wait_seconds": 0.003277042182162404, "world_seed": 31931}, {"action_trace_sha256": "d6611e67899be6e6b32760cbb48c549a1d6d39cd8be02f9f789fe7645a5e8e7d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 141, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8977fefb7f8dcc2b17c9ed93c8dd04189cdbd9921ee367d188e5bdf2c6928cc2", "task_kind": "run", "wait_seconds": 0.0037191249430179596, "world_seed": 31960}, {"action_trace_sha256": "0c1f35b91ef14327b7efbd6c129654d7f1f8f49a1553f14a2e305f64b3f3b5d8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 69, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4c4a0d4aa8f9d7af1b2438f764c44f10a25d4d1d54b0a7e2ca9b3305be6222db", "task_kind": "run", "wait_seconds": 0.0033663748763501644, "world_seed": 31991}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 182.26s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.log`.

### 2026-10-12 03:17:39 UTC — joint_act1_0015_dev

Status: **complete**. Finished 2026-10-12 03:17:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 11, 'act_clear': 37}; 8,538 valid transitions.

Mean terminal/cutoff floor 16.73; mean fight HP cost 18.17.

Total job wall time 8.08s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015_dev.log`.

### 2026-10-12 03:17:49 UTC — joint_act1_0016

Status: **complete**. Finished 2026-10-12 03:20:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0015.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 27 --world-start 32048 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 10 rounds, 1249 non-fault episodes, 6 surfaced faults, 8 truncations; outcomes {'act_clear': 934, 'defeat': 307, 'GameOutcome.UNDECIDED': 8}; 219,576 valid transitions.

Mean terminal/cutoff floor 16.62; mean fight HP cost 19.18.

PPO: 305 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 3.457583790877834, "preparation_seconds": 3.339325707871467, "rejected_kl": null, "rejected_minibatches": 0, "samples": 17319, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "c394a8ab1b569a705c190024316dd069b4058e26b386dd9aec3641dd302b7a30", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 47, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c67178195cc8ad7a5303416dcd5928c39484ec1d8cfac1f84cbd9a0df06c3902", "task_kind": "run", "wait_seconds": 0.0029899580404162407, "world_seed": 32102}, {"action_trace_sha256": "f6e9ee752a35dbe464a6bc3f74e0a3b6e4365712f1c16ceef707e1ea1813d7a5", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 58, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "9afe2dce568d688cc17ee403764632a13fb33b5ebb8cc0808bdf0ee924d36a1e", "task_kind": "run", "wait_seconds": 0.0036010839976370335, "world_seed": 32134}, {"action_trace_sha256": "8df9e363cc0dc0c174a3910274ab067fe095952a1afda72f273f36c4377d95d2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 45, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c67178195cc8ad7a5303416dcd5928c39484ec1d8cfac1f84cbd9a0df06c3902", "task_kind": "run", "wait_seconds": 0.002568083116784692, "world_seed": 32102}, {"action_trace_sha256": "3e8a46f808e43e786c43569f93fc5b0ebeeece1a5edff3fa889ceb91f1076301", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 104, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8ed3cb557e7ac072b0e6df9b9f2b07c64df0e07aacc5015b9fd65a78e7c6a703", "task_kind": "run", "wait_seconds": 0.0038670001085847616, "world_seed": 32131}, {"action_trace_sha256": "8e77d7f5f661a17cbc469aea7350241d544a29802daa37edb11c09ab20e4cefb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "9afe2dce568d688cc17ee403764632a13fb33b5ebb8cc0808bdf0ee924d36a1e", "task_kind": "run", "wait_seconds": 0.0030887918546795845, "world_seed": 32134}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 183.74s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.log`.

### 2026-10-12 03:20:54 UTC — joint_act1_0016_dev

Status: **complete**. Finished 2026-10-12 03:21:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 38, 'defeat': 10}; 8,278 valid transitions.

Mean terminal/cutoff floor 16.75; mean fight HP cost 16.15.

Total job wall time 7.92s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016_dev.log`.

### 2026-10-12 03:21:04 UTC — joint_act1_0017

Status: **complete**. Finished 2026-10-12 03:24:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0016.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 28 --world-start 32176 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1457 non-fault episodes, 2 surfaced faults, 8 truncations; outcomes {'act_clear': 1097, 'defeat': 352, 'GameOutcome.UNDECIDED': 8}; 263,687 valid transitions.

Mean terminal/cutoff floor 16.58; mean fight HP cost 18.46.

PPO: 470 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.16259620897471905, "preparation_seconds": 0.04587666713632643, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 8448, "samples": 8448, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "09b04db9b93b2989370ad30bcdd891b8794615a7c86f691986203b051e63e5fa", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 89, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "fc995fe2592abab5df5ed7874434b5344880aafbcaeaa95d7d8db9f53cbe11b3", "task_kind": "run", "wait_seconds": 0.002860416891053319, "world_seed": 32289}, {"action_trace_sha256": "af0457737ef63de9148c6eeffc42e110bfa4d120a1ce05be01e18686e6c260a9", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 152, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "0ce05696a93062d9cb574ce1db21114c4d08012fca78dbe95584f59c5e223b32", "task_kind": "run", "wait_seconds": 0.004754625027999282, "world_seed": 32179}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.64s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.log`.

### 2026-10-12 03:24:10 UTC — joint_act1_0017_dev

Status: **complete**. Finished 2026-10-12 03:24:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 38, 'defeat': 10}; 8,591 valid transitions.

Mean terminal/cutoff floor 16.79; mean fight HP cost 15.98.

Total job wall time 8.71s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017_dev.log`.

### 2026-10-12 03:24:20 UTC — joint_act1_0018

Status: **complete**. Finished 2026-10-12 03:27:26 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0017.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 29 --world-start 32304 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1447 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'act_clear': 1129, 'defeat': 310, 'GameOutcome.UNDECIDED': 8}; 258,557 valid transitions.

Mean terminal/cutoff floor 16.53; mean fight HP cost 16.12.

PPO: 579 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.15276666590943933, "preparation_seconds": 0.04544575000181794, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 6200, "samples": 6200, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.64s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.log`.

### 2026-10-12 03:27:26 UTC — joint_act1_0018_dev

Status: **complete**. Finished 2026-10-12 03:27:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 13, 'act_clear': 35}; 8,418 valid transitions.

Mean terminal/cutoff floor 16.73; mean fight HP cost 21.15.

Total job wall time 8.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018_dev.log`.

### 2026-10-12 03:27:36 UTC — joint_act1_0019

Status: **complete**. Finished 2026-10-12 03:30:42 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0018.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 30 --world-start 32432 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1480 non-fault episodes, 11 surfaced faults, 8 truncations; outcomes {'act_clear': 1086, 'defeat': 386, 'GameOutcome.UNDECIDED': 8}; 258,615 valid transitions.

Mean terminal/cutoff floor 16.56; mean fight HP cost 20.14.

PPO: 471 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.16177141689695418, "preparation_seconds": 0.05586620792746544, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 13736, "samples": 13736, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "7d7bafda597f7c4726fbe95f23c8c3ca75e6ec507d4fbbf4d4312fc64aaa9186", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 87, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1956904830de94491c530ee2123a7ab7a23c2055a0b911189aec641c2f16fe9e", "task_kind": "run", "wait_seconds": 0.0027771671302616596, "world_seed": 32542}, {"action_trace_sha256": "30153f3b35b50fd2337ea84af4be3d34c14d1fc1bcf78de11993bca2f937505b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 105, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b6cf14a086ecd1b49c86726a63858f0f6b162c6a7d4c53b940594268d1f9d1aa", "task_kind": "run", "wait_seconds": 0.0024291248992085457, "world_seed": 32459}, {"action_trace_sha256": "73151250e16332b582e4e633b5e4cc00e13bfbee3b15ae7b324423b4e0b5f747", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 138, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b6cf14a086ecd1b49c86726a63858f0f6b162c6a7d4c53b940594268d1f9d1aa", "task_kind": "run", "wait_seconds": 0.0030831668991595507, "world_seed": 32459}, {"action_trace_sha256": "a573d9e7615e073a4ea980634800d5b32e2171c75a781b469cc72bc0ab1d73b3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1956904830de94491c530ee2123a7ab7a23c2055a0b911189aec641c2f16fe9e", "task_kind": "run", "wait_seconds": 0.0034970000851899385, "world_seed": 32542}, {"action_trace_sha256": "3849b8a6dce6bd76180785e000a54fb07096160c678d594ccc548d80afd49b6e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 117, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b6cf14a086ecd1b49c86726a63858f0f6b162c6a7d4c53b940594268d1f9d1aa", "task_kind": "run", "wait_seconds": 0.003453084034845233, "world_seed": 32459}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.43s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.log`.

### 2026-10-12 03:30:42 UTC — joint_act1_0019_dev

Status: **complete**. Finished 2026-10-12 03:30:52 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 42, 'defeat': 6}; 8,200 valid transitions.

Mean terminal/cutoff floor 16.88; mean fight HP cost 9.54.

Total job wall time 8.00s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019_dev.log`.

### 2026-10-12 03:30:52 UTC — joint_act1_0020

Status: **complete**. Finished 2026-10-12 03:33:57 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0019.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 31 --world-start 32560 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1479 non-fault episodes, 14 surfaced faults, 8 truncations; outcomes {'defeat': 400, 'act_clear': 1071, 'GameOutcome.UNDECIDED': 8}; 263,965 valid transitions.

Mean terminal/cutoff floor 16.49; mean fight HP cost 20.83.

PPO: 415 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.18010329082608223, "preparation_seconds": 0.06765545788221061, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 13944, "samples": 13944, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "42165f9598d150f6d100488dea2530efbaef1421cb1b5c1bf6775d4bebdcae80", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 126, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "0bc2af29f7d91517b19db23962d9be56d7426114575d073d1873b54722896558", "task_kind": "run", "wait_seconds": 0.003615542082116008, "world_seed": 32644}, {"action_trace_sha256": "b8f149eedf7a774ad78f68e9fa8f59e8c51ccb398fff372f45efce24c698fd1e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 42, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c82b37a71e15bbf420c29dd1eba03be86fa648560f961b196b6680890412d419", "task_kind": "run", "wait_seconds": 0.003123458009213209, "world_seed": 32710}, {"action_trace_sha256": "273ed37c395cea04ef62c42d2ae7090d1033685283a2e1be550bf27f98fed3d3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 81, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b753a3c659465d44e0e7b0947f8b0f7545360e45c6ae04ddd51293131be9770b", "task_kind": "run", "wait_seconds": 0.003425583941861987, "world_seed": 32653}, {"action_trace_sha256": "f130f20d758d3dc7b06fcf4454e33092a94f6e11d2f32822412701db60e6be6f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 171, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "7f33a8415b9fa05f2b5b6f1e3f08c661ab7c225bf396043ca2770cb7d9e6f10c", "task_kind": "run", "wait_seconds": 0.0036656251177191734, "world_seed": 32699}, {"action_trace_sha256": "f0efd3063b60d99c7a38d440f106c7b5af3b8f0eda4722de7bd3ecccaf55759f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 36, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c82b37a71e15bbf420c29dd1eba03be86fa648560f961b196b6680890412d419", "task_kind": "run", "wait_seconds": 0.003348416881635785, "world_seed": 32710}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.60s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.log`.

### 2026-10-12 03:33:57 UTC — joint_act1_0020_dev

Status: **complete**. Finished 2026-10-12 03:34:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 39, 'defeat': 9}; 8,578 valid transitions.

Mean terminal/cutoff floor 16.81; mean fight HP cost 14.92.

Total job wall time 8.28s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020_dev.log`.

### 2026-10-12 03:34:08 UTC — joint_act1_0021

Status: **complete**. Finished 2026-10-12 03:37:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0020.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 32 --world-start 32688 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1478 non-fault episodes, 33 surfaced faults, 8 truncations; outcomes {'act_clear': 1196, 'defeat': 274, 'GameOutcome.UNDECIDED': 8}; 258,594 valid transitions.

Mean terminal/cutoff floor 16.55; mean fight HP cost 13.83.

PPO: 282 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.18310708412900567, "preparation_seconds": 0.06657174997963011, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 17174, "samples": 17174, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3b9fd73ff6b82f5cc982fdab7de6d4a8144b9927dd614130350d371deeac304b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 111, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "23d4d5b036e2cebc0fd60f0efeda1e3ef14d8855aae4fffc3081049ff095c317", "task_kind": "run", "wait_seconds": 0.0036921249702572823, "world_seed": 32756}, {"action_trace_sha256": "03b8aa71fa5bbf18024fc5fb89b975f9bff72f25675838951869e6cc4bff2eb0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 106, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "51cf61a8365209254a275d10caf31ae800e4493a8805e373b5f9d143a3501ea6", "task_kind": "run", "wait_seconds": 0.0029369578696787357, "world_seed": 32772}, {"action_trace_sha256": "b08bfb35231aa557c7b6482bf817f8366eb0d20787966af93b5842c8e4c7f656", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c82b37a71e15bbf420c29dd1eba03be86fa648560f961b196b6680890412d419", "task_kind": "run", "wait_seconds": 0.0033627080265432596, "world_seed": 32710}, {"action_trace_sha256": "b3dd7b7458e7ec44e7709ab901904424393ea1fec4ca84003a57b2e25ed1521c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 158, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "37755029044437af0d7d1bca6b04f448848e44e7c6904287ec764ac5106b5076", "task_kind": "run", "wait_seconds": 0.0027431659400463104, "world_seed": 32829}, {"action_trace_sha256": "d6105815d58334a2c40c71cc2a3e5675370a2270cda8f7818825b0d52d0fcf75", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 54, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c82b37a71e15bbf420c29dd1eba03be86fa648560f961b196b6680890412d419", "task_kind": "run", "wait_seconds": 0.0033807498402893543, "world_seed": 32710}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.51s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.log`.

### 2026-10-12 03:37:14 UTC — joint_act1_0021_dev

Status: **complete**. Finished 2026-10-12 03:37:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 40, 'defeat': 8}; 8,335 valid transitions.

Mean terminal/cutoff floor 16.83; mean fight HP cost 13.08.

Total job wall time 8.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021_dev.log`.

### 2026-10-12 03:37:24 UTC — joint_act1_0022

Status: **complete**. Finished 2026-10-12 03:40:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0021.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 33 --world-start 32816 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1524 non-fault episodes, 6 surfaced faults, 8 truncations; outcomes {'defeat': 276, 'act_clear': 1240, 'GameOutcome.UNDECIDED': 8}; 265,372 valid transitions.

Mean terminal/cutoff floor 16.70; mean fight HP cost 13.09.

PPO: 467 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.167498541995883, "preparation_seconds": 0.05961104203015566, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 20260, "samples": 20260, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "eb1dfa427eeebbaaa9308bdf543f15c7a87225c9ad248549e518cb40ab98fd71", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 70, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "01f1cd4d59ff94dda3b67181235dc1d14bf7f088a49bb5fe244ea9c19f2b98c7", "task_kind": "run", "wait_seconds": 0.003323584096506238, "world_seed": 32902}, {"action_trace_sha256": "b470210d53dea31a9bcae0337617e11311032af00d2c5f87ce921d49852096cb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 35, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "14c96bb8a6c1625f5ceeccdbfb3f3bc95d5f639aa7981958de78068b91cd2490", "task_kind": "run", "wait_seconds": 0.0038324170745909214, "world_seed": 32843}, {"action_trace_sha256": "cfc0d456b5f0e03816b86797563f8c3b5893a7f2571cd139de3b80d1cb997d9d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 53, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4da50a796129708e1deec464dde0acf5ac7eae3cec61dfdc282cc56f471747e9", "task_kind": "run", "wait_seconds": 0.0032375408336520195, "world_seed": 32938}, {"action_trace_sha256": "f6ec4c2d303899c1021dec8d25beccd12f85158e11278fff4ca088d684d73172", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 146, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "60a67bb8f74b74e4f464c426dfbf03633cf5097f808685406dc8d8138e8d6e21", "task_kind": "run", "wait_seconds": 0.00402787490747869, "world_seed": 32827}, {"action_trace_sha256": "a99dfe6e94fb97797926b1cf4af62ebc2dbedfd563f4a397e5a3c5dc4927631f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 158, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3f3281c880ba397d4b63c54e347b53c743eeeca6fb313ff90b98e1e510443c76", "task_kind": "run", "wait_seconds": 0.0039042499847710133, "world_seed": 32847}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.log`.

### 2026-10-12 03:40:30 UTC — joint_act1_0022_dev

Status: **complete**. Finished 2026-10-12 03:40:40 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 42, 'defeat': 6}; 8,317 valid transitions.

Mean terminal/cutoff floor 16.88; mean fight HP cost 9.42.

Total job wall time 8.01s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022_dev.log`.

### 2026-10-12 03:40:40 UTC — joint_act1_0023

Status: **complete**. Finished 2026-10-12 03:43:45 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0022.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 34 --world-start 32944 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1494 non-fault episodes, 8 surfaced faults, 8 truncations; outcomes {'act_clear': 1158, 'defeat': 328, 'GameOutcome.UNDECIDED': 8}; 258,812 valid transitions.

Mean terminal/cutoff floor 16.73; mean fight HP cost 16.51.

PPO: 538 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.16429258394055068, "preparation_seconds": 0.05684683402068913, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 16376, "samples": 16376, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "11cfe726e796bc64b5c923164cbc0e91066ad001839cb55520ef251690c6395e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 38, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ac0141c0d85a80af3fa9f944cff03c42c99e9999d9c1898de21ab33bbb1151e", "task_kind": "run", "wait_seconds": 0.0036695420276373625, "world_seed": 32970}, {"action_trace_sha256": "2801e411e4346dec7b1fa03d847e96f0a06e164fff982b1a40315df7942bee19", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 167, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "d6d77a361d2f5a305fc5577a22a6b84a6ffe04925a29819e88670ce2b27c9cd8", "task_kind": "run", "wait_seconds": 0.003347165882587433, "world_seed": 33067}, {"action_trace_sha256": "e89011d245a34b120d46c308ec0fec7eda4a8011ec19065fc845fcac5b116ebb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 62, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "31fff07e611fd3783eae19d1b24d840a26581fc8b4017f1fbbdfd62bfc0f13c4", "task_kind": "run", "wait_seconds": 0.0025961671490222216, "world_seed": 33078}, {"action_trace_sha256": "d63c220e03628ab1986a57b0b5297a88833696b6c79ef08379b7a45eeb419b8f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 125, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c6db086729ecfe6db50fdcd5f3cee9b9584dd79be5451220e23ded9ce0e960be", "task_kind": "run", "wait_seconds": 0.003840958932414651, "world_seed": 33051}, {"action_trace_sha256": "4115610316b21ab02f3200eb6f90261674693f4bc6b1c00fe0eef81f4d7147ea", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 37, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ac0141c0d85a80af3fa9f944cff03c42c99e9999d9c1898de21ab33bbb1151e", "task_kind": "run", "wait_seconds": 0.0028525409288704395, "world_seed": 32970}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.46s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.log`.

### 2026-10-12 03:43:45 UTC — joint_act1_0023_dev

Status: **complete**. Finished 2026-10-12 03:43:56 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 39, 'defeat': 9}; 8,418 valid transitions.

Mean terminal/cutoff floor 16.67; mean fight HP cost 14.42.

Total job wall time 8.15s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023_dev.log`.

### 2026-10-12 03:43:56 UTC — joint_act1_0024

Status: **complete**. Finished 2026-10-12 03:47:01 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0023.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 35 --world-start 33072 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 11 rounds, 1380 non-fault episodes, 17 surfaced faults, 8 truncations; outcomes {'defeat': 306, 'act_clear': 1066, 'GameOutcome.UNDECIDED': 8}; 252,483 valid transitions.

Mean terminal/cutoff floor 16.61; mean fight HP cost 17.79.

PPO: 467 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.1758378748781979, "preparation_seconds": 0.06733179208822548, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 20891, "samples": 20891, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3523e81fec25944900a64158620182f246678aeff550e0336e2a221ff9667892", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "27cfcd0f582ce5de926141febff779d9ded70f404a85cbbe65165b9447eedf8f", "task_kind": "run", "wait_seconds": 0.002823874820023775, "world_seed": 33190}, {"action_trace_sha256": "679d0456018678a35780c8e5cb75de26e5e2a661324518c6d4db0038923aa0b4", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 89, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ebad4ba6320d7abf1b97cd9f47d7b030ef1d50be2a9d1a7d56d10aa987afe1f", "task_kind": "run", "wait_seconds": 0.0026077080983668566, "world_seed": 33228}, {"action_trace_sha256": "2702cee03071cdf8d75c09894cf9eb7caf754fd491aa8774f30be8ca9102c466", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 69, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ebad4ba6320d7abf1b97cd9f47d7b030ef1d50be2a9d1a7d56d10aa987afe1f", "task_kind": "run", "wait_seconds": 0.0025849579833447933, "world_seed": 33228}, {"action_trace_sha256": "52791e05c2ecb1a84c2f83ed27c26db103e3c33b265699233bea382fe8de462d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 47, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "31fff07e611fd3783eae19d1b24d840a26581fc8b4017f1fbbdfd62bfc0f13c4", "task_kind": "run", "wait_seconds": 0.0026259589940309525, "world_seed": 33078}, {"action_trace_sha256": "b146d19899973faa10719060f713e4ff7bd6bd1f3cbf192315da44e89b76183e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 71, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ebad4ba6320d7abf1b97cd9f47d7b030ef1d50be2a9d1a7d56d10aa987afe1f", "task_kind": "run", "wait_seconds": 0.003226917004212737, "world_seed": 33228}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.log`.

### 2026-10-12 03:47:01 UTC — joint_act1_0024_dev

Status: **complete**. Finished 2026-10-12 03:47:11 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 41, 'defeat': 7}; 8,651 valid transitions.

Mean terminal/cutoff floor 16.75; mean fight HP cost 11.23.

Total job wall time 8.37s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024_dev.log`.

### 2026-10-12 03:47:11 UTC — joint_act1_0025

Status: **complete**. Finished 2026-10-12 03:50:17 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0024.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 36 --world-start 33200 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1447 non-fault episodes, 20 surfaced faults, 8 truncations; outcomes {'defeat': 298, 'act_clear': 1141, 'GameOutcome.UNDECIDED': 8}; 262,762 valid transitions.

Mean terminal/cutoff floor 16.65; mean fight HP cost 15.90.

PPO: 389 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.16172625008039176, "preparation_seconds": 0.0550783749204129, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 10169, "samples": 10169, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "aadf9454ab69467d814224352acb841a45c37353fd3def7529072519ec37f7c5", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 76, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2ebad4ba6320d7abf1b97cd9f47d7b030ef1d50be2a9d1a7d56d10aa987afe1f", "task_kind": "run", "wait_seconds": 0.0045903329737484455, "world_seed": 33228}, {"action_trace_sha256": "886ddd277b96a087692e0fbdab28d6233edda6ecd20b0e6c2336c689f4415e19", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 119, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c13058f5b493f9a43c737382afb50a687bce31662ae6699da79e1a1a269014a7", "task_kind": "run", "wait_seconds": 0.0023018328938633204, "world_seed": 33265}, {"action_trace_sha256": "4c1c4de05c3d3f3b31dcdbbf7675cc6430a6fcc719b12ee120b83ae6ae5f0e34", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 54, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "659b1b0b4849ffafe40e6258e7fb7ad66e969140558f1f610bdcb0acf3c0ce55", "task_kind": "run", "wait_seconds": 0.003242167178541422, "world_seed": 33278}, {"action_trace_sha256": "51b25519b3dc52c0468a47e1ab674e27d36f9bfa6c24c6473623f5209a7a2b96", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 54, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "659b1b0b4849ffafe40e6258e7fb7ad66e969140558f1f610bdcb0acf3c0ce55", "task_kind": "run", "wait_seconds": 0.002732957946136594, "world_seed": 33278}, {"action_trace_sha256": "818fc653773a308210f9075566e6c35b8575a21ccc902e3a5ccf2eceafa8fab9", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 127, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4b4b908520a1b2b8c9ee48f15e12a2ee6a1a78662f3dac047e28c57bd09a44e5", "task_kind": "run", "wait_seconds": 0.0027238749898970127, "world_seed": 33224}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.49s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.log`.

### 2026-10-12 03:50:17 UTC — joint_act1_0025_dev

Status: **complete**. Finished 2026-10-12 03:50:27 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 44, 'defeat': 4}; 8,975 valid transitions.

Mean terminal/cutoff floor 16.92; mean fight HP cost 6.06.

Total job wall time 8.51s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025_dev.log`.

### 2026-10-12 03:50:27 UTC — joint_act1_0026

Status: **complete**. Finished 2026-10-12 03:53:33 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0025.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 37 --world-start 33328 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1446 non-fault episodes, 24 surfaced faults, 8 truncations; outcomes {'defeat': 271, 'act_clear': 1167, 'GameOutcome.UNDECIDED': 8}; 263,936 valid transitions.

Mean terminal/cutoff floor 16.61; mean fight HP cost 14.24.

PPO: 339 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.15409945789724588, "preparation_seconds": 0.04295979207381606, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 10073, "samples": 10073, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "d390eb1e35020f3d4e1306fa56b361d5147e34df8b566fabe7bc528f4342d33b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 131, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1d833caaa5753e94278a612796acd6bcaa7dbb07fc3ba9b42e4291e73bc52105", "task_kind": "run", "wait_seconds": 0.0029245829209685326, "world_seed": 33394}, {"action_trace_sha256": "f56968e842386e21272f1f648f46461af6e3f6bbb4776c8f2711b271fa40e764", "attempted_action_index": 1, "error": "SimulatorError", "excluded_transitions": 25, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4d08558ca68b92c1ff39960640f30ba81d6bdf244bfea25a9000db77716303d3", "task_kind": "run", "wait_seconds": 0.002888290910050273, "world_seed": 33438}, {"action_trace_sha256": "3e5c3e319489e41bf339cb5613aa2368d4db87c1663b63c209fb0e14924600a7", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 38, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8f2e7210258147746a961058d6f4678c450b728650dd59e79c40c15dafa68e18", "task_kind": "run", "wait_seconds": 0.0030182499904185534, "world_seed": 33404}, {"action_trace_sha256": "bbb05da258fab1d4a89369125c84c2bcaefeea65be1793716fbfee16b82b9813", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 34, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8f2e7210258147746a961058d6f4678c450b728650dd59e79c40c15dafa68e18", "task_kind": "run", "wait_seconds": 0.002898917067795992, "world_seed": 33404}, {"action_trace_sha256": "b6d465a512b3dd2e3d4e41bb21d291f3ffe2f138010ac911cd94314c079e07cf", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 27, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "859c060d0f52d7eefc5a515fa09f453e24f3a278f7565a5e5f08c5c1f938adea", "task_kind": "run", "wait_seconds": 0.003227249952033162, "world_seed": 33402}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.log`.

### 2026-10-12 03:53:33 UTC — joint_act1_0026_dev

Status: **complete**. Finished 2026-10-12 03:53:43 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 45, 'defeat': 3}; 8,883 valid transitions.

Mean terminal/cutoff floor 16.94; mean fight HP cost 4.19.

Total job wall time 8.46s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026_dev.log`.

### 2026-10-12 03:53:43 UTC — joint_act1_0027

Status: **complete**. Finished 2026-10-12 03:56:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 38 --world-start 33456 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1477 non-fault episodes, 6 surfaced faults, 8 truncations; outcomes {'defeat': 270, 'act_clear': 1199, 'GameOutcome.UNDECIDED': 8}; 267,795 valid transitions.

Mean terminal/cutoff floor 16.64; mean fight HP cost 13.35.

PPO: 434 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.15997920813970268, "preparation_seconds": 0.050910916179418564, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 13120, "samples": 13120, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "e0639c639383cde233b519f5c7f18715cc34c68a90a7d36ab15c9e78a7041284", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ae5abd0cd5677ec3c101796da4b2ff2e0ec2d13fed401cc7785f1ac48472ce0b", "task_kind": "run", "wait_seconds": 0.0036057501565665007, "world_seed": 33465}, {"action_trace_sha256": "a4eea9986dfdd5242724eb8548ebe60a6b0641ed551ab108769d77e4cbfa52a8", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 45, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ae5abd0cd5677ec3c101796da4b2ff2e0ec2d13fed401cc7785f1ac48472ce0b", "task_kind": "run", "wait_seconds": 0.0038579581305384636, "world_seed": 33465}, {"action_trace_sha256": "4cdb9e3a9ac1591dbf69ca7e0d30c5143e0a773cbdd485a2be05b7d8a5fa7b3d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 40, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ae5abd0cd5677ec3c101796da4b2ff2e0ec2d13fed401cc7785f1ac48472ce0b", "task_kind": "run", "wait_seconds": 0.0032982919365167618, "world_seed": 33465}, {"action_trace_sha256": "a5fbc9343120d2acf8f9f8dca7978d18e432da54ef57cc9dc69c3a171538c531", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 20, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "051ab7bb8e7746460f88b3159f5f448e8c43abb47909753684316280595d9663", "task_kind": "run", "wait_seconds": 0.00399416615255177, "world_seed": 33585}, {"action_trace_sha256": "3860bfab12166e48b898b5e092e575f665e6221f59d640154251695151aa4ce7", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ae5abd0cd5677ec3c101796da4b2ff2e0ec2d13fed401cc7785f1ac48472ce0b", "task_kind": "run", "wait_seconds": 0.0034696669317781925, "world_seed": 33465}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.log`.

### 2026-10-12 03:56:49 UTC — joint_act1_0027_dev

Status: **complete**. Finished 2026-10-12 03:56:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 47 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 8, 'act_clear': 39}; 8,567 valid transitions.

Mean terminal/cutoff floor 16.83; mean fight HP cost 12.70.

Fault evidence: [{"action_trace_sha256": "8afb958453e716fe214f201ae4fc25137e761b7c79e17874ed5e97fc2a7e04b2", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 130, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "90cb7579f89c2d4afad45601834f354940ebe4b661aa8f32f7656c5885721142", "task_kind": "run", "wait_seconds": 0.0028639999218285084, "world_seed": 10046}].

Total job wall time 8.84s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027_dev.log`.

### 2026-10-12 03:56:59 UTC — joint_act1_0028

Status: **complete**. Finished 2026-10-12 04:00:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0027.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 39 --world-start 33584 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1529 non-fault episodes, 7 surfaced faults, 0 truncations; outcomes {'defeat': 320, 'act_clear': 1209}; 273,756 valid transitions.

Mean terminal/cutoff floor 16.68; mean fight HP cost 15.64.

PPO: 365 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "approx_kl": -3.0850060284137726e-09, "clip_fraction": 0.0, "early_stop": true, "elapsed_seconds": 0.22448187507689, "entropy": 0.41297054290771484, "policy_loss": -0.006793297827243805, "preparation_seconds": 0.06714883400127292, "ratio_mean": 0.9999999403953552, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 23057, "samples": 23057, "steps": 1, "value_loss": 0.019461093470454216}.

Fault evidence: [{"action_trace_sha256": "9e1dbf035440faf372abc443aefdd506f33cb22bd1bd89923bb6e836df2fe79c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 58, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1271295d53889d108f5a1334fb3d33e5d62909cc613fd8bd569ef9bf25a95f32", "task_kind": "run", "wait_seconds": 0.0025475837755948305, "world_seed": 33702}, {"action_trace_sha256": "7e0abd2d217e6a17d7ee7e78f2150abd8dc4fe8de4ae6ebd836caaf93be9f966", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 38, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "fdbc9d292d30a50d5c34e87badfbe0b590b24f3587ef0c20994a38717e89e0d1", "task_kind": "run", "wait_seconds": 0.0028498750180006027, "world_seed": 33649}, {"action_trace_sha256": "aa9b80ee8b22a976ab56b1cf9f10416ab96cb36f63d81425023a61b2b42f46ae", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 14, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "051ab7bb8e7746460f88b3159f5f448e8c43abb47909753684316280595d9663", "task_kind": "run", "wait_seconds": 0.002638167003169656, "world_seed": 33585}, {"action_trace_sha256": "1df09b28ea0e02b0cb6543344e81b92128846e935374cf6752cb0a7eb40de5c5", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b7221c9f7b441dd6a7e8eb571f2025d457cb130029123f289b74f69e57e3462f", "task_kind": "run", "wait_seconds": 0.002976750023663044, "world_seed": 33645}, {"action_trace_sha256": "6ec27c555ed1cd821436afedf41180a10bd6da5a6206695c2d07eda8f288703b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 35, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "fdbc9d292d30a50d5c34e87badfbe0b590b24f3587ef0c20994a38717e89e0d1", "task_kind": "run", "wait_seconds": 0.003402333939447999, "world_seed": 33649}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.log`.

### 2026-10-12 04:00:04 UTC — joint_act1_0028_dev

Status: **complete**. Finished 2026-10-12 04:00:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 42, 'defeat': 6}; 9,062 valid transitions.

Mean terminal/cutoff floor 16.88; mean fight HP cost 9.25.

Total job wall time 8.65s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028_dev.log`.

### 2026-10-12 04:00:15 UTC — joint_act1_0029

Status: **complete**. Finished 2026-10-12 04:03:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0028.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 40 --world-start 33712 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 11 rounds, 1368 non-fault episodes, 4 surfaced faults, 8 truncations; outcomes {'defeat': 301, 'act_clear': 1059, 'GameOutcome.UNDECIDED': 8}; 247,361 valid transitions.

Mean terminal/cutoff floor 16.61; mean fight HP cost 17.07.

PPO: 650 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.17417600005865097, "preparation_seconds": 0.06215841695666313, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 15448, "samples": 15448, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "4b4090f8a5c14c979c3accd7df0a9c2342434cd11173fcd12c5d782aa8ff1b80", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 148, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "00e98ea212652b9935fae57dd60e1c298f733e657046cb6a13302adaee4d231c", "task_kind": "run", "wait_seconds": 0.004812542116269469, "world_seed": 33858}, {"action_trace_sha256": "5251ee4ba93feea118c630fe3245e7221d17e77db4e4d1aed758bc92af40ec42", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 31, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3ec7a6a704dc79753e11925e6bab5de91c904d0ea3472e5052e6ad6098a5880c", "task_kind": "run", "wait_seconds": 0.0028682921547442675, "world_seed": 33817}, {"action_trace_sha256": "9dd9084b007e628d0615c57e6e7550a520909e6250939f0f85aa1b2749f3185d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 145, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "00e98ea212652b9935fae57dd60e1c298f733e657046cb6a13302adaee4d231c", "task_kind": "run", "wait_seconds": 0.0037779160775244236, "world_seed": 33858}, {"action_trace_sha256": "66c08d2143d2775f76039e1bfcc8beeb32f74e2a6ba9a0e8272156dab66b2ab7", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 43, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "3ec7a6a704dc79753e11925e6bab5de91c904d0ea3472e5052e6ad6098a5880c", "task_kind": "run", "wait_seconds": 0.003940416965633631, "world_seed": 33817}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.46s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.log`.

### 2026-10-12 04:03:20 UTC — joint_act1_0029_dev

Status: **complete**. Finished 2026-10-12 04:03:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 42, 'defeat': 6}; 8,810 valid transitions.

Mean terminal/cutoff floor 16.88; mean fight HP cost 9.40.

Total job wall time 8.48s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029_dev.log`.

### 2026-10-12 04:03:30 UTC — joint_act1_0030

Status: **complete**. Finished 2026-10-12 04:06:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0029.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 41 --world-start 33840 --horizon 2048 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 12 rounds, 1408 non-fault episodes, 8 surfaced faults, 8 truncations; outcomes {'act_clear': 1118, 'defeat': 282, 'GameOutcome.UNDECIDED': 8}; 256,683 valid transitions.

Mean terminal/cutoff floor 16.62; mean fight HP cost 14.93.

PPO: 509 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.1316992079373449, "preparation_seconds": 0.029934790916740894, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 352, "samples": 352, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "574bbe063941e1708f4a17da90ca66a2b397074b6558a5e0ef31f1e3ef3fc03b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 151, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "00e98ea212652b9935fae57dd60e1c298f733e657046cb6a13302adaee4d231c", "task_kind": "run", "wait_seconds": 0.0027066669426858425, "world_seed": 33858}, {"action_trace_sha256": "b1bbca0b54380d629a09f3e883d238d0831c40375ed53c7ce04148652ec930fa", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 108, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c09354dec4c293f486e1badab6a9676967b92e8eb5b30d995ff48bd3064d674b", "task_kind": "run", "wait_seconds": 0.002463000128045678, "world_seed": 33902}, {"action_trace_sha256": "2e89d46b2982361646823393a215d789c19166c372ed4b4d0d32ef17b0256d38", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 206, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "4884c404864f0eeb24b18639566495478acc55be4c12cfa24d476fa1980bfe9c", "task_kind": "run", "wait_seconds": 0.0033195000141859055, "world_seed": 33915}, {"action_trace_sha256": "498869b227b1721adce182dafebf4e8f73bdecb3580fd3565ec7e5b97637f8f9", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 34, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "a8c5d9b9eefe6018ef62740eabad9b45cb836ca70a4e3158712d1a7f527d3a3e", "task_kind": "run", "wait_seconds": 0.002658291021361947, "world_seed": 33922}, {"action_trace_sha256": "fab2064ba1be8ce244f1d8ae80c7d591fbc230fdf741e33b9ecb5c2c80586462", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 101, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c09354dec4c293f486e1badab6a9676967b92e8eb5b30d995ff48bd3064d674b", "task_kind": "run", "wait_seconds": 0.002878874773159623, "world_seed": 33902}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.40s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030.log`.

### 2026-10-12 04:06:36 UTC — joint_act1_0030_dev

Status: **complete**. Finished 2026-10-12 04:06:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 1 --horizon 2048 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'act_clear': 40, 'defeat': 8}; 8,468 valid transitions.

Mean terminal/cutoff floor 16.75; mean fight HP cost 13.52.

Total job wall time 8.38s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0030_dev.log`.

### 2026-10-12 04:06:46 UTC — joint_act3_initial_dev

Status: **complete**. Finished 2026-10-12 04:07:06 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_initial_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_initial_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 16,253 valid transitions.

Mean terminal/cutoff floor 28.40; mean fight HP cost 80.00.

Total job wall time 15.97s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_initial_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_initial_dev.log`.

### 2026-10-12 04:07:06 UTC — joint_act3_0000

Status: **complete**. Finished 2026-10-12 04:10:12 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0026.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 11 --world-start 40000 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 8 rounds, 937 non-fault episodes, 5 surfaced faults, 8 truncations; outcomes {'defeat': 927, 'victory': 2, 'GameOutcome.UNDECIDED': 8}; 280,084 valid transitions.

Mean terminal/cutoff floor 25.69; mean fight HP cost 79.39.

PPO: 171 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.1492321661207825, "preparation_seconds": 0.04390350007452071, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 12640, "samples": 12640, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "e548c59c2f31870d794ac3bd22bbec945d69e00db791091c66acb4bded10e27f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 33, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "090a0968498f400f76e045bc202a281f05a684900998d644b597ac02209f49ca", "task_kind": "run", "wait_seconds": 0.0031088341493159533, "world_seed": 40052}, {"action_trace_sha256": "e0c49145f8255c0330d4ba29980843e33c4ba768c0ea3991d9d1650d8fcce045", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 238, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "53ed5e14167a39fc5126eccd2f40e1b930b28667a8ab2bb81368701ae9d4558b", "task_kind": "run", "wait_seconds": 0.002931916853412986, "world_seed": 40020}, {"action_trace_sha256": "b5467bc74e44268568eb4deebf361b867ff040a299684cf51678787836ef342d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 231, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "53ed5e14167a39fc5126eccd2f40e1b930b28667a8ab2bb81368701ae9d4558b", "task_kind": "run", "wait_seconds": 0.003447999944910407, "world_seed": 40020}, {"action_trace_sha256": "9175e6594f9bde1075caf552a5f83a05fecf38734b6f20710dd7d34230ba0ede", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 89, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ce69b3340e9c6b97fd375955bf39a9a57821df76f29afa51fd44ebfc687e3154", "task_kind": "run", "wait_seconds": 0.0029536248184740543, "world_seed": 40081}, {"action_trace_sha256": "563996e66f6beb91b566b74a90d65db3ec1fb79051ef0fbe5bbb5d2ac26f0e9d", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 230, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "53ed5e14167a39fc5126eccd2f40e1b930b28667a8ab2bb81368701ae9d4558b", "task_kind": "run", "wait_seconds": 0.0027994581032544374, "world_seed": 40020}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.53s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.log`.

### 2026-10-12 04:10:12 UTC — joint_act3_0000_dev

Status: **complete**. Finished 2026-10-12 04:10:32 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 46 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'defeat': 46}; 15,371 valid transitions.

Mean terminal/cutoff floor 28.74; mean fight HP cost 80.00.

Fault evidence: [{"action_trace_sha256": "a9675f4bfd7ce30a78f4734790357fb1b7648df06feafda3cf29728c30c03d4f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 128, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e3c71392815e9e535a3bf543cf1c420f1efb205d256010b0ead9afaffd00e644", "task_kind": "run", "wait_seconds": 0.004405790939927101, "world_seed": 10387}, {"action_trace_sha256": "875d65dc740761af7849044c29d5f34e03ba5eda26607829488d52e13e927527", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 117, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "2245bf493af5fccf0da5d74b75a7d0e2d9d31e2027520a7094b298860c4ce406", "task_kind": "run", "wait_seconds": 0.002755583031103015, "world_seed": 10179}].

Total job wall time 15.98s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000_dev.log`.

### 2026-10-12 04:10:32 UTC — joint_act3_0001

Status: **complete**. Finished 2026-10-12 04:13:38 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0000.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 12 --world-start 40128 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 7 rounds, 890 non-fault episodes, 6 surfaced faults, 0 truncations; outcomes {'defeat': 890}; 265,801 valid transitions.

Mean terminal/cutoff floor 25.43; mean fight HP cost 80.00.

PPO: 304 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "approx_kl": 0.004627293641593109, "clip_fraction": 0.09579567354722102, "early_stop": true, "elapsed_seconds": 4.3161199579481035, "entropy": 0.47803330519160286, "policy_loss": -0.0010495907856059856, "preparation_seconds": 0.09360462496988475, "ratio_mean": 0.9994791101236813, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 39017, "samples": 39017, "steps": 61, "value_loss": 0.009001094084538397}.

Fault evidence: [{"action_trace_sha256": "8ea15074607a9e86a08f45fe82d96973083449d497d0e45aae8855c9b755821e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 78, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "680a5d8e9f358dfba889248bf9f63c986a950a1fc9a7eecd7bf8c53bede0e966", "task_kind": "run", "wait_seconds": 0.002715958980843425, "world_seed": 40261}, {"action_trace_sha256": "802555813d37593255c06d6b44d09a32194b7c166ca26b581558715ecf35e6eb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 44, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "f578c6b4e69f366ee28c07cd0544666472fb0e6c56272e0e9d4b7b1c282a2458", "task_kind": "run", "wait_seconds": 0.002866708906367421, "world_seed": 40281}, {"action_trace_sha256": "10431fbef1cd139ebc59e2e706254c990fad6e865b426cc9fce69f965d15079e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 183, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "204c63799646dc90079bce9f8050be561a937d046f6525c99a03c6d1e3e00592", "task_kind": "run", "wait_seconds": 0.004426209023222327, "world_seed": 40138}, {"action_trace_sha256": "969e8d28c736154b1641d7c39a3ee38da48fd7e549d9404d3bd8aa564ac73e3f", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 342, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b61cdf0eb0d09c3d33184e15f002655211c410b165aacb7ed92675cb6dad46d3", "task_kind": "run", "wait_seconds": 0.0030440420377999544, "world_seed": 40248}, {"action_trace_sha256": "b4376ab4917acc3963c501f7f1fdfc04cca79fbbb63f9cdd0f0b42612be62f24", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 112, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b04daa69a0c79668afc2b0af827bf90d2bf32796989156e7d3e0511b4d5f357f", "task_kind": "run", "wait_seconds": 0.003746374975889921, "world_seed": 40225}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.14s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.log`.

### 2026-10-12 04:13:38 UTC — joint_act3_0001_dev

Status: **complete**. Finished 2026-10-12 04:13:58 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 15,900 valid transitions.

Mean terminal/cutoff floor 28.54; mean fight HP cost 80.00.

Total job wall time 15.80s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001_dev.log`.

### 2026-10-12 04:13:58 UTC — joint_act3_0002

Status: **complete**. Finished 2026-10-12 04:17:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 13 --world-start 40256 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 7 rounds, 890 non-fault episodes, 6 surfaced faults, 0 truncations; outcomes {'defeat': 889, 'victory': 1}; 268,103 valid transitions.

Mean terminal/cutoff floor 25.21; mean fight HP cost 79.95.

PPO: 307 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "approx_kl": 0.002138096592137696, "clip_fraction": 0.06178977272727273, "early_stop": true, "elapsed_seconds": 1.465374791994691, "entropy": 0.4898802543228323, "policy_loss": 0.0073727859522808685, "preparation_seconds": 0.08963591721840203, "ratio_mean": 1.000607506795363, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 38433, "samples": 38433, "steps": 22, "value_loss": 0.0067710802170702}.

Fault evidence: [{"action_trace_sha256": "24431c352ffa1070205f67994b419b2dbc32fd25acf60d035222ecdc17c13cd7", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 335, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "26febc6186f3f079f439e7e797fa2729f6bfdaa9f9e02790981899699d42d593", "task_kind": "run", "wait_seconds": 0.003759792074561119, "world_seed": 40395}, {"action_trace_sha256": "c6bf31574b0270009b11de12d44801826f925434d2f1476350048d13a3278094", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 204, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c02b391dd2b5daa322ee645d76dc8a718195702b3fb83a5938b2157374f7b6b7", "task_kind": "run", "wait_seconds": 0.002928250003606081, "world_seed": 40331}, {"action_trace_sha256": "309d8557c0fab838e4b9e698eaf0b1bdff325e13032149a6ab77cdbb502e8ce9", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 330, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "26febc6186f3f079f439e7e797fa2729f6bfdaa9f9e02790981899699d42d593", "task_kind": "run", "wait_seconds": 0.0026457919739186764, "world_seed": 40395}, {"action_trace_sha256": "435fcb4782a315009d634d459640182e38109ea5034e3e09af7b233f824b120e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 82, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "680a5d8e9f358dfba889248bf9f63c986a950a1fc9a7eecd7bf8c53bede0e966", "task_kind": "run", "wait_seconds": 0.003658583154901862, "world_seed": 40261}, {"action_trace_sha256": "604152d5b6a8ed52e16e9844bae8a739daee3fd5ef280377fafd6cce6d8cca77", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 178, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "b1e4ff629a4a9d58bcc5657c08ef7308c2ab0de7286e7c6030196ff794b6c5c8", "task_kind": "run", "wait_seconds": 0.0032912089955061674, "world_seed": 40335}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.13s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.log`.

### 2026-10-12 04:17:04 UTC — joint_act3_0002_dev

Status: **complete**. Finished 2026-10-12 04:17:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 17,061 valid transitions.

Mean terminal/cutoff floor 29.65; mean fight HP cost 80.00.

Total job wall time 16.42s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002_dev.log`.

### 2026-10-12 04:17:24 UTC — joint_act3_0003

Status: **complete**. Finished 2026-10-12 04:20:29 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0002.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 14 --world-start 40384 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 7 rounds, 837 non-fault episodes, 9 surfaced faults, 8 truncations; outcomes {'defeat': 829, 'GameOutcome.UNDECIDED': 8}; 258,079 valid transitions.

Mean terminal/cutoff floor 25.51; mean fight HP cost 79.51.

PPO: 333 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.16590470890514553, "preparation_seconds": 0.05745099997147918, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 22536, "samples": 22536, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "6c14f0229dc9bd6c7b032536075a1400c7b78cf7575b495deccda057feaf288a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 310, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "26febc6186f3f079f439e7e797fa2729f6bfdaa9f9e02790981899699d42d593", "task_kind": "run", "wait_seconds": 0.0036610830575227737, "world_seed": 40395}, {"action_trace_sha256": "ab2ad8d69138ecb3702c4d70fee152df1f125b25aa904ca51a3c17de953bdebb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 31, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "496a50492a3dc5549292715a32effd894fdfc7f0faa64394cba40f2dc2b632fa", "task_kind": "run", "wait_seconds": 0.00306870904751122, "world_seed": 40450}, {"action_trace_sha256": "f1c82ad86f5a99919276ad6070be38596d794973b8a4d6f303f9e6100018410b", "attempted_action_index": 2, "error": "SimulatorError", "excluded_transitions": 435, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "d9f89c152ad2dd3115bb7ce5a3b6a4cc07cfad8ba09e811e7708034d7ebde399", "task_kind": "run", "wait_seconds": 0.0042066669557243586, "world_seed": 40505}, {"action_trace_sha256": "968ff8e193a7512b2e7d25f1c157b01b0f082906501448f89a8bd82b8199ffc4", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 244, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c3201313c3517721a748375e11590f2e1bd6a3c172de44600015f27dd1691b53", "task_kind": "run", "wait_seconds": 0.003305500140413642, "world_seed": 40490}, {"action_trace_sha256": "c9547afc4438bddfc6f5e626553ad4c5e552350dee6a0437ff032291e0516a70", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 320, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "d9f89c152ad2dd3115bb7ce5a3b6a4cc07cfad8ba09e811e7708034d7ebde399", "task_kind": "run", "wait_seconds": 0.002895041834563017, "world_seed": 40505}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.51s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.log`.

### 2026-10-12 04:20:29 UTC — joint_act3_0003_dev

Status: **complete**. Finished 2026-10-12 04:20:50 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 15,567 valid transitions.

Mean terminal/cutoff floor 27.48; mean fight HP cost 80.00.

Total job wall time 15.03s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003_dev.log`.

### 2026-10-12 04:20:50 UTC — joint_act3_0004

Status: **complete**. Finished 2026-10-12 04:23:56 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0003.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 15 --world-start 40512 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 7 rounds, 789 non-fault episodes, 2 surfaced faults, 8 truncations; outcomes {'defeat': 781, 'GameOutcome.UNDECIDED': 8}; 249,313 valid transitions.

Mean terminal/cutoff floor 26.15; mean fight HP cost 79.35.

PPO: 521 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.06098312488757074, "preparation_seconds": 0.053573374869301915, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 6200, "samples": 6200, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "2e515cdd3fcf3c850274511ed234bc55a8d7a38879e43176bb5e756b8593021c", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 272, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "31a95abef9c0dd192ee35dfc197baf678d85b12411fe78a3ced7a0dab421a182", "task_kind": "run", "wait_seconds": 0.0025587910786271095, "world_seed": 40554}, {"action_trace_sha256": "66ff3af2b3598c6417b0f300b76260d69fd6dbf755ea9e2210b6b966b4ef80af", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 204, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "ee42967c5ed01d10257142007bda1f2b619928103863b89775ff520ee291778f", "task_kind": "run", "wait_seconds": 0.0035897919442504644, "world_seed": 40623}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.log`.

### 2026-10-12 04:23:56 UTC — joint_act3_0004_dev

Status: **complete**. Finished 2026-10-12 04:24:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 46 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'defeat': 46}; 16,408 valid transitions.

Mean terminal/cutoff floor 29.63; mean fight HP cost 80.00.

Fault evidence: [{"action_trace_sha256": "ebf034095511502a6e51da9c5b054c3a9254b6e247d740f2b209b9fdf43e0d85", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 346, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "5c791cb4612845c822d7b1a058ee55910530aa8ba851f5b2fb68a9ae330d55ac", "task_kind": "run", "wait_seconds": 0.003484749933704734, "world_seed": 10315}, {"action_trace_sha256": "ddee5f7ba6f1785fba40bd7937b6ca0336a84d310525f62045a37f347eea0ae3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 329, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "43dee3e1fbda32f776355051d7c9be080339e5a474096391e22d2132ee184c7a", "task_kind": "run", "wait_seconds": 0.0031152081210166216, "world_seed": 10221}].

Total job wall time 16.90s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004_dev.log`.

### 2026-10-12 04:24:16 UTC — joint_act3_0005

Status: **complete**. Finished 2026-10-12 04:27:22 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0004.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 16 --world-start 40640 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 6 rounds, 693 non-fault episodes, 15 surfaced faults, 8 truncations; outcomes {'defeat': 685, 'GameOutcome.UNDECIDED': 8}; 217,505 valid transitions.

Mean terminal/cutoff floor 26.13; mean fight HP cost 79.28.

PPO: 766 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.18534654192626476, "preparation_seconds": 0.0716187919024378, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 18539, "samples": 18539, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "fba380f683fbdcbcc52424ceec9ea65deda5bb31fcc48071f227089487679965", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 348, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "8a259913cc6a0be08e91ed7aca09ed0dcf063ee5837aa24f4c89dc8d6908c616", "task_kind": "run", "wait_seconds": 0.0028256659861654043, "world_seed": 40728}, {"action_trace_sha256": "9bd2c21043e7a761807a5268736d811bac9ad289de26d42167c156595028d76e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 317, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "f2938f14ae5c8bdf79c5f0e2e3fe434e4336b2e7d7bc3d70e858d571c6b2ff18", "task_kind": "run", "wait_seconds": 0.0036864171270281076, "world_seed": 40771}, {"action_trace_sha256": "f7c449ec84f28aeb9e8f546e88bc685c2cf9bd89b6aac4f25b17882626c3831a", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 262, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "de32231eaf91c228254a38725e5bbf89cb171a2ce2d3aa4195c7a0686f46b472", "task_kind": "run", "wait_seconds": 0.0033942919690161943, "world_seed": 40652}, {"action_trace_sha256": "8d9b8269d48c83fe586a9ba903b70ea5d0c52f3d8caafae3b70815728893a705", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 62, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "68521232d7b2a688020b444e14490b15e8b2504709b31c00f6d63310cca62b24", "task_kind": "run", "wait_seconds": 0.0028674169443547726, "world_seed": 40718}, {"action_trace_sha256": "91db0d8e1437a6ed2c9bf6fe655176b3983c06f2e66691f2dadd09344553c3cb", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 34, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "a80dbb071b648fe0bd5f9cdea87c09f5a1b2284c5f97621ba9a1a7cd84083f5a", "task_kind": "run", "wait_seconds": 0.002888874849304557, "world_seed": 40743}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.09s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.log`.

### 2026-10-12 04:27:22 UTC — joint_act3_0005_dev

Status: **complete**. Finished 2026-10-12 04:27:42 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 16,346 valid transitions.

Mean terminal/cutoff floor 28.69; mean fight HP cost 80.00.

Total job wall time 15.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005_dev.log`.

### 2026-10-12 04:27:42 UTC — joint_act3_0006

Status: **complete**. Finished 2026-10-12 04:30:48 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0005.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 17 --world-start 40768 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 8 rounds, 895 non-fault episodes, 13 surfaced faults, 8 truncations; outcomes {'defeat': 887, 'GameOutcome.UNDECIDED': 8}; 262,307 valid transitions.

Mean terminal/cutoff floor 24.47; mean fight HP cost 79.53.

PPO: 338 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 0.1510491669178009, "preparation_seconds": 0.045576666947454214, "rejected_kl": null, "rejected_minibatches": 0, "reused_encoding_count": 2488, "samples": 2488, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "a826e11ae82ac713de5255d854ffe1d2d199ac92eb914cc02a1dba5d3031f0c3", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 70, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c9a057c3ec800e3f7bc541afe2268b3b107785027cd5d24f809dcb01ea9bbcf4", "task_kind": "run", "wait_seconds": 0.0025744999293237925, "world_seed": 40901}, {"action_trace_sha256": "99511e0424470433f86071475352d2d19fdc6eb0462f8a069e35cccbbba80b86", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 258, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1572ce7335b9f44c7d1a09eb48901487c710c81d3af1e099cfe0182014531661", "task_kind": "run", "wait_seconds": 0.0028577090706676245, "world_seed": 40821}, {"action_trace_sha256": "8f9b6b4cd5cfa4e1099f325b416ea8127d96ea4948bad5d90b743fdc05e0603b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 48, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c0ca0bf0d998dd36780fb3c77c48223aa8dc4f92f8f678656916e13a47a35da6", "task_kind": "run", "wait_seconds": 0.0027748330030590296, "world_seed": 40865}, {"action_trace_sha256": "477000d4bc2ed01b7d3dd0b08f05b4f4f33a1d7b44108aad7198d9e63aef28db", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 209, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1572ce7335b9f44c7d1a09eb48901487c710c81d3af1e099cfe0182014531661", "task_kind": "run", "wait_seconds": 0.0029609159100800753, "world_seed": 40821}, {"action_trace_sha256": "8fafec9f9e63baa105b5ec4508bc90c9c6e508dbc05b05a784cf5fccd2f00a69", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 223, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "49b59505a7db24ff23731aa8b588fbd251709ef4246c3c8001e1e21e15730377", "task_kind": "run", "wait_seconds": 0.0028042499907314777, "world_seed": 40911}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.log`.

### 2026-10-12 04:30:48 UTC — joint_act3_0006_dev

Status: **complete**. Finished 2026-10-12 04:31:08 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 240 --task-kind run --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --max-act 3 --horizon 4096 --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 48}; 15,709 valid transitions.

Mean terminal/cutoff floor 27.62; mean fight HP cost 80.00.

Total job wall time 14.78s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006_dev.log`.

### 2026-10-12 04:31:08 UTC — joint_act3_0007

Status: **running**. PID 12163; last supervisor heartbeat 2026-10-12 04:32:53 UTC.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0007.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0007.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0006.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 18 --world-start 40896 --horizon 4096 --max-act 3 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --kl-guard-before-update --learning-rate 0.0001 --progress-floor-weight 0.05 --reward-mode progress --terminal-hp-weight 0.0 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 4 rounds, 511 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 511}; 150,115 valid transitions.

Mean terminal/cutoff floor 25.28; mean fight HP cost 80.00.

PPO: 192 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "approx_kl": 0.0074065446664566, "clip_fraction": 0.10229492409432188, "early_stop": true, "elapsed_seconds": 8.753188582835719, "entropy": 0.4532901006387481, "policy_loss": -0.00246811758872148, "preparation_seconds": 0.07453291700221598, "ratio_mean": 0.9988592529127783, "rejected_kl": 0.04816603660583496, "rejected_minibatches": 1, "reused_encoding_count": 37641, "samples": 37641, "steps": 141, "value_loss": 0.0038627360267102294}.

Fault evidence: [{"action_trace_sha256": "f33c0d16cf85166936561d06c8704a5cd41739986163e08fec159fe325185e25", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 66, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "be701e1720613f6773ae42528ae672c62199d69d2741ca268aed6f925f41d802", "task_kind": "run", "wait_seconds": 0.002580750035122037, "world_seed": 40977}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0007.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0007.log`.

## Reproduce or resume

Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and `STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build`. The supervisor script is `scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints and the original UTC deadline. Follow the decision notes for recovery details.
