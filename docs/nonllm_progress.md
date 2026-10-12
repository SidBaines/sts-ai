# Non-LLM curriculum experiment progress log

Updated 2026-10-12 02:48:21 UTC. Campaign status: **running**; phase: **joint_act1**.

Allocation: 2026-10-11 20:08:46 UTC → 2026-10-12 06:30:00 UTC. Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.

This page is generated from the atomic campaign journal and raw metrics. It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. Development results are not final-test results. Simulator faults are separate from ordinary game defeats. [Short latest-first summary](nonllm_summary.md) records the major decisions and supporting evidence.

## Current selection

```json
{
  "combat": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt",
  "joint_act1": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt",
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

Status: **running**. PID 87315; last supervisor heartbeat 2026-10-12 02:45:32 UTC.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind run --seed 19 --world-start 31024 --horizon 2048 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0.0 --kl-guard-before-update --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 9 rounds, 1062 non-fault episodes, 20 surfaced faults, 8 truncations; outcomes {'defeat': 300, 'act_clear': 754, 'GameOutcome.UNDECIDED': 8}; 179,535 valid transitions.

Mean terminal/cutoff floor 16.55; mean fight HP cost 21.55.

PPO: 569 optimizer steps; last update {"algorithm_version": "masked_ppo_v2_before_update_kl", "early_stop": true, "elapsed_seconds": 1.812997000059113, "preparation_seconds": 1.701675415970385, "rejected_kl": null, "rejected_minibatches": 0, "samples": 9096, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "3c7b6e4f37919027759a199aad07deb49d0a555540d28238c48e52e917e110c0", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 50, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "902bf60326ed201544b12356ea3083f48382ed038ac054827f4f1fff87b3f824", "task_kind": "run", "wait_seconds": 0.001757000107318163, "world_seed": 31133}, {"action_trace_sha256": "ec7161916c0b4c233a8c8a39515521f77f1740300dbe25d11ef0e7018f21d65b", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 159, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "1b262244c39391ed1f71be2edd21317e0e51455d3b99b7de0576e33faac4ca05", "task_kind": "run", "wait_seconds": 0.0030285418033599854, "world_seed": 31140}, {"action_trace_sha256": "3f38436647fd7b94022acbdffac4ea66babb2b12af3ed20049aa3ca846cab233", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 138, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "dd586096d82805d610210b1e5e78d6a64dd406f3d1dd1415b86d40bac6920ac2", "task_kind": "run", "wait_seconds": 0.0024272501468658447, "world_seed": 31100}, {"action_trace_sha256": "beb0db4d30984a294eba63983cb260263daf3a5479c7b2293f5ed57370baec34", "attempted_action_index": 2, "error": "SimulatorError", "excluded_transitions": 142, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "0dba94db85d23c98ebbbf455e645564a652322e991f963e1e0c747652aee985f", "task_kind": "run", "wait_seconds": 0.004373667063191533, "world_seed": 31132}, {"action_trace_sha256": "8843587f31509046e51dc88e9ed3817023cbff369dcee30677a4daa81c65f87e", "attempted_action_index": 0, "error": "SimulatorError", "excluded_transitions": 7, "fault_stage": "step", "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "d37901b0de028470f614fe8c3180656a3523c4b3b4f3c37a2eac8d6e72f40cdb", "task_kind": "run", "wait_seconds": 0.0023986250162124634, "world_seed": 31064}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 182.07s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act1_0008.log`.

## Reproduce or resume

Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and `STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build`. The supervisor script is `scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints and the original UTC deadline. Follow the decision notes for recovery details.
