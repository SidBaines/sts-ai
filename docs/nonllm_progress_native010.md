# Archived native0.1.0 timeline

Native-auto OOC/lookahead evidence below is **quarantined** after the diagnosed
character-class bug. Full-control combat evidence is retained for the specific
validated Ironclad defect. This historical snapshot is not the active campaign.
See [current log](nonllm_progress.md) and [summary](nonllm_summary.md).

# Non-LLM curriculum experiment progress log

Updated 2026-10-12 01:03:15 UTC. Campaign status: **running**; phase: **ooc**.

Allocation: 2026-10-11 20:08:46 UTC → 2026-10-12 06:30:00 UTC. Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.

This page is generated from the atomic campaign journal and raw metrics. It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. Development results are not final-test results. Simulator faults are separate from ordinary game defeats. [Short latest-first summary](nonllm_summary.md) records the major decisions and supporting evidence.

## Current selection

```json
{
  "combat": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt",
  "combat_health": "data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt",
  "ooc": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt",
  "ooc_pilot": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt",
  "shared_bc": "/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt"
}
```

## Experiment timeline

### 2026-10-11 20:39:44 UTC — bc_small_none

Status: **complete**. Finished 2026-10-11 20:41:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size small --context none
```

Training: 6,934 updates, 6,956,259 examples, 80.43s; 86,488 examples/s; mean loss 0.6895.

Development imitation: {"accuracy": 0.6101060511540861, "examples": 4809, "loss": 1.1112753903203954, "phase_accuracy": {"combat": 0.5145278450363197, "out_of_combat": 0.8199335548172757}}.

Total job wall time 94.40s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.log`.

### 2026-10-11 20:41:24 UTC — bc_small_none_dev

Status: **failed**. Finished 2026-10-11 20:41:29 UTC; exit code 2.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 2048 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 20:41:29 UTC — bc_small_mean

Status: **complete**. Finished 2026-10-11 20:43:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size small --context mean
```

Training: 6,248 updates, 6,268,296 examples, 79.01s; 79,337 examples/s; mean loss 0.7003.

Development imitation: {"accuracy": 0.6254938656685382, "examples": 4809, "loss": 1.041778044086852, "phase_accuracy": {"combat": 0.535411622276029, "out_of_combat": 0.8232558139534883}}.

Total job wall time 93.98s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.log`.

### 2026-10-11 20:43:04 UTC — bc_small_mean_dev

Status: **failed**. Finished 2026-10-11 20:43:09 UTC; exit code 2.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 2048 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 20:43:09 UTC — bc_medium_none

Status: **complete**. Finished 2026-10-11 20:45:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size medium --context none
```

Training: 4,380 updates, 4,394,702 examples, 78.27s; 56,150 examples/s; mean loss 0.6877.

Development imitation: {"accuracy": 0.6184237887294656, "examples": 4809, "loss": 1.1586176665415757, "phase_accuracy": {"combat": 0.5242130750605327, "out_of_combat": 0.825249169435216}}.

Total job wall time 93.88s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.log`.

### 2026-10-11 20:43:59 UTC — bc_small_none_dev

Status: **complete**. Finished 2026-10-11 20:44:09 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 786 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.52.

Total job wall time 4.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_none_dev.log`.

### 2026-10-11 20:44:09 UTC — bc_small_mean_dev

Status: **complete**. Finished 2026-10-11 20:44:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 759 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.00.

Total job wall time 3.15s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_small_mean_dev.log`.

### 2026-10-11 20:44:14 UTC — bc_medium_none

Status: **complete**. Finished 2026-10-11 20:45:50 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size medium --context none
```

Training: 4,380 updates, 4,394,702 examples, 78.27s; 56,150 examples/s; mean loss 0.6877.

Development imitation: {"accuracy": 0.6184237887294656, "examples": 4809, "loss": 1.1586176665415757, "phase_accuracy": {"combat": 0.5242130750605327, "out_of_combat": 0.825249169435216}}.

Total job wall time 93.88s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.log`.

### 2026-10-11 20:45:50 UTC — bc_medium_none_dev

Status: **complete**. Finished 2026-10-11 20:45:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 765 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.12.

Total job wall time 2.97s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_none_dev.log`.

### 2026-10-11 20:45:55 UTC — bc_medium_mean

Status: **complete**. Finished 2026-10-11 20:47:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size medium --context mean
```

Training: 5,171 updates, 5,187,626 examples, 79.72s; 65,069 examples/s; mean loss 0.6563.

Development imitation: {"accuracy": 0.6238303181534622, "examples": 4809, "loss": 1.3944480326448863, "menu_accuracy": {"choice": 0.519394261424017, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5296610169491526, "out_of_combat": 0.8305647840531561}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.42867647058823527, "forced": 1.0}, "out_of_combat": {"choice": 0.7557471264367817, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 93.88s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean.log`.

### 2026-10-11 20:47:30 UTC — bc_medium_mean_dev

Status: **complete**. Finished 2026-10-11 20:47:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 756 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.98.

Total job wall time 2.90s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_medium_mean_dev.log`.

### 2026-10-11 20:47:35 UTC — bc_large_none

Status: **complete**. Finished 2026-10-11 20:49:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context none
```

Training: 1,965 updates, 1,972,069 examples, 79.50s; 24,806 examples/s; mean loss 0.7792.

Development imitation: {"accuracy": 0.6410896236223748, "examples": 4809, "loss": 0.8064467251040529, "menu_accuracy": {"choice": 0.5414452709883103, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5556900726392252, "out_of_combat": 0.8285714285714286}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4602941176470588, "forced": 1.0}, "out_of_combat": {"choice": 0.7528735632183908, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 93.98s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none.log`.

### 2026-10-11 20:49:10 UTC — bc_large_none_dev

Status: **complete**. Finished 2026-10-11 20:49:15 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 703 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.96.

Total job wall time 2.73s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_dev.log`.

### 2026-10-11 20:49:15 UTC — bc_large_mean

Status: **complete**. Finished 2026-10-11 20:50:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context mean
```

Training: 1,592 updates, 1,597,794 examples, 80.20s; 19,922 examples/s; mean loss 0.7889.

Development imitation: {"accuracy": 0.6429611145768351, "examples": 4809, "loss": 0.7964345099500123, "menu_accuracy": {"choice": 0.543836344314559, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5590193704600485, "out_of_combat": 0.8272425249169435}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.46433823529411766, "forced": 1.0}, "out_of_combat": {"choice": 0.7509578544061303, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.14s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean.log`.

### 2026-10-11 21:29:34 UTC — bc_large_mean_dev

Status: **complete**. Finished 2026-10-11 21:29:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 728 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.88.

Total job wall time 2.83s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_dev.log`.

### 2026-10-11 21:29:39 UTC — bc_large_none_lr0.0001

Status: **complete**. Finished 2026-10-11 21:31:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context none --learning-rate 0.0001
```

Training: 1,993 updates, 1,999,888 examples, 80.20s; 24,937 examples/s; mean loss 0.8095.

Development imitation: {"accuracy": 0.6350592638802246, "examples": 4809, "loss": 0.796403129325386, "menu_accuracy": {"choice": 0.5337407013815091, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5493341404358354, "out_of_combat": 0.8232558139534883}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4525735294117647, "forced": 1.0}, "out_of_combat": {"choice": 0.7452107279693486, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.19s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.log`.

### 2026-10-11 21:31:14 UTC — bc_large_none_lr0.0001_dev

Status: **complete**. Finished 2026-10-11 21:31:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 45, 'battle_defeat': 3}; 687 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.08.

Total job wall time 2.13s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_dev.log`.

### 2026-10-11 21:31:19 UTC — bc_large_none_lr0.0003

Status: **complete**. Finished 2026-10-11 21:32:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context none --learning-rate 0.0003 --fast-replay
```

Training: 1,986 updates, 1,992,720 examples, 80.20s; 24,846 examples/s; mean loss 0.7786.

Development imitation: {"accuracy": 0.6415055105011437, "examples": 4809, "loss": 0.8192290149227848, "menu_accuracy": {"choice": 0.5419766206163655, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5581113801452785, "out_of_combat": 0.8245847176079735}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4632352941176471, "forced": 1.0}, "out_of_combat": {"choice": 0.7471264367816092, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.35s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003.log`.

### 2026-10-11 21:32:59 UTC — bc_large_none_lr0.0003_dev

Status: **complete**. Finished 2026-10-11 21:33:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 732 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.35.

Total job wall time 2.23s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0003_dev.log`.

### 2026-10-11 21:33:04 UTC — bc_large_none_lr0.001

Status: **complete**. Finished 2026-10-11 21:34:44 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context none --learning-rate 0.001 --fast-replay
```

Training: 1,957 updates, 1,963,877 examples, 79.50s; 24,702 examples/s; mean loss 0.6513.

Development imitation: {"accuracy": 0.6084425036390102, "examples": 4809, "loss": 2.3938099335027747, "menu_accuracy": {"choice": 0.4997343251859724, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5066585956416465, "out_of_combat": 0.8318936877076412}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4007352941176471, "forced": 1.0}, "out_of_combat": {"choice": 0.7576628352490421, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001.log`.

### 2026-10-11 21:34:44 UTC — bc_large_none_lr0.001_dev

Status: **complete**. Finished 2026-10-11 21:34:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 40, 'battle_defeat': 8}; 731 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 12.94.

Total job wall time 2.21s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.001_dev.log`.

### 2026-10-11 21:34:49 UTC — bc_large_mean_lr0.0001

Status: **complete**. Finished 2026-10-11 21:36:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context mean --learning-rate 0.0001 --fast-replay
```

Training: 1,616 updates, 1,621,517 examples, 79.95s; 20,282 examples/s; mean loss 0.8173.

Development imitation: {"accuracy": 0.6311083385319193, "examples": 4809, "loss": 0.800829590122971, "menu_accuracy": {"choice": 0.528692879914984, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5460048426150121, "out_of_combat": 0.8179401993355482}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4485294117647059, "forced": 1.0}, "out_of_combat": {"choice": 0.7375478927203065, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001.log`.

### 2026-10-11 21:40:14 UTC — bc_large_mean_lr0.0001_dev

Status: **complete**. Finished 2026-10-11 21:40:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 694 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.33.

Total job wall time 1.98s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0001_dev.log`.

### 2026-10-11 21:40:19 UTC — bc_large_mean_lr0.0003

Status: **complete**. Finished 2026-10-11 21:41:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context mean --learning-rate 0.0003 --fast-replay
```

Training: 1,673 updates, 1,679,032 examples, 80.62s; 20,825 examples/s; mean loss 0.7866.

Development imitation: {"accuracy": 0.6427531711374506, "examples": 4809, "loss": 0.7910803471416712, "menu_accuracy": {"choice": 0.5435706695005313, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.562953995157385, "out_of_combat": 0.8179401993355482}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.46911764705882353, "forced": 1.0}, "out_of_combat": {"choice": 0.7375478927203065, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.17s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003.log`.

### 2026-10-11 21:41:54 UTC — bc_large_mean_lr0.0003_dev

Status: **complete**. Finished 2026-10-11 21:41:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 711 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.90.

Total job wall time 2.22s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.0003_dev.log`.

### 2026-10-11 21:41:59 UTC — bc_large_mean_lr0.001

Status: **complete**. Finished 2026-10-11 21:43:34 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context mean --learning-rate 0.001 --fast-replay
```

Training: 1,667 updates, 1,672,888 examples, 80.31s; 20,830 examples/s; mean loss 0.6973.

Development imitation: {"accuracy": 0.6128093158660844, "examples": 4809, "loss": 1.5744616214389686, "menu_accuracy": {"choice": 0.5053134962805526, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5187651331719129, "out_of_combat": 0.8192691029900332}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.41544117647058826, "forced": 1.0}, "out_of_combat": {"choice": 0.7394636015325671, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.09s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.log`.

### 2026-10-11 21:43:34 UTC — bc_large_mean_lr0.001_dev

Status: **complete**. Finished 2026-10-11 21:43:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 752 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.83.

Total job wall time 2.29s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001_dev.log`.

### 2026-10-11 21:43:39 UTC — bc_large_none_lr0.0001_rep23

Status: **complete**. Finished 2026-10-11 21:45:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --seed 23 --size large --context none --learning-rate 0.0001 --fast-replay
```

Training: 2,000 updates, 2,007,056 examples, 80.16s; 25,039 examples/s; mean loss 0.8102.

Development imitation: {"accuracy": 0.6323559991682263, "examples": 4809, "loss": 0.798576750014621, "menu_accuracy": {"choice": 0.5302869287991498, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5484261501210653, "out_of_combat": 0.8166112956810632}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.4514705882352941, "forced": 1.0}, "out_of_combat": {"choice": 0.735632183908046, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.04s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23.log`.

### 2026-10-11 21:45:14 UTC — bc_large_none_lr0.0001_rep23_dev

Status: **complete**. Finished 2026-10-11 21:45:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 682 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.54.

Total job wall time 2.21s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep23_dev.log`.

### 2026-10-11 21:45:19 UTC — bc_large_none_lr0.0001_rep37

Status: **complete**. Finished 2026-10-11 21:46:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --seed 37 --size large --context none --learning-rate 0.0001 --fast-replay
```

Training: 1,967 updates, 1,974,117 examples, 80.27s; 24,595 examples/s; mean loss 0.8105.

Development imitation: {"accuracy": 0.6331877729257642, "examples": 4809, "loss": 0.7946871898109339, "menu_accuracy": {"choice": 0.5313496280552603, "forced": 1.0}, "menu_counts": {"choice": 3764, "forced": 1045}, "phase_accuracy": {"combat": 0.5517554479418886, "out_of_combat": 0.8119601328903654}, "phase_counts": {"combat": 3304, "out_of_combat": 1505}, "phase_menu_accuracy": {"combat": {"choice": 0.45551470588235293, "forced": 1.0}, "out_of_combat": {"choice": 0.7289272030651341, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}, "out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 94.33s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37.log`.

### 2026-10-11 21:46:54 UTC — bc_large_none_lr0.0001_rep37_dev

Status: **complete**. Finished 2026-10-11 21:46:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 699 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.42.

Total job wall time 2.19s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001_rep37_dev.log`.

### 2026-10-11 21:46:59 UTC — combat_act3_bc

Status: **complete**. Finished 2026-10-11 21:50:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act3_200_seed11000/teacher.jsonl --dev-teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --phase combat --fast-replay
```

Training: 4,035 updates, 4,103,824 examples, 168.28s; 24,386 examples/s; mean loss 0.8617.

Development imitation: {"accuracy": 0.5317796610169492, "examples": 3304, "loss": 1.6163054459320143, "menu_accuracy": {"choice": 0.43125, "forced": 1.0}, "menu_counts": {"choice": 2720, "forced": 584}, "phase_accuracy": {"combat": 0.5317796610169492}, "phase_counts": {"combat": 3304}, "phase_menu_accuracy": {"combat": {"choice": 0.43125, "forced": 1.0}}, "phase_menu_counts": {"combat": {"choice": 2720, "forced": 584}}}.

Total job wall time 185.39s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.log`.

### 2026-10-11 21:50:10 UTC — combat_hardening_before_dev

Status: **complete**. Finished 2026-10-11 21:50:15 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 45, 'battle_defeat': 3}; 687 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.08.

Total job wall time 2.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev.log`.

### 2026-10-11 21:50:15 UTC — combat_hardening_after_dev

Status: **complete**. Finished 2026-10-11 21:50:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 712 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.12.

Total job wall time 2.04s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev.log`.

### 2026-10-11 21:50:20 UTC — combat_pilot_lr0.0001_potential

Status: **complete**. Finished 2026-10-11 21:51:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 22 rounds, 2816 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 2540, 'battle_defeat': 276}; 42,568 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 14.09.

PPO: 195 optimizer steps; last update {"approx_kl": 0.0008776862253268648, "clip_fraction": 0.009067912186895098, "early_stop": true, "elapsed_seconds": 0.9069815829861909, "entropy": 0.504085293837956, "policy_loss": -0.0064920588795627865, "preparation_seconds": 0.3949703748803586, "ratio_mean": 0.999641375882285, "samples": 1853, "steps": 7, "value_loss": 0.03155330754816532}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.08s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.log`.

### 2026-10-11 21:51:55 UTC — combat_pilot_lr0.0001_potential_dev

Status: **complete**. Finished 2026-10-11 21:52:00 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 55, 'battle_defeat': 9}; 877 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 13.38.

Total job wall time 2.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev.log`.

### 2026-10-11 21:52:00 UTC — combat_pilot_lr0.0003_potential

Status: **complete**. Finished 2026-10-11 21:53:31 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0003 --reward-mode potential --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 20 rounds, 2558 non-fault episodes, 2 surfaced faults, 0 truncations; outcomes {'battle_victory': 2352, 'battle_defeat': 206}; 38,936 valid transitions.

Mean terminal/cutoff floor 7.85; mean fight HP cost 12.89.

PPO: 147 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5591179998591542, "preparation_seconds": 0.4417792079038918, "samples": 1896, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 11, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}, {"error": "SimulatorError", "excluded_transitions": 23, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.43s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.log`.

### 2026-10-11 22:00:20 UTC — combat_hardening_before_dev_act1

Status: **complete**. Finished 2026-10-11 22:00:25 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 45, 'battle_defeat': 3}; 687 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.08.

Total job wall time 1.95s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_act1.log`.

### 2026-10-11 22:00:25 UTC — combat_hardening_before_dev_hard

Status: **complete**. Finished 2026-10-11 22:00:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 22:00:30 UTC — combat_hardening_after_dev_act1

Status: **complete**. Finished 2026-10-11 22:00:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 712 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.12.

Total job wall time 1.96s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_act1.log`.

### 2026-10-11 22:00:36 UTC — combat_hardening_after_dev_hard

Status: **complete**. Finished 2026-10-11 22:00:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 22:00:41 UTC — combat_pilot_lr0.0001_potential_dev_act1

Status: **complete**. Finished 2026-10-11 22:00:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 55, 'battle_defeat': 9}; 877 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 13.38.

Total job wall time 2.22s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_act1.log`.

### 2026-10-11 22:00:46 UTC — combat_pilot_lr0.0001_potential_dev_hard

Status: **complete**. Finished 2026-10-11 22:00:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 6, 'battle_victory': 20}; 606 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 32.96.

Total job wall time 2.88s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0001_potential_dev_hard.log`.

### 2026-10-11 22:00:51 UTC — combat_pilot_lr0.0003_potential_dev_act1

Status: **complete**. Finished 2026-10-11 22:00:56 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 59, 'battle_defeat': 5}; 909 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.25.

Total job wall time 2.20s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_act1.log`.

### 2026-10-11 22:00:56 UTC — combat_pilot_lr0.0003_potential_dev_hard

Status: **complete**. Finished 2026-10-11 22:01:01 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 6, 'battle_victory': 20}; 589 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 30.38.

Total job wall time 2.92s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_dev_hard.log`.

### 2026-10-11 22:01:01 UTC — combat_pilot_lr0.001_potential

Status: **complete**. Finished 2026-10-11 22:02:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 24 rounds, 3071 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'battle_victory': 2754, 'battle_defeat': 317}; 44,923 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 14.86.

PPO: 99 optimizer steps; last update {"approx_kl": 0.014889688881729464, "clip_fraction": 0.07459577133080789, "early_stop": true, "elapsed_seconds": 0.8884364168625325, "entropy": 0.41229434524263653, "policy_loss": -0.00502160404409681, "preparation_seconds": 0.40276462491601706, "ratio_mean": 1.004225390298026, "samples": 1945, "steps": 7, "value_loss": 0.021372010665280477}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 13, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.log`.

### 2026-10-11 22:02:36 UTC — combat_pilot_lr0.001_potential_dev_act1

Status: **complete**. Finished 2026-10-11 22:02:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 12, 'battle_victory': 52}; 927 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 15.50.

Total job wall time 2.35s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_act1.log`.

### 2026-10-11 22:02:41 UTC — combat_pilot_lr0.001_potential_dev_hard

Status: **complete**. Finished 2026-10-11 22:02:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 21, 'battle_defeat': 5}; 755 valid transitions.

Mean terminal/cutoff floor 30.19; mean fight HP cost 35.23.

Total job wall time 3.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.001_potential_dev_hard.log`.

### 2026-10-11 22:02:46 UTC — combat_pilot_lr0.0003_sparse

Status: **complete**. Finished 2026-10-11 22:04:21 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0003 --reward-mode sparse --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2939 non-fault episodes, 5 surfaced faults, 0 truncations; outcomes {'battle_victory': 2676, 'battle_defeat': 263}; 44,771 valid transitions.

Mean terminal/cutoff floor 7.85; mean fight HP cost 12.67.

PPO: 103 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5013672499917448, "preparation_seconds": 0.39590129093267024, "samples": 1955, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 23, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}, {"error": "SimulatorError", "excluded_transitions": 23, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}, {"error": "SimulatorError", "excluded_transitions": 23, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}, {"error": "SimulatorError", "excluded_transitions": 23, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}, {"error": "SimulatorError", "excluded_transitions": 24, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "c42d85db89484d47cda96543f44f8baab2356b051585005422052fa3d910cd06", "task_kind": "battle", "world_seed": 10216}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.log`.

### 2026-10-11 22:04:21 UTC — combat_pilot_lr0.0003_sparse_dev_act1

Status: **complete**. Finished 2026-10-11 22:04:26 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 55, 'battle_defeat': 9}; 945 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 13.00.

Total job wall time 2.34s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_act1.log`.

### 2026-10-11 22:04:26 UTC — combat_pilot_lr0.0003_sparse_dev_hard

Status: **complete**. Finished 2026-10-11 22:04:31 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 16, 'battle_defeat': 10}; 586 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 37.46.

Total job wall time 3.01s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_sparse_dev_hard.log`.

### 2026-10-11 22:04:31 UTC — combat_pilot_clip0.1

Status: **complete**. Finished 2026-10-11 22:06:06 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2837 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 2585, 'battle_defeat': 244, 'GameOutcome.UNDECIDED': 8}; 44,324 valid transitions.

Mean terminal/cutoff floor 7.85; mean fight HP cost 12.19.

PPO: 176 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.15450362488627434, "preparation_seconds": 0.05361312488093972, "samples": 224, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.28s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.log`.

### 2026-10-11 22:06:06 UTC — combat_pilot_clip0.1_dev_act1

Status: **complete**. Finished 2026-10-11 22:06:11 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 9, 'battle_victory': 55}; 999 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 11.77.

Total job wall time 2.38s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_act1.log`.

### 2026-10-11 22:06:11 UTC — combat_pilot_clip0.1_dev_hard

Status: **complete**. Finished 2026-10-11 22:06:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 21, 'battle_defeat': 5}; 635 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 32.15.

Total job wall time 2.99s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_dev_hard.log`.

### 2026-10-11 22:06:16 UTC — combat_pilot_entropy0.005

Status: **complete**. Finished 2026-10-11 22:07:52 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.005 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2896 non-fault episodes, 0 surfaced faults, 7 truncations; outcomes {'battle_victory': 2639, 'battle_defeat': 250, 'GameOutcome.UNDECIDED': 7}; 44,326 valid transitions.

Mean terminal/cutoff floor 7.83; mean fight HP cost 12.06.

PPO: 170 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.3384784581139684, "preparation_seconds": 0.2314178750384599, "samples": 1096, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.log`.

### 2026-10-11 22:07:52 UTC — combat_pilot_entropy0.005_dev_act1

Status: **complete**. Finished 2026-10-11 22:07:57 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 58, 'battle_defeat': 6}; 993 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 10.78.

Total job wall time 2.39s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_act1.log`.

### 2026-10-11 22:07:57 UTC — combat_pilot_entropy0.005_dev_hard

Status: **complete**. Finished 2026-10-11 22:08:02 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 7, 'battle_victory': 19}; 626 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 32.35.

Total job wall time 3.22s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.005_dev_hard.log`.

### 2026-10-11 22:08:02 UTC — combat_pilot_entropy0.02

Status: **complete**. Finished 2026-10-11 22:09:37 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.02 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 22 rounds, 2816 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 2583, 'battle_defeat': 233}; 44,911 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 13.23.

PPO: 183 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5520621249452233, "preparation_seconds": 0.4493112911004573, "samples": 2101, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.38s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.log`.

### 2026-10-11 22:09:37 UTC — combat_pilot_entropy0.02_dev_act1

Status: **complete**. Finished 2026-10-11 22:09:42 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 55, 'battle_defeat': 9}; 990 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.97.

Total job wall time 2.34s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_act1.log`.

### 2026-10-11 22:09:42 UTC — combat_pilot_entropy0.02_dev_hard

Status: **complete**. Finished 2026-10-11 22:09:47 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 7, 'battle_victory': 19}; 646 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 35.38.

Total job wall time 3.16s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_entropy0.02_dev_hard.log`.

### 2026-10-11 22:09:47 UTC — combat_pilot_gamma0.99

Status: **complete**. Finished 2026-10-11 22:11:22 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 0.99 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2942 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 2683, 'battle_defeat': 251, 'GameOutcome.UNDECIDED': 8}; 43,363 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 12.54.

PPO: 166 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.4429639168083668, "preparation_seconds": 0.34135487489402294, "samples": 1672, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.59s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.log`.

### 2026-10-11 22:11:22 UTC — combat_pilot_gamma0.99_dev_act1

Status: **complete**. Finished 2026-10-11 22:11:27 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 58, 'battle_defeat': 6}; 939 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.44.

Total job wall time 2.35s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_act1.log`.

### 2026-10-11 22:11:27 UTC — combat_pilot_gamma0.99_dev_hard

Status: **complete**. Finished 2026-10-11 22:11:32 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 20, 'battle_defeat': 6}; 606 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 33.65.

Total job wall time 3.03s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_dev_hard.log`.

### 2026-10-11 22:11:32 UTC — combat_pilot_terminal_hp0.25

Status: **complete**. Finished 2026-10-11 22:13:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.25 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2878 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 2639, 'battle_defeat': 231, 'GameOutcome.UNDECIDED': 8}; 43,286 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 12.40.

PPO: 169 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.29348529200069606, "preparation_seconds": 0.17759241699241102, "samples": 808, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.43s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.log`.

### 2026-10-11 22:13:07 UTC — combat_pilot_terminal_hp0.25_dev_act1

Status: **complete**. Finished 2026-10-11 22:13:12 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 7, 'battle_victory': 57}; 927 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.36.

Total job wall time 2.38s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_act1.log`.

### 2026-10-11 22:13:12 UTC — combat_pilot_terminal_hp0.25_dev_hard

Status: **complete**. Finished 2026-10-11 22:13:17 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 19, 'battle_defeat': 7}; 642 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 36.19.

Total job wall time 3.07s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_terminal_hp0.25_dev_hard.log`.

### 2026-10-11 22:13:17 UTC — combat_pilot_scratch

Status: **failed**. Finished 2026-10-11 22:13:22 UTC; exit code 2.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.0 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --size large --context none --fast-replay --warm-optimizer
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 22:13:22 UTC — combat_pilot_lr0.0003_potential_expanded_dev_act1

Status: **complete**. Finished 2026-10-11 22:13:27 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 59, 'battle_defeat': 5}; 909 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.25.

Total job wall time 2.21s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_act1.log`.

### 2026-10-11 22:13:27 UTC — combat_pilot_lr0.0003_potential_expanded_dev_hard

Status: **complete**. Finished 2026-10-11 22:13:32 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 24, 'battle_defeat': 8}; 565 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 26.66.

Total job wall time 3.18s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_lr0.0003_potential_expanded_dev_hard.log`.

### 2026-10-11 22:13:32 UTC — combat_pilot_gamma0.99_expanded_dev_act1

Status: **complete**. Finished 2026-10-11 22:13:37 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 58, 'battle_defeat': 6}; 939 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 12.44.

Total job wall time 2.34s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_act1.log`.

### 2026-10-11 22:13:37 UTC — combat_pilot_gamma0.99_expanded_dev_hard

Status: **complete**. Finished 2026-10-11 22:13:43 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 653 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 21.41.

Total job wall time 3.50s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_gamma0.99_expanded_dev_hard.log`.

### 2026-10-11 22:13:43 UTC — combat_pilot_clip0.1_expanded_dev_act1

Status: **complete**. Finished 2026-10-11 22:13:48 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 9, 'battle_victory': 55}; 999 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 11.77.

Total job wall time 2.36s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_act1.log`.

### 2026-10-11 22:13:48 UTC — combat_pilot_clip0.1_expanded_dev_hard

Status: **complete**. Finished 2026-10-11 22:13:53 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 657 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 20.56.

Total job wall time 3.27s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1_expanded_dev_hard.log`.

### 2026-10-11 22:13:53 UTC — combat_initial_dev_act1

Status: **complete**. Finished 2026-10-11 22:13:58 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 727 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.15.

Total job wall time 1.99s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_act1.log`.

### 2026-10-11 22:13:58 UTC — combat_initial_dev_hard

Status: **complete**. Finished 2026-10-11 22:14:03 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 657 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 20.56.

Total job wall time 3.29s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_initial_dev_hard.log`.

### 2026-10-11 22:14:03 UTC — combat_0000

Status: **complete**. Finished 2026-10-11 22:17:08 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_clip0.1.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 11 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act3_200_seed11000/stage_subsets/battles_act1_early.jsonl --horizon 256 --teacher-simulations 200 --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 65 rounds, 8252 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 8242, 'battle_defeat': 2, 'GameOutcome.UNDECIDED': 8}; 113,311 valid transitions.

Mean terminal/cutoff floor 1.81; mean fight HP cost 2.99.

PPO: 512 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.22280783392488956, "preparation_seconds": 0.11557383392937481, "samples": 688, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.48s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000.log`.

### 2026-10-11 22:17:12 UTC — combat_0000_dagger

Status: **complete**. Finished 2026-10-11 22:17:44 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 717 updates, 717,648 examples, 24.09s; 29,787 examples/s; mean loss 0.7074.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.38s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.log`.

### 2026-10-11 22:21:19 UTC — combat_pilot_scratch

Status: **complete**. Finished 2026-10-11 22:22:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --size large --context none --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --learning-rate 0.0001 --reward-mode potential --gamma 1.0 --clip-ratio 0.2 --entropy-coef 0.01 --terminal-hp-weight 0.0 --fast-replay
```

Live collection: 22 rounds, 2795 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_defeat': 452, 'battle_victory': 2335, 'GameOutcome.UNDECIDED': 8}; 45,046 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 21.58.

PPO: 200 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.41249599982984364, "preparation_seconds": 0.3119769578333944, "samples": 1536, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.log`.

### 2026-10-11 22:22:54 UTC — combat_pilot_scratch_dev_act1

Status: **complete**. Finished 2026-10-11 22:22:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.pt --device cpu --threads 2 --workers 8 --episodes 64 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 64 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 54, 'battle_defeat': 10}; 920 valid transitions.

Mean terminal/cutoff floor 7.27; mean fight HP cost 16.11.

Total job wall time 2.41s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_act1.log`.

### 2026-10-11 22:22:59 UTC — combat_pilot_scratch_dev_hard

Status: **complete**. Finished 2026-10-11 22:23:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 26 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 6, 'battle_victory': 20}; 624 valid transitions.

Mean terminal/cutoff floor 30.15; mean fight HP cost 35.65.

Total job wall time 2.89s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_pilot_scratch_dev_hard.log`.

### 2026-10-11 22:25:15 UTC — combat_hardening_before_dev_hard

Status: **complete**. Finished 2026-10-11 22:25:20 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 22:25:20 UTC — combat_hardening_after_dev_hard

Status: **complete**. Finished 2026-10-11 22:25:25 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 22:25:29 UTC — combat_0000_dev_act1

Status: **complete**. Finished 2026-10-11 22:25:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 753 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.46.

Total job wall time 1.93s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_act1.log`.

### 2026-10-11 22:25:35 UTC — combat_0000_dev_hard

Status: **complete**. Finished 2026-10-11 22:25:40 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 624 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 22.44.

Total job wall time 3.33s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dev_hard.log`.

### 2026-10-11 22:25:40 UTC — combat_0001

Status: **complete**. Finished 2026-10-11 22:28:45 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0000_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 12 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 39 rounds, 4961 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 4495, 'battle_defeat': 458, 'GameOutcome.UNDECIDED': 8}; 87,447 valid transitions.

Mean terminal/cutoff floor 8.09; mean fight HP cost 12.53.

PPO: 447 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5010877919849008, "preparation_seconds": 0.38926816708408296, "samples": 1600, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.75s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.log`.

### 2026-10-11 22:28:45 UTC — combat_0001_dev_act1

Status: **complete**. Finished 2026-10-11 22:28:50 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 754 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 12.04.

Total job wall time 2.19s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_act1.log`.

### 2026-10-11 22:28:50 UTC — combat_0001_dev_hard

Status: **complete**. Finished 2026-10-11 22:28:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 27, 'battle_defeat': 5}; 635 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 19.62.

Total job wall time 3.55s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001_dev_hard.log`.

### 2026-10-11 22:28:55 UTC — combat_0002

Status: **complete**. Finished 2026-10-11 22:32:00 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0001.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 13 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 43 rounds, 5469 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 5014, 'battle_defeat': 447, 'GameOutcome.UNDECIDED': 8}; 91,996 valid transitions.

Mean terminal/cutoff floor 7.59; mean fight HP cost 12.39.

PPO: 482 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.4135402499232441, "preparation_seconds": 0.30312512489035726, "samples": 1520, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.65s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.log`.

### 2026-10-11 22:32:00 UTC — combat_0002_dev_act1

Status: **complete**. Finished 2026-10-11 22:32:05 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 726 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 13.56.

Total job wall time 2.05s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_act1.log`.

### 2026-10-11 22:32:05 UTC — combat_0002_dev_hard

Status: **complete**. Finished 2026-10-11 22:32:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 27, 'battle_defeat': 5}; 659 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 22.28.

Total job wall time 3.50s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002_dev_hard.log`.

### 2026-10-11 22:32:10 UTC — combat_0003

Status: **complete**. Finished 2026-10-11 22:35:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0002.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 14 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 42 rounds, 5376 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 4898, 'battle_defeat': 478}; 90,088 valid transitions.

Mean terminal/cutoff floor 7.88; mean fight HP cost 12.86.

PPO: 481 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.450205666013062, "preparation_seconds": 0.4428505830001086, "samples": 2187, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.70s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003.log`.

### 2026-10-11 22:35:28 UTC — combat_0003_dagger

Status: **complete**. Finished 2026-10-11 22:36:03 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 686 updates, 686,624 examples, 23.35s; 29,401 examples/s; mean loss 0.8083.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.log`.

### 2026-10-11 22:36:03 UTC — combat_0003_dev_act1

Status: **complete**. Finished 2026-10-11 22:36:08 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 764 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.92.

Total job wall time 2.10s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_act1.log`.

### 2026-10-11 22:36:08 UTC — combat_0003_dev_hard

Status: **complete**. Finished 2026-10-11 22:36:13 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 657 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 21.81.

Total job wall time 3.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dev_hard.log`.

### 2026-10-11 22:36:13 UTC — combat_0004

Status: **complete**. Finished 2026-10-11 22:39:18 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0003_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 15 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 39 rounds, 4974 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 4464, 'battle_defeat': 502, 'GameOutcome.UNDECIDED': 8}; 93,455 valid transitions.

Mean terminal/cutoff floor 7.76; mean fight HP cost 15.90.

PPO: 456 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5060490421019495, "preparation_seconds": 0.3984534170012921, "samples": 1984, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.76s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.log`.

### 2026-10-11 22:39:18 UTC — combat_0004_dev_act1

Status: **complete**. Finished 2026-10-11 22:39:23 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 805 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 12.71.

Total job wall time 2.06s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_act1.log`.

### 2026-10-11 22:39:23 UTC — combat_0004_dev_hard

Status: **complete**. Finished 2026-10-11 22:39:28 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 25, 'battle_defeat': 7}; 670 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 25.09.

Total job wall time 3.31s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004_dev_hard.log`.

### 2026-10-11 22:39:28 UTC — combat_0005

Status: **complete**. Finished 2026-10-11 22:42:34 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0004.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 16 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 44 rounds, 5512 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 4878, 'battle_defeat': 626, 'GameOutcome.UNDECIDED': 8}; 92,192 valid transitions.

Mean terminal/cutoff floor 6.91; mean fight HP cost 12.41.

PPO: 512 optimizer steps; last update {"samples": 0, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.23s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.log`.

### 2026-10-11 22:42:34 UTC — combat_0005_dev_act1

Status: **complete**. Finished 2026-10-11 22:42:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 818 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.17.

Total job wall time 2.24s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_act1.log`.

### 2026-10-11 22:42:39 UTC — combat_0005_dev_hard

Status: **complete**. Finished 2026-10-11 22:42:44 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 25, 'battle_defeat': 7}; 638 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 29.09.

Total job wall time 3.45s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005_dev_hard.log`.

### 2026-10-11 22:42:44 UTC — combat_0006

Status: **complete**. Finished 2026-10-11 22:45:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0005.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 17 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 42 rounds, 5376 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 5044, 'battle_defeat': 332}; 88,602 valid transitions.

Mean terminal/cutoff floor 7.84; mean fight HP cost 14.47.

PPO: 480 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5383785830345005, "preparation_seconds": 0.4336907500401139, "samples": 2111, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.32s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006.log`.

### 2026-10-11 22:46:09 UTC — combat_0006_dagger

Status: **complete**. Finished 2026-10-11 22:46:44 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 632 updates, 632,048 examples, 23.06s; 27,404 examples/s; mean loss 0.8244.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.69s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.log`.

### 2026-10-11 22:46:44 UTC — combat_0006_dev_act1

Status: **complete**. Finished 2026-10-11 22:46:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 779 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.83.

Total job wall time 2.21s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_act1.log`.

### 2026-10-11 22:46:49 UTC — combat_0006_dev_hard

Status: **complete**. Finished 2026-10-11 22:46:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 25, 'battle_defeat': 7}; 659 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 23.53.

Total job wall time 3.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dev_hard.log`.

### 2026-10-11 22:46:54 UTC — combat_0007

Status: **complete**. Finished 2026-10-11 22:49:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0006_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 18 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Live collection: 38 rounds, 4803 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 4246, 'battle_defeat': 549, 'GameOutcome.UNDECIDED': 8}; 87,371 valid transitions.

Mean terminal/cutoff floor 8.50; mean fight HP cost 14.20.

PPO: 444 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.37538591702468693, "preparation_seconds": 0.2682349169626832, "samples": 1176, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.62s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.log`.

### 2026-10-11 22:49:59 UTC — combat_0007_dev_act1

Status: **complete**. Finished 2026-10-11 22:50:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 827 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 12.35.

Total job wall time 2.23s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_act1.log`.

### 2026-10-11 22:50:04 UTC — combat_0007_dev_hard

Status: **complete**. Finished 2026-10-11 22:50:09 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 25, 'battle_defeat': 7}; 677 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 24.88.

Total job wall time 3.56s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007_dev_hard.log`.

### 2026-10-11 22:50:09 UTC — combat_0008

Status: **complete**. Finished 2026-10-11 22:53:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 19 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 1 --fast-replay --warm-optimizer
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-11 23:18:08 UTC — combat_hardening_before_dev_hard

Status: **complete**. Finished 2026-10-11 23:18:13 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_defeat': 5, 'battle_victory': 27}; 592 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 25.25.

Total job wall time 2.51s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_before_dev_hard.log`.

### 2026-10-11 23:18:13 UTC — combat_hardening_after_dev_hard

Status: **complete**. Finished 2026-10-11 23:18:18 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_act3_bc.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 661 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 24.88.

Total job wall time 2.65s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_hardening_after_dev_hard.log`.

### 2026-10-11 23:18:18 UTC — combat_0008

Status: **complete**. Finished 2026-10-11 23:21:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0007.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 19 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act3.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 16 rounds, 2043 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_defeat': 349, 'battle_victory': 1686, 'GameOutcome.UNDECIDED': 8}; 45,522 valid transitions.

Mean terminal/cutoff floor 40.81; mean fight HP cost 27.73.

PPO: 153 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.9959670829121023, "preparation_seconds": 0.8859472500625998, "samples": 2656, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.27s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.log`.

### 2026-10-11 23:21:24 UTC — combat_0008_dev_act1

Status: **complete**. Finished 2026-10-11 23:21:29 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 873 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 13.94.

Total job wall time 1.96s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_act1.log`.

### 2026-10-11 23:21:29 UTC — combat_0008_dev_hard

Status: **complete**. Finished 2026-10-11 23:21:34 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 25, 'battle_defeat': 7}; 626 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 22.66.

Total job wall time 2.55s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008_dev_hard.log`.

### 2026-10-11 23:21:34 UTC — combat_0009

Status: **complete**. Finished 2026-10-11 23:24:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0008.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 20 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act1.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 43 rounds, 5455 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 4592, 'battle_defeat': 855, 'GameOutcome.UNDECIDED': 8}; 106,899 valid transitions.

Mean terminal/cutoff floor 8.39; mean fight HP cost 16.97.

PPO: 504 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.3946223750244826, "preparation_seconds": 0.2881957080680877, "samples": 1360, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.63s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009.log`.

### 2026-10-11 23:25:08 UTC — combat_0009_dagger

Status: **complete**. Finished 2026-10-11 23:25:43 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 627 updates, 627,648 examples, 22.78s; 27,556 examples/s; mean loss 0.8353.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.52s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.log`.

### 2026-10-11 23:25:43 UTC — combat_0009_dev_act1

Status: **complete**. Finished 2026-10-11 23:25:48 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 746 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.00.

Total job wall time 1.89s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_act1.log`.

### 2026-10-11 23:25:48 UTC — combat_0009_dev_hard

Status: **complete**. Finished 2026-10-11 23:25:53 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 757 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 23.78.

Total job wall time 2.84s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dev_hard.log`.

### 2026-10-11 23:25:53 UTC — combat_0010

Status: **complete**. Finished 2026-10-11 23:28:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0009_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 21 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act2.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2939 non-fault episodes, 0 surfaced faults, 7 truncations; outcomes {'battle_victory': 2172, 'battle_defeat': 760, 'GameOutcome.UNDECIDED': 7}; 69,662 valid transitions.

Mean terminal/cutoff floor 24.26; mean fight HP cost 24.82.

PPO: 296 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.8469410839024931, "preparation_seconds": 0.7406288338825107, "samples": 2776, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.13s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.log`.

### 2026-10-11 23:28:59 UTC — combat_0010_dev_act1

Status: **complete**. Finished 2026-10-11 23:29:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 639 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.21.

Total job wall time 1.78s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_act1.log`.

### 2026-10-11 23:29:04 UTC — combat_0010_dev_hard

Status: **complete**. Finished 2026-10-11 23:29:09 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 27, 'battle_defeat': 5}; 634 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 18.16.

Total job wall time 2.70s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010_dev_hard.log`.

### 2026-10-11 23:29:09 UTC — combat_0011

Status: **complete**. Finished 2026-10-11 23:32:14 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0010.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 22 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act3.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 15 rounds, 1806 non-fault episodes, 12 surfaced faults, 8 truncations; outcomes {'battle_victory': 1402, 'battle_defeat': 396, 'GameOutcome.UNDECIDED': 8}; 44,567 valid transitions.

Mean terminal/cutoff floor 41.18; mean fight HP cost 25.18.

PPO: 212 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.28328037494793534, "preparation_seconds": 0.1763714998960495, "samples": 488, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 7, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 1, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 6, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 7, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 7, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.log`.

### 2026-10-11 23:32:14 UTC — combat_0011_dev_act1

Status: **complete**. Finished 2026-10-11 23:32:19 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 680 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.23.

Total job wall time 1.82s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_act1.log`.

### 2026-10-11 23:32:19 UTC — combat_0011_dev_hard

Status: **complete**. Finished 2026-10-11 23:32:24 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 640 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 16.72.

Total job wall time 2.57s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011_dev_hard.log`.

### 2026-10-11 23:32:24 UTC — combat_0012

Status: **complete**. Finished 2026-10-11 23:35:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0011.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 23 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act1.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 49 rounds, 6165 non-fault episodes, 2 surfaced faults, 8 truncations; outcomes {'battle_victory': 5840, 'battle_defeat': 317, 'GameOutcome.UNDECIDED': 8}; 101,378 valid transitions.

Mean terminal/cutoff floor 7.61; mean fight HP cost 10.69.

PPO: 547 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.18013491691090167, "preparation_seconds": 0.07583304191939533, "samples": 320, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 7, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "12fb9e8d360d007807d406f8d1bf147fe43052afc0ea1b89794920ae3b65707f", "task_kind": "battle", "world_seed": 13329}, {"error": "SimulatorError", "excluded_transitions": 8, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "12fb9e8d360d007807d406f8d1bf147fe43052afc0ea1b89794920ae3b65707f", "task_kind": "battle", "world_seed": 13329}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.39s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012.log`.

### 2026-10-11 23:36:06 UTC — combat_0012_dagger

Status: **complete**. Finished 2026-10-11 23:36:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 588 updates, 588,432 examples, 22.57s; 26,073 examples/s; mean loss 0.8220.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.60s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.log`.

### 2026-10-11 23:36:41 UTC — combat_0012_dev_act1

Status: **complete**. Finished 2026-10-11 23:36:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 756 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.04.

Total job wall time 2.02s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_act1.log`.

### 2026-10-11 23:36:46 UTC — combat_0012_dev_hard

Status: **complete**. Finished 2026-10-11 23:36:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 27, 'battle_defeat': 5}; 657 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 17.09.

Total job wall time 2.63s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dev_hard.log`.

### 2026-10-11 23:36:51 UTC — combat_0013

Status: **complete**. Finished 2026-10-11 23:39:57 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0012_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 24 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act2.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2876 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_defeat': 630, 'battle_victory': 2238, 'GameOutcome.UNDECIDED': 8}; 70,161 valid transitions.

Mean terminal/cutoff floor 23.62; mean fight HP cost 23.69.

PPO: 302 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.48142733285203576, "preparation_seconds": 0.37298020790331066, "samples": 1384, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.72s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.log`.

### 2026-10-11 23:39:57 UTC — combat_0013_dev_act1

Status: **complete**. Finished 2026-10-11 23:40:02 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 724 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.56.

Total job wall time 1.85s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_act1.log`.

### 2026-10-11 23:40:02 UTC — combat_0013_dev_hard

Status: **complete**. Finished 2026-10-11 23:40:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 28, 'battle_defeat': 4}; 603 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 14.03.

Total job wall time 2.55s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013_dev_hard.log`.

### 2026-10-11 23:40:07 UTC — combat_0014

Status: **complete**. Finished 2026-10-11 23:43:12 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0013.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 25 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act3.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 16 rounds, 1973 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 1701, 'battle_defeat': 264, 'GameOutcome.UNDECIDED': 8}; 46,676 valid transitions.

Mean terminal/cutoff floor 40.38; mean fight HP cost 23.03.

PPO: 183 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.5213650001678616, "preparation_seconds": 0.40662254206836224, "samples": 1184, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.76s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.log`.

### 2026-10-11 23:43:12 UTC — combat_0014_dev_act1

Status: **complete**. Finished 2026-10-11 23:43:17 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 740 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.75.

Total job wall time 1.83s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_act1.log`.

### 2026-10-11 23:43:17 UTC — combat_0014_dev_hard

Status: **complete**. Finished 2026-10-11 23:43:22 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 630 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 16.25.

Total job wall time 2.60s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014_dev_hard.log`.

### 2026-10-11 23:43:22 UTC — combat_0015

Status: **complete**. Finished 2026-10-11 23:46:28 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0014.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 26 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act1.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 45 rounds, 5760 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 5161, 'battle_defeat': 599}; 106,676 valid transitions.

Mean terminal/cutoff floor 7.90; mean fight HP cost 15.32.

PPO: 520 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.6082345000468194, "preparation_seconds": 0.5056994999758899, "samples": 2486, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.56s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015.log`.

### 2026-10-11 23:47:12 UTC — combat_0015_dagger

Status: **complete**. Finished 2026-10-11 23:47:47 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 576 updates, 576,144 examples, 22.36s; 25,770 examples/s; mean loss 0.8119.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.62s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.log`.

### 2026-10-11 23:47:47 UTC — combat_0015_dev_act1

Status: **complete**. Finished 2026-10-11 23:47:52 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 751 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.71.

Total job wall time 2.06s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_act1.log`.

### 2026-10-11 23:47:52 UTC — combat_0015_dev_hard

Status: **complete**. Finished 2026-10-11 23:47:57 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 26, 'battle_defeat': 6}; 703 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 17.03.

Total job wall time 2.78s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dev_hard.log`.

### 2026-10-11 23:47:57 UTC — combat_0016

Status: **complete**. Finished 2026-10-11 23:51:02 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0015_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 27 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act2.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 23 rounds, 2820 non-fault episodes, 26 surfaced faults, 8 truncations; outcomes {'battle_defeat': 575, 'battle_victory': 2237, 'GameOutcome.UNDECIDED': 8}; 63,875 valid transitions.

Mean terminal/cutoff floor 24.34; mean fight HP cost 23.22.

PPO: 262 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.252406416926533, "preparation_seconds": 0.14936604187823832, "samples": 448, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 4, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40eaf0ab49d710d5dc3484836819a1d952435aea009bbdec10073f37c8586819", "task_kind": "battle", "world_seed": 11060}, {"error": "SimulatorError", "excluded_transitions": 4, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40eaf0ab49d710d5dc3484836819a1d952435aea009bbdec10073f37c8586819", "task_kind": "battle", "world_seed": 11060}, {"error": "SimulatorError", "excluded_transitions": 4, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40eaf0ab49d710d5dc3484836819a1d952435aea009bbdec10073f37c8586819", "task_kind": "battle", "world_seed": 11060}, {"error": "SimulatorError", "excluded_transitions": 4, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40eaf0ab49d710d5dc3484836819a1d952435aea009bbdec10073f37c8586819", "task_kind": "battle", "world_seed": 11060}, {"error": "SimulatorError", "excluded_transitions": 3, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "40eaf0ab49d710d5dc3484836819a1d952435aea009bbdec10073f37c8586819", "task_kind": "battle", "world_seed": 11060}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.log`.

### 2026-10-11 23:51:02 UTC — combat_0016_dev_act1

Status: **complete**. Finished 2026-10-11 23:51:07 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 711 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.79.

Total job wall time 1.86s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_act1.log`.

### 2026-10-11 23:51:07 UTC — combat_0016_dev_hard

Status: **complete**. Finished 2026-10-11 23:51:12 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 28, 'battle_defeat': 4}; 595 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 18.03.

Total job wall time 2.68s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016_dev_hard.log`.

### 2026-10-11 23:51:12 UTC — combat_0017

Status: **complete**. Finished 2026-10-11 23:54:18 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0016.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 28 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act3.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 16 rounds, 1959 non-fault episodes, 13 surfaced faults, 8 truncations; outcomes {'battle_victory': 1552, 'battle_defeat': 399, 'GameOutcome.UNDECIDED': 8}; 43,331 valid transitions.

Mean terminal/cutoff floor 41.31; mean fight HP cost 26.67.

PPO: 175 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.4719196669757366, "preparation_seconds": 0.36251366697251797, "samples": 1000, "steps": 0}.

Fault evidence: [{"error": "SimulatorError", "excluded_transitions": 8, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 6, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 7, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 8, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}, {"error": "SimulatorError", "excluded_transitions": 6, "message": "SimulatorError: Simulator reported undefined behavior; trajectory is invalid", "task_id": "e884914b276825d150594e022afc9962ac9361b8543b758381888a45005bacd3", "task_kind": "battle", "world_seed": 11225}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.82s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.log`.

### 2026-10-11 23:54:18 UTC — combat_0017_dev_act1

Status: **complete**. Finished 2026-10-11 23:54:23 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 44, 'battle_defeat': 4}; 725 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 12.48.

Total job wall time 1.87s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_act1.log`.

### 2026-10-11 23:54:23 UTC — combat_0017_dev_hard

Status: **complete**. Finished 2026-10-11 23:54:28 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 30, 'battle_defeat': 2}; 604 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 15.72.

Total job wall time 2.55s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017_dev_hard.log`.

### 2026-10-11 23:54:28 UTC — combat_0018

Status: **complete**. Finished 2026-10-11 23:57:33 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0017.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 29 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act1.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 48 rounds, 6144 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 5720, 'battle_defeat': 424}; 102,148 valid transitions.

Mean terminal/cutoff floor 7.77; mean fight HP cost 14.97.

PPO: 562 optimizer steps; last update {"approx_kl": 0.003946752431147615, "clip_fraction": 0.12975803079704443, "early_stop": false, "elapsed_seconds": 1.0373991660308093, "entropy": 0.7287936260302862, "policy_loss": -0.02921517383462439, "preparation_seconds": 0.4484290408436209, "ratio_mean": 0.9995619853337606, "samples": 2228, "steps": 12, "value_loss": 0.0229271676701804}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.12s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018.log`.

### 2026-10-11 23:58:23 UTC — combat_0018_dagger

Status: **complete**. Finished 2026-10-11 23:58:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.metrics.json --seconds 30 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_aggregate.jsonl --phase combat --learning-rate 0.0001 --fast-replay
```

Training: 573 updates, 573,072 examples, 22.19s; 25,824 examples/s; mean loss 0.7957.

Development imitation: {"accuracy": 0.0, "examples": 0, "loss": 0.0, "menu_accuracy": {}, "menu_counts": {}, "phase_accuracy": {}, "phase_counts": {}, "phase_menu_accuracy": {}, "phase_menu_counts": {}}.

Total job wall time 31.64s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.log`.

### 2026-10-11 23:58:59 UTC — combat_0018_dev_act1

Status: **complete**. Finished 2026-10-11 23:59:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 755 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.44.

Total job wall time 2.04s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_act1.log`.

### 2026-10-11 23:59:04 UTC — combat_0018_dev_hard

Status: **complete**. Finished 2026-10-11 23:59:09 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 27, 'battle_defeat': 5}; 690 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 19.44.

Total job wall time 2.76s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dev_hard.log`.

### 2026-10-11 23:59:09 UTC — combat_0019

Status: **complete**. Finished 2026-10-12 00:02:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 30 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act2.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Metrics at this path belong to a later rerun; this invocation has no completed metrics.

### 2026-10-12 00:05:13 UTC — combat_0019

Status: **complete**. Finished 2026-10-12 00:06:49 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.metrics.json --seconds 92.03934001922607 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0018_dagger.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind battle --seed 30 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/train_act2.jsonl --horizon 256 --teacher-simulations 200 --clip-ratio 0.1 --entropy-coef 0.01 --gamma 1.0 --learning-rate 0.0001 --reward-mode potential --terminal-hp-weight 0.0 --deferred-bootstrap --max-act 3 --fast-replay --warm-optimizer
```

Live collection: 14 rounds, 1701 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 1328, 'battle_defeat': 365, 'GameOutcome.UNDECIDED': 8}; 35,096 valid transitions.

Mean terminal/cutoff floor 23.80; mean fight HP cost 22.65.

PPO: 153 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.3299852078780532, "preparation_seconds": 0.18037412478588521, "samples": 608, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 92.54s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.log`.

### 2026-10-12 00:06:49 UTC — combat_0019_dev_act1

Status: **complete**. Finished 2026-10-12 00:06:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_act1.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_act1.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.pt --device cpu --threads 2 --workers 8 --episodes 48 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --fast-replay
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 43, 'battle_defeat': 5}; 708 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 9.62.

Total job wall time 1.84s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_act1.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_act1.log`.

### 2026-10-12 00:06:54 UTC — combat_0019_dev_hard

Status: **complete**. Finished 2026-10-12 00:06:59 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_hard.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_hard.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019.pt --device cpu --threads 2 --workers 8 --episodes 256 --seed 777 --seconds 90 --task-kind battle --split dev --horizon 256 --simulations 200 --fast-observations --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl --max-act 3 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 646 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 16.84.

Total job wall time 2.68s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_hard.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/combat_0019_dev_hard.log`.

### 2026-10-12 00:06:59 UTC — ooc_specialist_bc

Status: **complete**. Finished 2026-10-12 00:09:04 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.metrics.json --seconds 120 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_none_lr0.0001.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --phase ooc --fast-replay
```

Training: 2,963 updates, 2,821,867 examples, 115.48s; 24,437 examples/s; mean loss 0.2252.

Development imitation: {"accuracy": 0.8079734219269104, "examples": 1505, "loss": 2.3522584224460132, "menu_accuracy": {"choice": 0.7231800766283525, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.8079734219269104}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.7231800766283525, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 122.89s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.log`.

### 2026-10-12 00:09:04 UTC — ooc_scripted_matched_bc

Status: **complete**. Finished 2026-10-12 00:10:40 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_matched_scripted_teacher.jsonl --dev-teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --phase ooc --fast-replay
```

Training: 7,174 updates, 1,836,544 examples, 89.59s; 20,500 examples/s; mean loss 0.0138.

Development imitation: {"accuracy": 0.7787375415282392, "examples": 1505, "loss": 13.282171732246677, "menu_accuracy": {"choice": 0.6810344827586207, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.7787375415282392}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.6810344827586207, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 92.92s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.log`.

### 2026-10-12 00:10:40 UTC — ooc_scripted_matched_dev

Status: **complete**. Finished 2026-10-12 00:10:55 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 12, 'act_clear': 19}; 1,847 valid transitions.

Mean terminal/cutoff floor 15.48; mean fight HP cost 30.81.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 11, "message": "Curriculum step exceeded 10s", "task_id": "1cab65a18600a73714b249dd96c7484ab7ec43720e61acafebac834a506a50f0", "task_kind": "ooc", "world_seed": 10121}].

Total job wall time 13.09s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_dev.log`.

### 2026-10-12 00:10:55 UTC — ooc_lookahead_matched_bc

Status: **complete**. Finished 2026-10-12 00:12:30 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_bc.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_bc.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_specialist_bc.pt --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/ooc_lookahead_v1/teacher.jsonl --dev-teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --phase ooc --fast-replay
```

Training: 7,133 updates, 1,826,048 examples, 89.59s; 20,382 examples/s; mean loss 0.0485.

Development imitation: {"accuracy": 0.6604651162790698, "examples": 1505, "loss": 9.188773100083056, "menu_accuracy": {"choice": 0.5105363984674329, "forced": 1.0}, "menu_counts": {"choice": 1044, "forced": 461}, "phase_accuracy": {"out_of_combat": 0.6604651162790698}, "phase_counts": {"out_of_combat": 1505}, "phase_menu_accuracy": {"out_of_combat": {"choice": 0.5105363984674329, "forced": 1.0}}, "phase_menu_counts": {"out_of_combat": {"choice": 1044, "forced": 461}}}.

Total job wall time 92.67s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_bc.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_bc.log`.

### 2026-10-12 00:12:30 UTC — ooc_lookahead_matched_dev

Status: **complete**. Finished 2026-10-12 00:12:45 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_bc.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 19, 'act_clear': 12}; 1,668 valid transitions.

Mean terminal/cutoff floor 15.00; mean fight HP cost 47.74.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 30, "message": "Curriculum step exceeded 10s", "task_id": "0edc24fd7b228127071b5b5211a130d43842fc26a8491f2f5a9b4fba57b9e790", "task_kind": "ooc", "world_seed": 10297}].

Total job wall time 13.00s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lookahead_matched_dev.log`.

### 2026-10-12 00:12:45 UTC — ooc_warmstart_reference_dev

Status: **complete**. Finished 2026-10-12 00:13:00 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_warmstart_reference_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_warmstart_reference_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 12, 'act_clear': 19}; 1,847 valid transitions.

Mean terminal/cutoff floor 15.48; mean fight HP cost 30.81.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 11, "message": "Curriculum step exceeded 10s", "task_id": "1cab65a18600a73714b249dd96c7484ab7ec43720e61acafebac834a506a50f0", "task_kind": "ooc", "world_seed": 10121}].

Total job wall time 12.47s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_warmstart_reference_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_warmstart_reference_dev.log`.

### 2026-10-12 00:13:00 UTC — ooc_lr0.0001_progress0.05

Status: **complete**. Finished 2026-10-12 00:14:35 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 12 rounds, 764 non-fault episodes, 4 surfaced faults, 0 truncations; outcomes {'act_clear': 514, 'defeat': 250}; 48,665 valid transitions.

Mean terminal/cutoff floor 16.19; mean fight HP cost 25.44.

PPO: 103 optimizer steps; last update {"approx_kl": 0.01027401895055391, "clip_fraction": 0.0096435546875, "early_stop": true, "elapsed_seconds": 1.2670454168692231, "entropy": 0.03932938305661082, "policy_loss": 0.012153716466855258, "preparation_seconds": 0.6987012499012053, "ratio_mean": 0.9988293573260307, "samples": 4151, "steps": 8, "value_loss": 0.07500418275594711}.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 41, "message": "Curriculum step exceeded 10s", "task_id": "5f6a994a5ccf4bffa4943b69a394b29b024597c093bd7f5c49b84f60678144fc", "task_kind": "ooc", "world_seed": 22049}, {"error": "SimulatorTimeout", "excluded_transitions": 41, "message": "Curriculum step exceeded 10s", "task_id": "5f6a994a5ccf4bffa4943b69a394b29b024597c093bd7f5c49b84f60678144fc", "task_kind": "ooc", "world_seed": 22049}, {"error": "SimulatorTimeout", "excluded_transitions": 25, "message": "Curriculum step exceeded 10s", "task_id": "630a38b4fe71a86b7dd51be4c17a3d1680ef84b0a4a441dcd6819e8e20384d2c", "task_kind": "ooc", "world_seed": 22081}, {"error": "SimulatorTimeout", "excluded_transitions": 41, "message": "Curriculum step exceeded 10s", "task_id": "5f6a994a5ccf4bffa4943b69a394b29b024597c093bd7f5c49b84f60678144fc", "task_kind": "ooc", "world_seed": 22049}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.03s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05.log`.

### 2026-10-12 00:14:36 UTC — ooc_lr0.0001_progress0.05_dev

Status: **complete**. Finished 2026-10-12 00:14:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 13, 'act_clear': 19}; 2,000 valid transitions.

Mean terminal/cutoff floor 16.16; mean fight HP cost 32.53.

Total job wall time 2.14s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.05_dev.log`.

### 2026-10-12 00:14:41 UTC — ooc_lr0.0003_progress0.05

Status: **complete**. Finished 2026-10-12 00:16:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 14 rounds, 865 non-fault episodes, 4 surfaced faults, 7 truncations; outcomes {'act_clear': 492, 'defeat': 366, 'GameOutcome.UNDECIDED': 7}; 53,577 valid transitions.

Mean terminal/cutoff floor 15.87; mean fight HP cost 33.15.

PPO: 48 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.378306332975626, "preparation_seconds": 0.3694246250670403, "samples": 2080, "steps": 0}.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 52, "message": "Curriculum step exceeded 10s", "task_id": "e4dc36c8b520193addf181fd365144cca7964c9b414a6496af887de50a3b4a84", "task_kind": "ooc", "world_seed": 22073}, {"error": "SimulatorTimeout", "excluded_transitions": 24, "message": "Curriculum step exceeded 10s", "task_id": "630a38b4fe71a86b7dd51be4c17a3d1680ef84b0a4a441dcd6819e8e20384d2c", "task_kind": "ooc", "world_seed": 22081}, {"error": "SimulatorTimeout", "excluded_transitions": 24, "message": "Curriculum step exceeded 10s", "task_id": "630a38b4fe71a86b7dd51be4c17a3d1680ef84b0a4a441dcd6819e8e20384d2c", "task_kind": "ooc", "world_seed": 22081}, {"error": "SimulatorTimeout", "excluded_transitions": 52, "message": "Curriculum step exceeded 10s", "task_id": "e4dc36c8b520193addf181fd365144cca7964c9b414a6496af887de50a3b4a84", "task_kind": "ooc", "world_seed": 22073}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.49s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.log`.

### 2026-10-12 00:16:16 UTC — ooc_lr0.0003_progress0.05_dev

Status: **complete**. Finished 2026-10-12 00:16:31 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 10, 'act_clear': 21}; 1,886 valid transitions.

Mean terminal/cutoff floor 15.58; mean fight HP cost 25.19.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 63, "message": "Curriculum step exceeded 10s", "task_id": "55870348ff47dfab96580daf375d9c9d62622e3fb2f8b71bb91573b47d0c3d6f", "task_kind": "ooc", "world_seed": 10177}].

Total job wall time 12.50s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05_dev.log`.

### 2026-10-12 00:16:31 UTC — ooc_lr0.001_progress0.05

Status: **complete**. Finished 2026-10-12 00:18:06 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.001 --reward-mode progress --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 21 rounds, 1294 non-fault episodes, 2 surfaced faults, 8 truncations; outcomes {'act_clear': 861, 'defeat': 425, 'GameOutcome.UNDECIDED': 8}; 78,585 valid transitions.

Mean terminal/cutoff floor 16.07; mean fight HP cost 25.57.

PPO: 40 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.2236695410683751, "preparation_seconds": 0.11688275006599724, "samples": 592, "steps": 0}.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 14, "message": "Curriculum step exceeded 10s", "task_id": "aab11778018afcdd0bf97dbae620b7c5e7ee93fa0d457d8152ef2b2b2a520690", "task_kind": "ooc", "world_seed": 22042}, {"error": "SimulatorTimeout", "excluded_transitions": 47, "message": "Curriculum step exceeded 10s", "task_id": "edf15da126f82f6fe17abd201ce6816ed04f81e6f01655ee3ff723c7e4ff2a96", "task_kind": "ooc", "world_seed": 22019}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.40s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05.log`.

### 2026-10-12 00:18:06 UTC — ooc_lr0.001_progress0.05_dev

Status: **complete**. Finished 2026-10-12 00:18:11 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 12, 'act_clear': 20}; 1,961 valid transitions.

Mean terminal/cutoff floor 16.09; mean fight HP cost 29.84.

Total job wall time 2.09s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.001_progress0.05_dev.log`.

### 2026-10-12 00:18:11 UTC — ooc_lr0.0001_sparse

Status: **complete**. Finished 2026-10-12 00:19:47 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode sparse --progress-floor-weight 0.05 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 10 rounds, 635 non-fault episodes, 5 surfaced faults, 0 truncations; outcomes {'act_clear': 413, 'defeat': 222}; 40,018 valid transitions.

Mean terminal/cutoff floor 16.07; mean fight HP cost 27.20.

PPO: 77 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.7858251668512821, "preparation_seconds": 0.675448166904971, "samples": 4020, "steps": 0}.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 16, "message": "Curriculum step exceeded 10s", "task_id": "ec9df317fb06b7ab2f897aadf45425626efd1e1cf3932556e646fe2a6cdae6b2", "task_kind": "ooc", "world_seed": 22074}, {"error": "SimulatorTimeout", "excluded_transitions": 53, "message": "Curriculum step exceeded 10s", "task_id": "e4dc36c8b520193addf181fd365144cca7964c9b414a6496af887de50a3b4a84", "task_kind": "ooc", "world_seed": 22073}, {"error": "SimulatorTimeout", "excluded_transitions": 16, "message": "Curriculum step exceeded 10s", "task_id": "ec9df317fb06b7ab2f897aadf45425626efd1e1cf3932556e646fe2a6cdae6b2", "task_kind": "ooc", "world_seed": 22074}, {"error": "SimulatorTimeout", "excluded_transitions": 40, "message": "Curriculum step exceeded 10s", "task_id": "5f6a994a5ccf4bffa4943b69a394b29b024597c093bd7f5c49b84f60678144fc", "task_kind": "ooc", "world_seed": 22049}, {"error": "SimulatorTimeout", "excluded_transitions": 24, "message": "Curriculum step exceeded 10s", "task_id": "630a38b4fe71a86b7dd51be4c17a3d1680ef84b0a4a441dcd6819e8e20384d2c", "task_kind": "ooc", "world_seed": 22081}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.75s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse.log`.

### 2026-10-12 00:19:47 UTC — ooc_lr0.0001_sparse_dev

Status: **complete**. Finished 2026-10-12 00:20:02 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 14, 'act_clear': 17}; 1,821 valid transitions.

Mean terminal/cutoff floor 15.26; mean fight HP cost 35.45.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 16, "message": "Curriculum step exceeded 10s", "task_id": "1cab65a18600a73714b249dd96c7484ab7ec43720e61acafebac834a506a50f0", "task_kind": "ooc", "world_seed": 10121}].

Total job wall time 12.49s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_sparse_dev.log`.

### 2026-10-12 00:20:02 UTC — ooc_lr0.0001_progress0.02

Status: **complete**. Finished 2026-10-12 00:34:03 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_scripted_matched_bc.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02.metrics.json --task-kind ooc --split train --world-start 22000 --episodes 64 --seed 11 --horizon 512 --max-act 1 --simulations 200 --gamma 1 --clip-ratio .1 --entropy-coef .01 --goal-act-clear --learning-rate 0.0001 --reward-mode progress --progress-floor-weight 0.02 --terminal-hp-weight 0 --device mps --inference-device cpu --workers 8 --threads 2 --batch-size 1024 --updates 100000 --seconds 90.0 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 20 rounds, 1279 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'act_clear': 873, 'defeat': 406}; 81,635 valid transitions.

Mean terminal/cutoff floor 16.25; mean fight HP cost 24.66.

PPO: 177 optimizer steps; last update {"approx_kl": 0.0014614661631640047, "clip_fraction": 0.007339342497289181, "early_stop": true, "elapsed_seconds": 1.0603073749225587, "entropy": 0.035875802394002676, "policy_loss": 0.0010170929599553347, "preparation_seconds": 0.7053694170899689, "ratio_mean": 0.9991263002157211, "samples": 4089, "steps": 4, "value_loss": 0.06753212306648493}.

Fault evidence: [{"error": "SimulatorTimeout", "excluded_transitions": 25, "message": "Curriculum step exceeded 10s", "task_id": "630a38b4fe71a86b7dd51be4c17a3d1680ef84b0a4a441dcd6819e8e20384d2c", "task_kind": "ooc", "world_seed": 22081}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 90.08s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02.log`.

### 2026-10-12 00:34:03 UTC — ooc_lr0.0001_progress0.02_dev

Status: **complete**. Finished 2026-10-12 00:34:08 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02.pt --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02_dev.metrics.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02_dev.metrics.json --task-kind ooc --split dev --world-start 10000 --episodes 32 --seed 777 --horizon 1024 --max-act 1 --simulations 200 --gamma 1 --goal-act-clear --terminal-hp-weight 0 --device cpu --inference-device cpu --workers 8 --threads 2 --seconds 120.0 --fast-observations --fast-replay --deferred-bootstrap
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 17, 'act_clear': 15}; 1,881 valid transitions.

Mean terminal/cutoff floor 15.34; mean fight HP cost 41.69.

Total job wall time 2.07s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0001_progress0.02_dev.log`.

### 2026-10-12 00:34:08 UTC — ooc_initial_dev

Status: **complete**. Finished 2026-10-12 00:34:23 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_initial_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_initial_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 10, 'act_clear': 21}; 1,887 valid transitions.

Mean terminal/cutoff floor 15.58; mean fight HP cost 25.19.

Fault evidence: [{"action_trace_sha256": "4ef93df2cd0cf0919f96a31a4c21b565b4aa81205118c8c5c665ed85be75d048", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 57, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "55870348ff47dfab96580daf375d9c9d62622e3fb2f8b71bb91573b47d0c3d6f", "task_kind": "ooc", "wait_seconds": 10.005190582945943, "world_seed": 10177}].

Total job wall time 12.90s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_initial_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_initial_dev.log`.

### 2026-10-12 00:34:23 UTC — ooc_0000

Status: **complete**. Finished 2026-10-12 00:37:28 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 11 --world-start 20000 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 18 rounds, 2202 non-fault episodes, 6 surfaced faults, 8 truncations; outcomes {'defeat': 848, 'act_clear': 1346, 'GameOutcome.UNDECIDED': 8}; 136,811 valid transitions.

Mean terminal/cutoff floor 16.08; mean fight HP cost 29.98.

PPO: 87 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.453256709035486, "preparation_seconds": 0.3383164170663804, "samples": 1920, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "1ed49b5d88d7286905ab728a5482aaed2a7bb6e7091e54162cf8cdb05118274c", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 23, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "dc5f31f82b9fd64cbe56fa2c6a713ead319aeee2569f72ba5df9ae001f05e53f", "task_kind": "ooc", "wait_seconds": 10.006483291042969, "world_seed": 20127}, {"action_trace_sha256": "1ef60eca7b1a2234551a8fb5fe59669aff33c0912a144b271ee13ab144c4fef5", "attempted_action_index": 1, "error": "SimulatorTimeout", "excluded_transitions": 40, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "963103611c433f5b73b5ccc0a8f1a0dc5e36db67bef72e32fbfef2ad7f36cb06", "task_kind": "ooc", "wait_seconds": 10.00438900012523, "world_seed": 20098}, {"action_trace_sha256": "c6bc8895a74c4f6922a7e401c722c505f73f8f06b766227add87ee50fa997d9b", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 13, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "13b7fd159da2ea9509f44e83a41fef7f6346d4c8d38d69ed1620fa411fc9f0d3", "task_kind": "ooc", "wait_seconds": 10.005408791825175, "world_seed": 20119}, {"action_trace_sha256": "8c6c432fc156d94e9e5a7d400fb7d322df845b62f2ff35bcb3dd5e82d5561d84", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 7, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "edac1598b20d9affb9eab3ca08ce0c85733e41f015c3daaf87f90d1996c16447", "task_kind": "ooc", "wait_seconds": 10.005122874863446, "world_seed": 20110}, {"action_trace_sha256": "ce4486b911646e7cef5b78b472bfb2f41cd4891bc23d50354d5da8db630e6597", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 56, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "e6a13e4cd17d06ed1ee8c15aa7a2c85650a2058cd00a1051a1d5ce3825047713", "task_kind": "ooc", "wait_seconds": 10.004888124996796, "world_seed": 20056}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.61s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.log`.

### 2026-10-12 00:37:28 UTC — ooc_0000_dev

Status: **complete**. Finished 2026-10-12 00:37:33 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 12, 'act_clear': 20}; 1,907 valid transitions.

Mean terminal/cutoff floor 15.41; mean fight HP cost 28.75.

Total job wall time 2.58s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000_dev.log`.

### 2026-10-12 00:37:33 UTC — ooc_0001

Status: **complete**. Finished 2026-10-12 00:40:39 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0000.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 12 --world-start 20128 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 9 rounds, 1032 non-fault episodes, 12 surfaced faults, 8 truncations; outcomes {'defeat': 355, 'act_clear': 669, 'GameOutcome.UNDECIDED': 8}; 63,707 valid transitions.

Mean terminal/cutoff floor 15.98; mean fight HP cost 26.54.

PPO: 39 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.2844787500798702, "preparation_seconds": 0.18037541606463492, "samples": 1008, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "30b99d3a9d619af16ea1166e970deb7e1ddf7b7aa953034063985c7443eedad6", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 5, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5da0e82d309958aa09f6150825f7a77fd600d43a69fad6ec5c749ef6996371c8", "task_kind": "ooc", "wait_seconds": 10.005468709161505, "world_seed": 20192}, {"action_trace_sha256": "6ba1e944d90a05f5a9a1e69aba096470a59d7a4e360bf0b5f7160b5801c823fd", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 11, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "38641529fdf60a050ca2834f720bbf690cb7d5ce3d7b6dc377039655185b5eb2", "task_kind": "ooc", "wait_seconds": 10.005384749965742, "world_seed": 20171}, {"action_trace_sha256": "cc6b3477ff3004da46161da70bf3b5dc79bb16ac15903c286ca2c6101a7b11c4", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 7, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "a110bcf77180bc9b8c5f474c1085e98e67b9a2f5e7636e0011c7635114f76e5c", "task_kind": "ooc", "wait_seconds": 9.991535709006712, "world_seed": 20136}, {"action_trace_sha256": "688ef84da69a8d25260b67612b46ce018df66d300e9f7762b453422a472fbe40", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 56, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "e2450e7b7e421f9866c4d91840cb1943ac5eb4ac54f91eda6dda9933dd0056ca", "task_kind": "ooc", "wait_seconds": 10.005331791937351, "world_seed": 20202}, {"action_trace_sha256": "8745c5bd15002a2b011a29a4545d9ea43e9eed82e2ef9e963730dbf672d35d42", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 7, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "a110bcf77180bc9b8c5f474c1085e98e67b9a2f5e7636e0011c7635114f76e5c", "task_kind": "ooc", "wait_seconds": 10.004680332960561, "world_seed": 20136}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.44s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001.log`.

### 2026-10-12 00:40:39 UTC — ooc_0001_dev

Status: **complete**. Finished 2026-10-12 00:40:54 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001_dev.unused.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001_dev.metrics.json --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001.pt --device cpu --threads 2 --workers 8 --episodes 32 --seed 777 --seconds 90 --task-kind ooc --split dev --horizon 2048 --simulations 200 --fast-observations --world-start 10000 --horizon 1024 --world-start 10000 --fast-replay
```

Live collection: 1 rounds, 31 non-fault episodes, 1 surfaced faults, 0 truncations; outcomes {'defeat': 17, 'act_clear': 14}; 1,866 valid transitions.

Mean terminal/cutoff floor 15.77; mean fight HP cost 42.94.

Fault evidence: [{"action_trace_sha256": "59376f68b8d394e680dea577b9086c8b06de59246f0edace0aa570dc33e1549f", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 59, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "55870348ff47dfab96580daf375d9c9d62622e3fb2f8b71bb91573b47d0c3d6f", "task_kind": "ooc", "wait_seconds": 10.0047323750332, "world_seed": 10177}].

Total job wall time 12.99s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001_dev.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0001_dev.log`.

### 2026-10-12 00:40:54 UTC — ooc_0002

Status: **running**. PID 60003; last supervisor heartbeat 2026-10-12 00:40:54 UTC.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0002.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0002.metrics.json --seconds 180 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --checkpoint /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_lr0.0003_progress0.05.pt --workers 8 --updates 100000 --episodes 128 --fast-observations --inference-device cpu --task-kind ooc --seed 13 --world-start 20256 --horizon 1024 --gamma 1.0 --clip-ratio 0.1 --entropy-coef 0.01 --terminal-hp-weight 0.0 --learning-rate 0.0003 --reward-mode progress --progress-floor-weight 0.05 --deferred-bootstrap --fast-replay --warm-optimizer
```

Live collection: 15 rounds, 1800 non-fault episodes, 8 surfaced faults, 7 truncations; outcomes {'act_clear': 1141, 'defeat': 652, 'GameOutcome.UNDECIDED': 7}; 113,732 valid transitions.

Mean terminal/cutoff floor 16.07; mean fight HP cost 27.93.

PPO: 69 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.23324058298021555, "preparation_seconds": 0.13065512501634657, "samples": 605, "steps": 0}.

Fault evidence: [{"action_trace_sha256": "df95b170dcff8d5925d36ee12cf3ae07692d9e6f3088e00a9242183bcc422c9c", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 18, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5d4cee117ad9f5640c4c840977deb28100b21c149e04de9c14f84775b5748d76", "task_kind": "ooc", "wait_seconds": 10.004850582918152, "world_seed": 20409}, {"action_trace_sha256": "b4e06dabf84faee23645887be153053925683df0fc5deacd9ad52689e670af0b", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 18, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5d4cee117ad9f5640c4c840977deb28100b21c149e04de9c14f84775b5748d76", "task_kind": "ooc", "wait_seconds": 10.00642825011164, "world_seed": 20409}, {"action_trace_sha256": "b4e06dabf84faee23645887be153053925683df0fc5deacd9ad52689e670af0b", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 18, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5d4cee117ad9f5640c4c840977deb28100b21c149e04de9c14f84775b5748d76", "task_kind": "ooc", "wait_seconds": 10.005144875030965, "world_seed": 20409}, {"action_trace_sha256": "b4e06dabf84faee23645887be153053925683df0fc5deacd9ad52689e670af0b", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 18, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5d4cee117ad9f5640c4c840977deb28100b21c149e04de9c14f84775b5748d76", "task_kind": "ooc", "wait_seconds": 10.002754166955128, "world_seed": 20409}, {"action_trace_sha256": "b4e06dabf84faee23645887be153053925683df0fc5deacd9ad52689e670af0b", "attempted_action_index": 0, "error": "SimulatorTimeout", "excluded_transitions": 18, "fault_stage": "step", "message": "Curriculum step exceeded 10s", "task_id": "5d4cee117ad9f5640c4c840977deb28100b21c149e04de9c14f84775b5748d76", "task_kind": "ooc", "wait_seconds": 10.004041124833748, "world_seed": 20409}].

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 181.36s.

Raw metrics: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0002.metrics.json`. Log: `/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/ooc_0002.log`.

## Reproduce or resume

Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and `STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/environment-base/SlayTheSpireAI/external/sts_lightspeed/build`. The supervisor script is `scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints and the original UTC deadline. Follow the decision notes for recovery details.
