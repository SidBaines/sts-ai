# Non-LLM curriculum experiment progress log

Updated 2026-10-11 21:42:14 UTC. Campaign status: **running**; phase: **architecture_screens**.

Allocation: 2026-10-11 20:08:46 UTC → 2026-10-12 06:30:00 UTC. Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.

This page is generated from the atomic campaign journal and raw metrics. It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. Development results are not final-test results. Simulator faults are separate from ordinary game defeats.

## Current selection

```json
{}
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

Status: **running**. PID 17921; last supervisor heartbeat 2026-10-11 21:42:09 UTC.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py bc --output /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.pt --metrics /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_v1/campaign/bc_large_mean_lr0.001.metrics.json --seconds 90 --device mps --threads 2 --batch-size 1024 --seed 11 --simulations 200 --teacher /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/teacher.jsonl --updates 100000 --size large --context mean --learning-rate 0.001 --fast-replay
```

No completed metrics yet; inspect the condition log for a running job or failure.

## Reproduce or resume

Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and `STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/environment-base/SlayTheSpireAI/external/sts_lightspeed/build`. The supervisor script is `scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints and the original UTC deadline. Follow the decision notes for recovery details.
