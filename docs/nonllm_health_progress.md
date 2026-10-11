# Non-LLM curriculum experiment progress log

Updated 2026-10-11 23:25:13 UTC. Campaign status: **health_complete**; phase: **screens**.

Allocation: 2026-10-11 20:08:46 UTC → 2026-10-12 06:30:00 UTC. Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.

This page is generated from the atomic campaign journal and raw metrics. It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. Development results are not final-test results. Simulator faults are separate from ordinary game defeats.

## Current selection

```json
{}
```

## Experiment timeline

### 2026-10-11 23:03:00 UTC — health_hp0.25

Status: **complete**. Finished 2026-10-11 23:06:05 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.metrics.json --tasks data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/balanced_train_64_per_act.jsonl --task-kind battle --split train --max-act 3 --horizon 256 --episodes 128 --seed 11 --seconds 180 --updates 100000 --batch-size 1024 --threads 2 --workers 8 --device mps --inference-device cpu --gamma 1 --clip-ratio .1 --learning-rate 1e-4 --entropy-coef .01 --reward-mode potential --terminal-hp-weight 0.25 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 19 rounds, 2325 non-fault episodes, 0 surfaced faults, 8 truncations; outcomes {'battle_victory': 1972, 'battle_defeat': 345, 'GameOutcome.UNDECIDED': 8}; 47,690 valid transitions.

Mean terminal/cutoff floor 23.75; mean fight HP cost 22.15.

PPO: 191 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.24438858404755592, "preparation_seconds": 0.14348987489938736, "samples": 408, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.log`.

### 2026-10-11 23:06:05 UTC — health_hp1.0

Status: **complete**. Finished 2026-10-11 23:09:10 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py ppo --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.metrics.json --tasks data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/balanced_train_64_per_act.jsonl --task-kind battle --split train --max-act 3 --horizon 256 --episodes 128 --seed 11 --seconds 180 --updates 100000 --batch-size 1024 --threads 2 --workers 8 --device mps --inference-device cpu --gamma 1 --clip-ratio .1 --learning-rate 1e-4 --entropy-coef .01 --reward-mode potential --terminal-hp-weight 1.0 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --warm-optimizer
```

Live collection: 19 rounds, 2339 non-fault episodes, 0 surfaced faults, 7 truncations; outcomes {'battle_victory': 1980, 'battle_defeat': 352, 'GameOutcome.UNDECIDED': 7}; 47,722 valid transitions.

Mean terminal/cutoff floor 23.75; mean fight HP cost 21.78.

PPO: 160 optimizer steps; last update {"early_stop": true, "elapsed_seconds": 0.334606165997684, "preparation_seconds": 0.2304019161965698, "samples": 720, "steps": 0}.

Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.

Total job wall time 180.47s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.log`.

### 2026-10-11 23:09:11 UTC — reference_act1_48

Status: **complete**. Finished 2026-10-11 23:09:16 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_act1_48.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_act1_48.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt --task-kind battle --episodes 48 --max-act 1 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 41, 'battle_defeat': 7}; 727 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.15.

Total job wall time 1.88s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_act1_48.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_act1_48.log`.

### 2026-10-11 23:09:16 UTC — reference_expanded_hard32

Status: **complete**. Finished 2026-10-11 23:09:21 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_expanded_hard32.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_expanded_hard32.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt --task-kind battle --episodes 32 --max-act 3 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 29, 'battle_defeat': 3}; 657 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 20.56.

Total job wall time 3.12s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_expanded_hard32.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_expanded_hard32.log`.

### 2026-10-11 23:09:21 UTC — reference_composite_act1_24

Status: **complete**. Finished 2026-10-11 23:09:26 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_composite_act1_24.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_composite_act1_24.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --combat-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference.pt --ooc-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/ooc_reference.pt --task-kind run --world-start 10000 --episodes 24 --max-act 1 --horizon 2048
```

Live collection: 1 rounds, 24 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 19, 'act_clear': 5}; 4,709 valid transitions.

Mean terminal/cutoff floor 13.75; mean fight HP cost 63.17.

Total job wall time 3.42s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_composite_act1_24.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/reference_composite_act1_24.log`.

### 2026-10-11 23:09:26 UTC — hp0.25_act1_48

Status: **complete**. Finished 2026-10-11 23:09:31 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_act1_48.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_act1_48.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.pt --task-kind battle --episodes 48 --max-act 1 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 688 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 10.71.

Total job wall time 1.81s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_act1_48.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_act1_48.log`.

### 2026-10-11 23:09:31 UTC — hp0.25_expanded_hard32

Status: **complete**. Finished 2026-10-11 23:09:36 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_expanded_hard32.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_expanded_hard32.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.pt --task-kind battle --episodes 32 --max-act 3 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 30, 'battle_defeat': 2}; 676 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 18.09.

Total job wall time 3.38s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_expanded_hard32.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_expanded_hard32.log`.

### 2026-10-11 23:09:36 UTC — hp0.25_composite_act1_24

Status: **complete**. Finished 2026-10-11 23:09:41 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_composite_act1_24.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_composite_act1_24.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --combat-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp0.25.pt --ooc-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/ooc_reference.pt --task-kind run --world-start 10000 --episodes 24 --max-act 1 --horizon 2048
```

Live collection: 1 rounds, 24 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 22, 'act_clear': 2}; 4,559 valid transitions.

Mean terminal/cutoff floor 13.54; mean fight HP cost 73.33.

Total job wall time 3.29s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_composite_act1_24.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp0.25_composite_act1_24.log`.

### 2026-10-11 23:09:41 UTC — hp1.0_act1_48

Status: **complete**. Finished 2026-10-11 23:09:46 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_act1_48.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_act1_48.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.pt --task-kind battle --episodes 48 --max-act 1 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_seed10000/battles.jsonl
```

Live collection: 1 rounds, 48 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 42, 'battle_defeat': 6}; 684 valid transitions.

Mean terminal/cutoff floor 6.88; mean fight HP cost 11.08.

Total job wall time 1.80s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_act1_48.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_act1_48.log`.

### 2026-10-11 23:09:46 UTC — hp1.0_expanded_hard32

Status: **complete**. Finished 2026-10-11 23:09:51 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_expanded_hard32.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_expanded_hard32.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.pt --task-kind battle --episodes 32 --max-act 3 --horizon 256 --tasks /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/panels_expanded/dev_hard.jsonl
```

Live collection: 1 rounds, 32 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'battle_victory': 30, 'battle_defeat': 2}; 643 valid transitions.

Mean terminal/cutoff floor 31.31; mean fight HP cost 18.53.

Total job wall time 3.18s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_expanded_hard32.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_expanded_hard32.log`.

### 2026-10-11 23:09:51 UTC — hp1.0_composite_act1_24

Status: **complete**. Finished 2026-10-11 23:09:56 UTC; exit code 0.

Exact command:

```text
/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/.venv/bin/python /Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/scripts/train_nonllm.py evaluate --output data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_composite_act1_24.metrics.unused.pt --metrics data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_composite_act1_24.metrics.json --split dev --seed 777 --device cpu --inference-device cpu --workers 8 --threads 2 --simulations 200 --fast-observations --fast-replay --deferred-bootstrap --seconds 90 --combat-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/health_hp1.0.pt --ooc-checkpoint data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/ooc_reference.pt --task-kind run --world-start 10000 --episodes 24 --max-act 1 --horizon 2048
```

Live collection: 1 rounds, 24 non-fault episodes, 0 surfaced faults, 0 truncations; outcomes {'defeat': 21, 'act_clear': 3}; 4,322 valid transitions.

Mean terminal/cutoff floor 13.25; mean fight HP cost 69.92.

Total job wall time 3.21s.

Raw metrics: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_composite_act1_24.metrics.json`. Log: `data/nonllm_v1/campaign/health_ablation_4b0ab67c028d/hp1.0_composite_act1_24.log`.

## Reproduce or resume

Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and `STS_LIGHTSPEED_BUILD=/Users/sidbaines/.codex/worktrees/environment-base/SlayTheSpireAI/external/sts_lightspeed/build`. The supervisor script is `scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints and the original UTC deadline. Follow the decision notes for recovery details.
