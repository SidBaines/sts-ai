# Run the Act 3 boss study

The [experiment design](../../research/experiments/2026-11-18-act3-boss-refits.md) fixes both milestones and all selection rules. From the workstream root, use the research Python interpreter to run `study.py` in this directory once. With the shared queue running, start `monitor.py` as a detached process and redirect its output to `data/rl_runs/joint_a3_from_0f52/monitor.log`. The monitor uses a single-writer lock and resumes from the queue and saved decisions.

The [dashboard](http://127.0.0.1:8795/) follows the continuous strategy-training lineage. The [progress feed](http://127.0.0.1:8795/progress) covers the specialist fits, both gates and final panels. The monitor writes the gate decisions, endpoint selection, confirmation verdict and any qualifying champion export. Create `STOP` inside the run directory to stop new submissions. The power-aware queue controls jobs already submitted.
