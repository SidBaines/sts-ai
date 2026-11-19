# Equal-budget refit timing

Run `study.py` once with the research Python interpreter to queue strategy training and the movement probe. Run `run.py` from this directory as a detached, unbuffered Python process, logging to `data/rl_runs/refit_timing_from_9700/monitor.log`. Its queue jobs pause on battery. It checks for `STOP` in the run directory before advancing stages.

Both arms use the same trajectory while they share the same deployed policy. At an accepted refit the next training job resumes the exact strategy, optimizer, normalizer and random state from the pre-refit checkpoint. The run directory contains the six decisions, frozen nomination, separate endpoint panels and final verdict. The [study page](../../research/experiments/2026-11-19-refit-timing.md) fixes the comparisons.
