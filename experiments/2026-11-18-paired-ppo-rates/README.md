# Paired learning rates

The [experiment page](../../research/experiments/2026-11-18-paired-ppo-rates.md) fixes the training and evaluation protocol. The shared queue and standard trainer own checkpointing, pausing and fault records.

`launch.py` submits the control and challenger. It reads `control.json`, verifies the parent checkpoint, and changes only the output path and learning rate. A verified partial run can resume with `--resume-arm ARM --retry-name UNIQUE_NAME`.

`monitor.py` writes local run events and the human-readable progress log. It verifies training task hashes against `expected_worlds.json`, freezes the endpoints, submits the two evaluation stages, and saves paired results. Run it with the research interpreter and `PYTHONPATH=src`. A `STOP` file under the run directory stops new stage submission. Existing queued jobs can be cancelled individually using the queue runner.

The dashboard at http://127.0.0.1:8795/ serves the current training logs and links to the progress timeline.
