# Status

Serving benchmark running on 23 October 2026. No training is running.

- Pod: llm-throughput-20261023, qccxnwlrjo151f; one A40, Secure Cloud, $0.59/hour.
- Maximum lifetime: two hours, termination at 17:16 BST on 23 October.
- Check: `ssh runpod-llm-throughput-20261023 'tail -30 /workspace/llm_throughput/logs/server.log'`.
- Stop the benchmark between cases by creating `/workspace/llm_throughput/STOP`.
- Outputs are copied locally to run k2_throughput_20261023 as cases finish.
- Next: compare serving settings and reasoning levels; choose a training pilot.
