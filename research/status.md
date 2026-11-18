# Status

Active study: [paired PPO learning rates](experiments/2026-11-18-paired-ppo-rates.md). Sid authorized training and both evaluation phases on 18 November. The [progress log](log.md) records milestones.

The [live training dashboard](http://127.0.0.1:8795/) shows both arms and their development evaluations.

The control has completed all 24 iterations. The challenger is running on the same verified configuration and training-world schedule. The power-aware queue pauses on battery. Each trainer saves `latest.pt` after every training iteration and a frozen checkpoint every four iterations. The starting champion is `champion_4058cbf0e271.pt`.

- Checkout: `~/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-large-batch`.
- Check jobs with `python3 scripts/rl/queue_runner.py status` from this checkout. The run is `data/rl_runs/ppo_fresh_worlds_from_4058`; each arm writes `log.jsonl` there.
- To stop automatic stage submission, create `STOP` in the run directory. To stop an active job use `python3 scripts/rl/queue_runner.py cancel JOB_NAME`.
