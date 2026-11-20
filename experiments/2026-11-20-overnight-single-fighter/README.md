# Overnight single-fighter learning

`config.json` fixes the worlds, checkpoint identities, longer imitation and feedback budgets. `ppo.json` contains the joint-learning settings shared by both arms.

Run `reserve.py` before opening a new study. Submit `advance.py` to `scripts/rl/queue_runner.py` in a separate scheduling lane with the research interpreter, this checkout's `src` on `PYTHONPATH`, and the non-LLM native build on `STS_LIGHTSPEED_BUILD`. The scheduler submits remaining jobs as dependencies finish. It recognizes jobs already present in the queue. The two full-size trainers share one MPS lane. A light verifier runs one update per architecture on separate matched worlds.

The dashboard reads the study heartbeat and the research progress log. `status.json` lives in the run directory alongside the saved first selection, frozen finalist, retention verdict, trained-endpoint comparisons and final verdict. The scheduler files the result and figures locally on completion. Unresolved games stay in the records.

Place `STOP` in the run directory to halt new submissions. Already submitted jobs remain managed by the power-aware queue. Failed attempts retain their outputs. Trainers resume at the last saved update, and other failed jobs have two bounded retries. Explicit queue cancellations are respected. Once the cause of an exhausted retry is repaired, the run records and current queue state identify the remaining stage.

A standby `advance.py --recover-scheduler` job waits in the same scheduling lane. It exits after normal completion or an explicit stop. If the original scheduler exits unexpectedly, the standby reads the saved decisions and the existing queue before submitting any remaining work.
