# Log

Temporary running log, newest first. Anything that matters is also in its home file.

- 2026-10-25: Review of the pilot plan written up in [the brainstorm](brainstorms/2026-10-25-getting-the-model-to-learn.md): the model is at trivial-baseline skill, combat rewards are mostly near-ties, and a staged plan starting with decisive out-of-combat states is proposed. Pilot on hold pending Sid's decisions.
- 2026-10-25: Mac speed study (`mac_local_20261025`): MLX conversion matches the original model; laptop generation is about 4× slower than RTX vLLM serving, and checkpointed LoRA updates match the RTX speed. See [the page](experiments/2026-10-25-mac-local-speed.md).
- 2026-10-25: Environment bug found: the out-of-combat text lists potions by count, not slot (7 of 48 prompts show the wrong potions). Needs a fix on main.
- 2026-10-25: Code fixes committed: full-game RL now scores model-invalid games, policy errors surface, held-out worlds are guarded; few-world intervals withheld; baselines added to the speed-check ledger.
