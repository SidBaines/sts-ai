# Status

Nothing running. Preparatory goal completed on 23 October 2026.

- Branch: codex/llm-throughput; latest main merged, core equals env-v0.1.5.1.
- Run: k2_throughput_20261023; raw results, checkpoints and launch records local.
- Serving, prefix-cache, actual HF generation and optimizer checks are complete.
- Teacher-RL mechanics and checkpoint continuation: 16/16 valid responses,
  four nonzero updates; no learning-strength claim.
- Trained adapter: three real CLI decisions across both phases, replay verified.
- LLM library: 52 tests pass with Torch. Environment suite: 247 pass, 19 skip.
- Full-game outcome RL is implemented and unit-tested; no full-game learning run.
- RTX pod ra7v5afx03jcji stopped at 16:28 UTC; disk retained, deletion timer cancelled.
- Earlier A40 qccxnwlrjo151f returns not found at final lookup. Its setup logs
  are local; no comparable timing completed, and deletion cause is unverified.
- Results are in the experiment page and complete CSVs. No main study is queued.
- Next: agree on effort, backend/dtype, output cap and loss normalization.
