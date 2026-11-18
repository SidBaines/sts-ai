# Run the Mac code benchmarks

The [research page](../../research/experiments/2026-11-18-mac-performance.md) gives the results and exact baseline commits. Use those commits when recreating historical baselines. Run from the performance workstream checkout. Use disposable output directories. The champion, teacher binary and LLM adapter are read-only inputs.

## Whole games and trainer

`compare_paths.py` runs fresh processes for each arm. Give it the original environment checkout, the environment candidate, the workstream containing this study, the research Python interpreter, the native build and the directory containing the champions and training runs. These paths are listed in the active workstream's research/status.md and in the local archive inventory.

```sh
python3 experiments/2026-11-18-mac-performance/compare_paths.py \
  --baseline "$ORIGINAL_ENV" --candidate "$CANDIDATE_ENV" \
  --workstream "$STUDY" --original-workstream "$ORIGINAL_NONLLM" \
  --python "$RL_PYTHON" --native "$RL_BUILD" --run-root "$RUN_ROOT" \
  --suite train --production-shape --combined --device mps --order ba \
  --out data/mac_performance/full_mps
```

Both arms start from the promoted champion. This is a fresh disposable iteration with the saved learning recipe. The production shape uses 3,072 games, 12 collectors, 384 game slots, chunks of 128, eight update threads, three PPO epochs, minibatches of 4,096, microbatches of 1,024 and 1,024-world evaluations before and after training. The endpoint forces an evaluation even though the ordinary run evaluates every fourth iteration. The candidate arm also moves diagnostic calculations to Metal. Use `check_train.py REFERENCE CANDIDATE` with the research interpreter to compare the resulting checkpoints and records.

Use `--suite env` for public stepping, `--suite eval` for the champion evaluator, and `--suite collect` for a single sampled collector. `--order abba` or `baab` alternates sources. With `--suite train --compare-diagnostics --device mps`, both arms use the candidate environment and vary only diagnostic placement. The smaller trainer probes set `--games` and `--collectors`. The production benchmark uses normal process isolation for collectors and access to macOS shared memory and Metal. Compare timings on AC power with other intensive workloads idle.

## Component probes

Use the research interpreter with `PYTHONPATH` pointing at the intended environment and workstream sources, and `STS_LIGHTSPEED_BUILD` pointing at the matching build. Each script accepts `--help`.

- `profile_policy.py` profiles real evaluation or collection. Its output includes a hash of game records. The profiler adds substantial overhead. Use separate unprofiled runs for speed comparisons.
- `bench_parallel_eval.py` exercises the production multiprocessing evaluator and saves one outcome and trajectory signature per world.
- `bench_encoder.py` samples public champion decisions and checks every encoded array and coverage counter.
- `bench_search.py` uses the recorded champion game's action prefix to reach specified fights. The teacher build must support root search statistics. It compares vote and detailed teacher outputs and verifies the original fight state after each query.
- `prepare_native.py` creates two independent copies of the pinned teacher source, including its local Git dependencies. Supply a fresh output path and a dedicated CMake 3.31.6 tools virtualenv. The candidate applies `native-playout.patch` and regenerates the workstream overlay. In each generated project run `BUILD_DIR=build_perf bash scripts/rl/build_native_rl.sh`. Benchmark each new binary in its own fresh Python process.
- `bench_records.py` compares three serializers on a real compressed training collection and checks the decompressed bytes.
- `bench_summary.py` compares the original summary writer and a bounded-memory writer. Use the original workstream source at `817d78f` for this probe. Run each arm in a separate process so the peak-resident-memory counter resets.
- `bench_mlx.py` measures the saved model on real prompts and sampled output tokens. `bench_mlx_prefix.py` tests reuse of a shared prompt prefix during sampled generation. Run both with the LLM workstream's MLX interpreter, `HF_HUB_OFFLINE=1` and `--worktree` pointing at its checkout.
