# K2-Horizon on Apple silicon with MLX

Speed and correctness checks for running the LLM workstream on a Mac. Results:
`research/experiments/2026-10-25-mac-local-speed.md`.

Set up an environment with the `llm` extra plus mlx-lm. `.venv-mlx/` is gitignored:

```sh
uv venv .venv-mlx --python 3.12
uv pip install -p .venv-mlx/bin/python -e '.[llm]' 'mlx-lm==0.31.3'
```

1. `mlx_model.py --output data/models/k2-horizon-0.9b-mlx` builds an MLX-loadable
   directory. It checks that the pinned Hugging Face config is a plain Llama
   layout with YaRN position scaling, writes a Llama config, and symlinks the
   unchanged BF16 weights and tokenizer from the Hugging Face cache.
2. `parity.py` teacher-forces saved completions through the original model
   (FP32 on CPU, and BF16 on MPS) and through MLX (FP32 and BF16), and compares
   token log-probabilities.
3. `bench_generate.py` times generation on saved prompts with mlx-lm's
   continuous batch generator. It scores answers with `sts_llm.generation.parse_action`
   and the saved teacher table. Each `--case` is
   `name:effort:concurrency:samples:top_p:max_tokens[:limit]`.
   `--one-group-at-a-time` generates each state's samples as their own batch.
   Use it for RL groups: mixed prompt lengths in one batch run about 2–3× slower.
4. `bench_fixed.py` is the equal-work probe: fixed input and output lengths,
   with no stop tokens.
5. `bench_update.py` times LoRA policy-gradient updates on saved completions
   with the same loss as `sts_llm.training`. Use `--grad-checkpoint`: on this
   machine it is both faster and smaller.
6. `collate.py --run data/llm_runs/<run> --output research/figures/mac-local`
   writes the CSVs.

Run with `PYTHONPATH=src`. Run one timing job at a time, because concurrent
GPU work distorts the numbers. The existing PyTorch trainer also runs on the
Mac with `--device mps`, but its updates are about 3.7× slower than MLX.
