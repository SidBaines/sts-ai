"""Launch the Interactive Rollout Studio (FastAPI + offline browser UI).

The Studio lets you interactively drive Slay the Spire agents: load a position,
sample N decisions from a chosen method (heuristic / model / user-choice /
first / random), edit the LLM framing/prompt, branch, and cache everything as
canonical rollout JSONL for later analysis. Runs fully offline once a local model
is downloaded.

Setup (once):
    .venv/bin/python -m pip install -e '.[app,llm]'   # app=FastAPI, llm=MLX
    scripts/build_lightspeed.sh                        # build the simulator

Run:
    PYTHONPATH=src .venv/bin/python scripts/interactive_app.py
    # then open http://127.0.0.1:8000

All rollouts/sessions are cached under --cache-dir (default data/interactive),
in the canonical JSONL format, so summarize_rollouts/compute_risk_proxies/
compare_models/visualize_rollout all work on Studio output.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", default="data/interactive", help="where sessions + templates are cached")
    parser.add_argument(
        "--model-backend", default="mlx", choices=["mlx", "vllm"],
        help="default LLM backend for the 'model' method (mlx = local/offline)",
    )
    parser.add_argument(
        "--model", default=None,
        help="default model id/path for the 'model' method (None -> agent default, "
             "e.g. mlx-community/Qwen3-4B-4bit). Point at a local path for offline use.",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - clear message for a missing extra
        raise SystemExit(
            "uvicorn is not installed. Install the app extra: "
            ".venv/bin/python -m pip install -e '.[app]'"
        ) from exc

    from sts_ai.interactive.server import create_app

    app = create_app(cache_dir=args.cache_dir, model_backend=args.model_backend, model_id=args.model)
    print(f"Interactive Rollout Studio on http://{args.host}:{args.port}  (cache: {args.cache_dir})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
