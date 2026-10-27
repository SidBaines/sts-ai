#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec .venv-mlx/bin/python -u experiments/2026-10-27-learning-trials/run_trials.py "${1:?run name}"
