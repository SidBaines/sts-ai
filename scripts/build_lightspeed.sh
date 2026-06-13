#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STS_DIR="$ROOT/external/sts_lightspeed"
PYTHON="$ROOT/.venv/bin/python"
CMAKE="$ROOT/.venv/bin/cmake"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv"
fi

"$PYTHON" -m pip install --upgrade pip cmake

if [[ ! -d "$STS_DIR/.git" ]]; then
  git clone https://github.com/gamerpuppy/sts_lightspeed "$STS_DIR"
fi

cd "$STS_DIR"
git submodule update --init --recursive

if ! grep -q "resolve_current_battle" bindings/slaythespire.cpp; then
  git apply "$ROOT/patches/sts_lightspeed_python_api.patch"
fi

"$CMAKE" -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON" \
  -DPYTHON_EXECUTABLE="$PYTHON" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

"$CMAKE" --build build --target slaythespire -j 8
