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

# Upstream pins pybind11 v2.7.1, which predates Python 3.11 support
# (CPython 3.11 made PyFrameObject opaque, so 2.7.1 fails to compile). Bump to a
# tag that supports Python 3.8-3.13 so the build works on modern interpreters
# (e.g. RunPod's py3.11 images). Re-applied each run since the submodule update
# above resets it to the upstream pin.
( cd pybind11 && git fetch --tags --force --depth 1 origin v2.13.6 && git checkout -q v2.13.6 )

if ! grep -q "resolve_current_battle" bindings/slaythespire.cpp; then
  git apply "$ROOT/patches/sts_lightspeed_python_api.patch"
fi

"$CMAKE" -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON" \
  -DPYTHON_EXECUTABLE="$PYTHON" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

"$CMAKE" --build build --target slaythespire -j 8
