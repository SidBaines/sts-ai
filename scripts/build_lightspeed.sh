#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STS_DIR="${STS_LIGHTSPEED_SOURCE:-$ROOT/external/sts_lightspeed}"
BUILD_DIR="${STS_LIGHTSPEED_BUILD:-$STS_DIR/build}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PIN="7476a81954020087da31d41d16fddf475746ec2d"
PYBIND_TAG="v2.13.6"
PATCH="$ROOT/patches/sts_lightspeed_python_api.patch"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv"
fi

if ! "$PYTHON" -m cmake --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install 'cmake==4.1.3'
fi

if [[ ! -d "$STS_DIR/.git" ]]; then
  mkdir -p "$(dirname "$STS_DIR")"
  git clone https://github.com/gamerpuppy/sts_lightspeed "$STS_DIR"
fi
if [[ "$(git -C "$STS_DIR" rev-parse HEAD)" != "$PIN" ]]; then
  if [[ -n "$(git -C "$STS_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "Source checkout differs from the pinned commit and has local changes; use a fresh STS_LIGHTSPEED_SOURCE." >&2
    exit 1
  fi
  git -C "$STS_DIR" fetch origin "$PIN"
  git -C "$STS_DIR" checkout --detach "$PIN"
fi
if git -C "$STS_DIR" apply --check "$PATCH" 2>/dev/null; then
  git -C "$STS_DIR" apply "$PATCH"
elif ! git -C "$STS_DIR" apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "Simulator source has a different patch; use a fresh STS_LIGHTSPEED_SOURCE." >&2
  exit 1
fi
# Upstream uses a relative URL; explicitly resolve it for local/fork clones.
git -C "$STS_DIR" config submodule.pybind11.url https://github.com/pybind/pybind11
git -C "$STS_DIR" submodule update --init --recursive --depth 1
git -C "$STS_DIR/pybind11" fetch --depth 1 origin "refs/tags/$PYBIND_TAG:refs/tags/$PYBIND_TAG"
git -C "$STS_DIR/pybind11" checkout --detach "$PYBIND_TAG"
"$PYTHON" "$ROOT/scripts/native_build_manifest.py" check-source "$STS_DIR" "$ROOT"

"$PYTHON" -m cmake -S "$STS_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON" \
  -DPYTHON_EXECUTABLE="$PYTHON" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

"$PYTHON" -m cmake --build "$BUILD_DIR" --target slaythespire -j "${STS_BUILD_JOBS:-8}"
"$PYTHON" "$ROOT/scripts/native_build_manifest.py" write "$STS_DIR" "$ROOT" "$BUILD_DIR"
echo "Verified simulator ready: $BUILD_DIR"
