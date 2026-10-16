#!/usr/bin/env bash
# Build the sts_rl native module: pinned upstream + versioned environment patch
# (patches/sts_lightspeed_python_api.patch) + patches/sts_lightspeed_rl_panel.patch.
# Uses the existing isolated .venv-native for cmake; never provisions dependencies.
# Output: external/sts_lightspeed_rl/${BUILD_DIR:-build013_clean} (set STS_LIGHTSPEED_BUILD).
# Every build requires a new or empty BUILD_DIR. Reused outputs can contain stale
# objects even when sources verify; overwriting a loaded dylib can kill its users.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${STS_RL_SOURCE:-$ROOT/external/sts_lightspeed_rl}"
B="$S/${BUILD_DIR:-build013_clean}"
PY="${PYTHON:-/Users/sidbaines/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12}"
HELPER="$ROOT/scripts/rl/native_build_manifest.py"
[[ -x "$ROOT/.venv-native/bin/python" ]] || { echo "Missing .venv-native build-tools runtime; provision it separately." >&2; exit 1; }
# Preserve all prior outputs, including dotfiles. Fail closed on unreadable paths.
if [[ -e "$B" || -L "$B" ]]; then
  if [[ ! -d "$B" ]] || ! BUILD_CONTENTS="$(find "$B/." -mindepth 1 -maxdepth 1 -print -quit)" || [[ -n "$BUILD_CONTENTS" ]]; then
    echo "Build output is not an empty directory: $B; choose fresh BUILD_DIR." >&2
    exit 1
  fi
fi
PIN="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream_commit"])' "$ROOT/src/sts_env/_backend/native_manifest.json")"
if [[ ! -e "$S" ]]; then
  git clone -q https://github.com/gamerpuppy/sts_lightspeed "$S"
  git -C "$S" checkout -q --detach "$PIN"
  git -C "$S" apply "$ROOT/patches/sts_lightspeed_python_api.patch"
  git -C "$S" config submodule.pybind11.url https://github.com/pybind/pybind11
  git -C "$S" submodule update -q --init --recursive --depth 1
  git -C "$S/pybind11" fetch -q --depth 1 origin "refs/tags/v2.13.6:refs/tags/v2.13.6"
  git -C "$S/pybind11" checkout -q --detach v2.13.6
  git -C "$S" apply "$ROOT/patches/sts_lightspeed_rl_panel.patch"
fi
# Existing sources are read-only: an exact reconstruction must match before any
# configure/build action. Keep this identity outside the source/build outputs.
SOURCE_CHECK="$(mktemp "${TMPDIR:-/tmp}/sts-rl-source-check.XXXXXX")"
trap 'rm -f "$SOURCE_CHECK"' EXIT
"$PY" "$HELPER" check-source "$S" "$ROOT" "$B" > "$SOURCE_CHECK"
"$ROOT/.venv-native/bin/python" -m cmake -S "$S" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
"$ROOT/.venv-native/bin/python" -m cmake --build "$B" --target slaythespire -j "${STS_BUILD_JOBS:-6}"
# Publish only immediately after this successful fresh build; the helper must
# match the precheck to its postcheck before atomically publishing provenance.
"$PY" "$HELPER" write "$S" "$ROOT" "$B" --checked-source "$SOURCE_CHECK"
