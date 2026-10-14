#!/usr/bin/env bash
# Build the sts_rl native module: pinned upstream + versioned environment patch
# (patches/sts_lightspeed_python_api.patch) + patches/sts_lightspeed_rl_panel.patch.
# Uses an isolated .venv-native for cmake so no frozen venv is modified.
# Output: external/sts_lightspeed_rl/${BUILD_DIR:-build} (point STS_LIGHTSPEED_BUILD at it).
# Set BUILD_DIR to build beside a module that running jobs have loaded: overwriting a
# loaded dylib in place on macOS can kill those processes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="$ROOT/external/sts_lightspeed_rl"
B="$S/${BUILD_DIR:-build013_clean}"
PY="${PYTHON:-/Users/sidbaines/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12}"
PIN=7476a81954020087da31d41d16fddf475746ec2d
[[ -x "$ROOT/.venv-native/bin/python" ]] || { "$PY" -m venv "$ROOT/.venv-native"; "$ROOT/.venv-native/bin/python" -m pip -q install 'cmake==3.31.6'; }
if [[ ! -d "$S/.git" ]]; then
  git clone -q https://github.com/gamerpuppy/sts_lightspeed "$S"
  git -C "$S" checkout -q --detach "$PIN"
  git -C "$S" apply "$ROOT/patches/sts_lightspeed_python_api.patch"
  git -C "$S" config submodule.pybind11.url https://github.com/pybind/pybind11
  git -C "$S" submodule update -q --init --recursive --depth 1
  git -C "$S/pybind11" fetch -q --depth 1 origin "refs/tags/v2.13.6:refs/tags/v2.13.6"
  git -C "$S/pybind11" checkout -q --detach v2.13.6
  git -C "$S" add -A bindings include src
  git -C "$S" -c user.email=rl@local -c user.name=sts_rl commit -q -m "versioned environment patch"
  git -C "$S" apply "$ROOT/patches/sts_lightspeed_rl_panel.patch"
fi
"$ROOT/.venv-native/bin/python" -m cmake -S "$S" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
"$ROOT/.venv-native/bin/python" -m cmake --build "$B" --target slaythespire -j "${STS_BUILD_JOBS:-6}"
"$PY" - "$ROOT" "$S" "$B" <<'PYEOF'
import hashlib, json, subprocess, sys, sysconfig
from pathlib import Path
root, s, b = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
expected = json.loads((root / "src/sts_env/_backend/native_manifest.json").read_text())
name = "slaythespire" + sysconfig.get_config_var("EXT_SUFFIX")
manifest = {**expected,
            "rl_panel_patch_sha256": hashlib.sha256((root / "patches/sts_lightspeed_rl_panel.patch").read_bytes()).hexdigest(),
            "module_file": name, "module_sha256": hashlib.sha256((b / name).read_bytes()).hexdigest(),
            "python_cache_tag": sys.implementation.cache_tag, "python_version": sys.version.split()[0],
            "platform": sysconfig.get_platform(),
            "json_commit": subprocess.check_output(["git", "-C", str(s / "json"), "rev-parse", "HEAD"], text=True).strip(),
            "cmake_version": "cmake version 3.31.6", "variant": "sts_rl_panel"}
(b / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest written", manifest["module_sha256"][:16])
PYEOF
