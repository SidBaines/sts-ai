#!/usr/bin/env bash
# Build the test-only native fixture used by
# tests/integration/test_canonical_search_sampling.py. It creates scratch
# BattleContext copies with reordered hidden piles; it is never installed as an
# environment or policy API. Run scripts/build_lightspeed.sh first, so the
# source tree holds the patched headers. Prints the fixture path; export it as
# STS_SEARCH_TEST_FIXTURE (and STS_REQUIRE_SEARCH_TEST_FIXTURE=1 to fail
# instead of skipping when it is missing).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STS_DIR="${STS_LIGHTSPEED_SOURCE:-$ROOT/external/sts_lightspeed}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT_DIR="${1:-$STS_DIR/test_fixtures}"
CXX="${CXX:-c++}"

if ! git -C "$STS_DIR" apply --check --reverse "$ROOT/patches/sts_lightspeed_python_api.patch" 2>/dev/null; then
  echo "Simulator source at $STS_DIR does not carry this checkout's patch; run scripts/build_lightspeed.sh first." >&2
  exit 1
fi

SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PY_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
case "$(uname -s)" in
  Darwin) LINK=(-bundle -undefined dynamic_lookup) ;;
  *) LINK=(-shared -fPIC) ;;
esac

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/sts_hidden_pile_fixture$SUFFIX"
"$CXX" -std=gnu++17 -O2 -w "${LINK[@]}" \
  -I"$STS_DIR/include" -I"$STS_DIR/pybind11/include" -I"$PY_INCLUDE" \
  "$ROOT/tests/native/hidden_pile_fixture.cpp" -o "$OUT"
echo "$OUT"
