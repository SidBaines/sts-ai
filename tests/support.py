"""Native-build test gate for the environment release."""
from __future__ import annotations

import importlib.util
import os
from types import ModuleType
import unittest


def simulator_available() -> bool:
    """Whether the independently built native module can be imported."""
    try:
        from sts_env._backend.lightspeed_import import import_lightspeed
        import_lightspeed()
    except Exception:
        return False
    return True


def requires_simulator(test_item):
    """Skip absent native builds locally; require them in release/CI runs."""
    if os.environ.get("STS_REQUIRE_SIMULATOR") == "1" or simulator_available():
        return test_item
    return unittest.skip(
        "sts_lightspeed is not built; run scripts/build_lightspeed.sh. "
        "Set STS_REQUIRE_SIMULATOR=1 to fail instead of skip."
    )(test_item)


def load_search_test_fixture() -> ModuleType:
    """Load the test-only native fixture, honoring the release requirement gate."""
    path = os.environ.get("STS_SEARCH_TEST_FIXTURE")
    if not path:
        if os.environ.get("STS_REQUIRE_SEARCH_TEST_FIXTURE") == "1":
            raise RuntimeError("Native regression fixture is required (STS_SEARCH_TEST_FIXTURE)")
        raise unittest.SkipTest("Build the test-only hidden-pile fixture for this regression")
    spec = importlib.util.spec_from_file_location("sts_hidden_pile_fixture", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native regression fixture from {path!r}")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture
