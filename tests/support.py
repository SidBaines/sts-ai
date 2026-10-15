"""Native-build test gate for the environment release."""
from __future__ import annotations

import os
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
