"""Shared test helpers.

The fast unit tier (``tests/unit``) is pure Python and must run without the
native simulator. The integration tier (``tests/integration``) exercises the
built ``sts_lightspeed`` module and is gated with :func:`requires_simulator`.
"""
from __future__ import annotations

import os
import unittest


def simulator_available() -> bool:
    """True if the built ``sts_lightspeed`` module can be imported."""
    try:
        from sts_ai.lightspeed_import import import_lightspeed

        import_lightspeed()
    except Exception:
        return False
    return True


_REQUIRE = os.environ.get("STS_REQUIRE_SIMULATOR") == "1"
_SKIP_REASON = (
    "sts_lightspeed is not built; run scripts/build_lightspeed.sh. "
    "Set STS_REQUIRE_SIMULATOR=1 to fail instead of skip."
)


def requires_simulator(test_item):
    """Gate a test method or ``TestCase`` on the built simulator.

    Default behaviour is to **skip** with a clear reason when the build is
    missing, so the unit suite (and a partial checkout) stays green. Set
    ``STS_REQUIRE_SIMULATOR=1`` (e.g. in CI) to instead let the test run and
    fail loudly, so a missing build cannot silently pass the integration
    gate — i.e. fail-closed, matching the repo's simulator-fault policy.
    """
    if _REQUIRE or simulator_available():
        return test_item
    return unittest.skip(_SKIP_REASON)(test_item)
