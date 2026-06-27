"""Shared test helpers and optional-dependency gates.

The fast unit tier (``tests/unit``) is pure Python and must run without the
native simulator. The integration tier (``tests/integration``) exercises the
built ``sts_lightspeed`` module and optional GPU backends via explicit gates.
"""
from __future__ import annotations

import importlib.util
import os
import unittest


def _module_installed(name: str) -> bool:
    """True if ``name`` is importable, checked WITHOUT importing it.

    Using ``find_spec`` (not a real ``import``) matters during test collection:
    importing e.g. ``mlx_lm`` eagerly pulls ``transformers`` into ``sys.modules``,
    and an unrelated test's ``assertWarns`` (which scans every module's
    ``__warningregistry__``) then trips transformers' lazy ``__getattr__`` into a
    torch/torchvision import that may be absent. find_spec avoids that entirely.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


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


def vllm_available() -> bool:
    """True if the optional ``vllm`` package is installed (not imported)."""
    return _module_installed("vllm")


_REQUIRE_VLLM = os.environ.get("STS_REQUIRE_VLLM") == "1"
_VLLM_SKIP_REASON = (
    "vLLM is not installed (CUDA-only); install with `.[vllm]`. "
    "Set STS_REQUIRE_VLLM=1 to fail instead of skip."
)


def requires_vllm(test_item):
    """Gate a test method or ``TestCase`` on the optional vLLM backend."""
    if _REQUIRE_VLLM or vllm_available():
        return test_item
    return unittest.skip(_VLLM_SKIP_REASON)(test_item)


def mlx_available() -> bool:
    """True if the optional ``mlx_lm`` package is installed (not imported)."""
    return _module_installed("mlx_lm")


_REQUIRE_MLX = os.environ.get("STS_REQUIRE_MLX") == "1"
_MLX_SKIP_REASON = (
    "MLX is not installed (Apple Silicon-only); install with `.[train-mlx]`. "
    "Set STS_REQUIRE_MLX=1 to fail instead of skip."
)


def requires_mlx(test_item):
    """Gate a test method or ``TestCase`` on the optional MLX backend."""
    if _REQUIRE_MLX or mlx_available():
        return test_item
    return unittest.skip(_MLX_SKIP_REASON)(test_item)


def fastapi_available() -> bool:
    """True if the optional ``fastapi`` package is installed (not imported)."""
    return _module_installed("fastapi")


_REQUIRE_FASTAPI = os.environ.get("STS_REQUIRE_FASTAPI") == "1"
_FASTAPI_SKIP_REASON = (
    "fastapi is not installed; install with `.[app]`. "
    "Set STS_REQUIRE_FASTAPI=1 to fail instead of skip."
)


def requires_fastapi(test_item):
    """Gate a test method or ``TestCase`` on the optional FastAPI app backend."""
    if _REQUIRE_FASTAPI or fastapi_available():
        return test_item
    return unittest.skip(_FASTAPI_SKIP_REASON)(test_item)


def torch_available() -> bool:
    """True if the optional ``torch`` package is installed (not imported)."""
    return _module_installed("torch")


_REQUIRE_TORCH = os.environ.get("STS_REQUIRE_TORCH") == "1"
_TORCH_SKIP_REASON = (
    "torch is not installed (CUDA training-only); install with `.[train-cuda]`. "
    "Set STS_REQUIRE_TORCH=1 to fail instead of skip."
)


def requires_torch(test_item):
    """Gate a test method or ``TestCase`` on the optional torch backend."""
    if _REQUIRE_TORCH or torch_available():
        return test_item
    return unittest.skip(_TORCH_SKIP_REASON)(test_item)
