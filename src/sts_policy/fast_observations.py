"""Optional process-local caches for exact public glossary rendering.

The base files remain unchanged. Cache immutable text-to-status results and
precompile exactly the existing per-name patterns; do not merge alternatives,
since overlapping names such as Minion and Minion Leader must both be returned.
"""
from __future__ import annotations
from functools import lru_cache
import re


def install_status_caches() -> None:
    from sts_env._backend import glossary
    if getattr(glossary, "_curriculum_status_cache", False):
        return
    names = tuple(glossary.STATUS_DB)
    word_patterns = tuple((name, re.compile(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])")) for name in names)
    amount_patterns = tuple((name, re.compile(rf"(?<![A-Za-z]){re.escape(name)}\s+-?\d+")) for name in names)

    @lru_cache(maxsize=8192)
    def words(text: str) -> frozenset[str]:
        return frozenset(name for name, pattern in word_patterns if pattern.search(text))

    @lru_cache(maxsize=8192)
    def amounts(text: str) -> frozenset[str]:
        return frozenset(name for name, pattern in amount_patterns if pattern.search(text))

    # Callers add names to returned sets; never expose the cached immutable set.
    glossary._status_names_in_text = lambda text: set(words(text))
    glossary._scan_status_names = lambda text: set(amounts(text))
    glossary._curriculum_status_cache = True
