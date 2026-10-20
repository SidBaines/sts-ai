"""Copies for the acyclic dict/list/scalar trees in public state and metadata."""
from __future__ import annotations

from typing import Any


def copy_tree(value: Any) -> Any:
    """Copy mutable JSON containers; their scalar leaves are immutable.

    This deliberately does not implement deepcopy's arbitrary-object protocol or
    memoization. Native public serializers and JSON build manifests return trees,
    with no cycles or mutable leaves other than dictionaries and lists.
    """
    if isinstance(value, dict):
        return {key: copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_tree(item) for item in value]
    return value
