from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalAction:
    """Internal action identity; this is not the research trace schema."""

    index: int
    bits: int
    description: str
