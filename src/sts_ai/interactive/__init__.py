"""Interactive Rollout Studio.

A FastAPI backend + offline browser UI for interactively driving Slay the Spire
agents: load a position, sample N decisions from a chosen method (heuristic /
model / user-choice / first / random), edit the LLM framing/prompt, branch to
explore alternatives, and cache everything as canonical rollout JSONL.

The core (`replay`, `templates`, `store`, `session`) is pure-Python and
UI-agnostic so it stays unit-testable; `server` is the thin FastAPI shell and
lazily imports FastAPI so this package imports without the optional `app` extra.

See `CLAUDE.md` in this directory for the area-specific gotchas (replay-only
branching, the llm-combat unsupported-state caveat, schema reuse).
"""
from __future__ import annotations
