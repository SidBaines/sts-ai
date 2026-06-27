"""On-disk cache for interactive sessions.

One directory per session under ``root`` (default ``data/interactive``):

    <session_id>/
        decisions.jsonl   canonical DecisionRecord dicts (schemas.py contract,
                          UNCHANGED -> summarize_rollouts / compute_risk_proxies /
                          compare_models / visualize_rollout all work as-is)
        meta.json         RolloutMeta dict (build_rollout_meta), optional
        session.json      Studio reconstruction data: lineage, seed/config,
                          per-decision method, current framing/template, status

This module is pure I/O over dicts (no simulator), so it unit-tests with
synthetic records. `RolloutSession` (session.py) owns building these dicts.
Branching/editing rewrites ``decisions.jsonl`` wholesale each save — sessions are
short, so a full rewrite is cheap and keeps the file consistent with history.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SESSION_JSON = "session.json"
DECISIONS_JSONL = "decisions.jsonl"
META_JSON = "meta.json"


@dataclass
class StoredSession:
    """The ``session.json`` payload — everything needed to rebuild + branch a
    session, beyond the canonical decision stream."""

    session_id: str
    label: str = ""
    # lineage
    parent_id: str | None = None
    branch_point: int | None = None  # decision_index in the parent this forked at
    # env identity / config (replay inputs)
    world_seed: int = 0
    ascension: int = 0
    combat_control: str = "llm"
    max_act: int = 3
    battle_simulations: int = 2000
    # agent / prompt config
    model_backend: str = "mlx"  # "mlx" (local, offline) | "vllm" (CUDA)
    model_id: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    thinking: bool = False
    framing: str = ""
    prompt_template: str | None = None  # None => default render_action_prompt path
    use_advanced_template: bool = False
    # per-committed-decision provenance (parallel to decisions.jsonl)
    methods: list[str] = field(default_factory=list)
    # lifecycle
    status: str = "active"  # "active" | "terminal" | "error"
    stopped_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def n_decisions(self) -> int:
        return len(self.methods)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_decisions"] = self.n_decisions
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredSession":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary(self) -> dict[str, Any]:
        """Compact record for the session tree / browser."""
        return {
            "session_id": self.session_id,
            "label": self.label,
            "parent_id": self.parent_id,
            "branch_point": self.branch_point,
            "world_seed": self.world_seed,
            "combat_control": self.combat_control,
            "n_decisions": self.n_decisions,
            "status": self.status,
            "stopped_reason": self.stopped_reason,
            "updated_at": self.updated_at,
        }


class SessionStore:
    def __init__(self, root: str | Path = "data/interactive") -> None:
        self.root = Path(root)

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def exists(self, session_id: str) -> bool:
        return (self.session_dir(session_id) / SESSION_JSON).is_file()

    def save(
        self,
        stored: StoredSession,
        decisions: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """Write the three files atomically-enough for a single local writer."""
        directory = self.session_dir(stored.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SESSION_JSON).write_text(
            json.dumps(stored.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (directory / DECISIONS_JSONL).open("w", encoding="utf-8") as handle:
            for record in decisions:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if meta is not None:
            (directory / META_JSON).write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return directory

    def load_stored(self, session_id: str) -> StoredSession:
        path = self.session_dir(session_id) / SESSION_JSON
        if not path.is_file():
            raise KeyError(f"no session {session_id!r} under {self.root}")
        return StoredSession.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_decisions(self, session_id: str) -> list[dict[str, Any]]:
        path = self.session_dir(session_id) / DECISIONS_JSONL
        if not path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def list_stored(self) -> list[StoredSession]:
        if not self.root.is_dir():
            return []
        out: list[StoredSession] = []
        for child in sorted(self.root.iterdir()):
            sj = child / SESSION_JSON
            if sj.is_file():
                try:
                    out.append(StoredSession.from_dict(json.loads(sj.read_text(encoding="utf-8"))))
                except (json.JSONDecodeError, TypeError):
                    continue  # skip a corrupt/foreign dir rather than crash the browser
        return out

    def delete(self, session_id: str) -> bool:
        directory = self.session_dir(session_id)
        if not directory.is_dir():
            return False
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        return True
