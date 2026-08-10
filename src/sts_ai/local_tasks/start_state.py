from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


START_STATE_SIGNATURE_SCHEMA_VERSION = 2
PUBLIC_COMBAT_OBSERVATION = "combat_public_v2"


class LocalTaskStartSignatureError(RuntimeError):
    """A replayed task start differs from its validated public signature."""


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _encounter_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    combat = summary.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        raise LocalTaskStartSignatureError(
            "local-task start signature requires a pending combat encounter"
        )
    counts = Counter(
        str(enemy.get("name", ""))
        for enemy in enemies
        if isinstance(enemy, dict)
    )
    if not counts or "" in counts:
        raise LocalTaskStartSignatureError(
            "local-task start signature requires named combat enemies"
        )
    return {
        "enemy_counts": dict(sorted(counts.items())),
        "exact_composition": True,
    }


def start_state_payload_sha256(payload: dict[str, Any]) -> str:
    envelope = {
        "schema_version": START_STATE_SIGNATURE_SCHEMA_VERSION,
        "payload": payload,
    }
    return hashlib.sha256(_canonical_json(envelope)).hexdigest()


def build_start_state_signature(env: Any) -> dict[str, Any]:
    """Fingerprint the complete public task-start choice presented to a human.

    The state component is exactly the ``combat_public_v2`` serializer output.
    That serializer exposes sorted pile aggregates rather than hidden draw order.
    Ordered *display* descriptions capture the action menu the policy can choose
    from while deliberately excluding raw action bits and simulator-private state.

    The env's configured observation mode is restored before returning, so this
    check is identical for legacy and public-observation evaluation arms.
    """
    env.advance_to_decision()
    if getattr(env, "bc", None) is None:
        raise LocalTaskStartSignatureError(
            "local-task start signature requires a pending combat decision"
        )

    original_observation = getattr(env, "combat_observation", "legacy")
    try:
        env.combat_observation = PUBLIC_COMBAT_OBSERVATION
        state_text = str(env.describe_state())
        action_descriptions = [
            str(action.description) for action in env.legal_actions()
        ]
    finally:
        env.combat_observation = original_observation

    if not action_descriptions:
        raise LocalTaskStartSignatureError(
            "local-task start signature requires at least one displayed legal action"
        )
    payload = {
        "combat_observation": PUBLIC_COMBAT_OBSERVATION,
        "state_text": state_text,
        "legal_action_descriptions": action_descriptions,
        "encounter": _encounter_from_summary(env.summary()),
    }
    return {
        "schema_version": START_STATE_SIGNATURE_SCHEMA_VERSION,
        "sha256": start_state_payload_sha256(payload),
        "payload": payload,
    }


def validate_stored_start_state_signature(
    env: Any,
    window: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a stored signature, or no-op for unvalidated source manifests."""
    expected = window.get("start_state_signature")
    if expected is None:
        return None
    if not isinstance(expected, dict):
        raise LocalTaskStartSignatureError(
            f"window {window.get('window_id')!r} has an invalid start-state signature"
        )
    if expected.get("schema_version") != START_STATE_SIGNATURE_SCHEMA_VERSION:
        raise LocalTaskStartSignatureError(
            f"window {window.get('window_id')!r} has unsupported start-state "
            f"signature schema {expected.get('schema_version')!r}"
        )
    expected_payload = expected.get("payload")
    expected_sha256 = expected.get("sha256")
    if (
        not isinstance(expected_payload, dict)
        or not isinstance(expected_sha256, str)
        or start_state_payload_sha256(expected_payload) != expected_sha256
    ):
        raise LocalTaskStartSignatureError(
            f"window {window.get('window_id')!r} has an internally inconsistent "
            "start-state signature"
        )

    actual = build_start_state_signature(env)
    if actual["sha256"] != expected_sha256:
        raise LocalTaskStartSignatureError(
            f"replay public start state diverged for {window.get('window_id')!r}: "
            f"expected_sha256={expected_sha256}, actual_sha256={actual['sha256']}"
        )
    return actual
