"""Detect encounter-inconsistent powers in serialized combat observations."""
from __future__ import annotations

from dataclasses import dataclass
import re


_GREMLIN_NOB_ENEMY_POWERS = frozenset(
    {"Enrage", "Strength", "Vulnerable", "Weak"}
)
_SUPPORTED_TASK_IDS = frozenset({"gremlin_nob"})
_POWER_RE = re.compile(r"([A-Za-z][A-Za-z' -]*?) (-?\d+)")
_PARENTHETICAL_RE = re.compile(r"\((?:[^()]|\([^()]*\))*\)")


@dataclass(frozen=True)
class SanityFinding:
    """One encounter-inconsistent power found in serialized state text."""

    side: str
    power: str
    amount: int
    reason: str


def _enemy_lines(state_text: str) -> list[str]:
    lines = state_text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "Enemies:"
        )
    except StopIteration:
        return []

    result: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Hand:") or stripped.startswith("Piles:"):
            break
        result.append(line)
    return result


def _parse_power_field(field: str) -> tuple[str, int] | None:
    match = _POWER_RE.fullmatch(_PARENTHETICAL_RE.sub("", field).strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _enemy_power_findings(state_text: str) -> list[SanityFinding]:
    findings: list[SanityFinding] = []
    for line in _enemy_lines(state_text):
        _before_intent, separator, after_intent = line.partition("intent ")
        if not separator:
            continue
        # The first comma-delimited field contains the intent name/description.
        # Later fields are the serializer's power list, except that a no-attack
        # descriptor may share its field with the first power.
        fields = after_intent.split(",")
        for field in fields[1:]:
            parsed = _parse_power_field(field)
            if parsed is None:
                continue
            power, amount = parsed
            if power not in _GREMLIN_NOB_ENEMY_POWERS:
                findings.append(
                    SanityFinding(
                        side="enemy",
                        power=power,
                        amount=amount,
                        reason="enemy_power_impossible_for_encounter",
                    )
                )
    return findings


def _prefixed_line(state_text: str, prefix: str) -> str | None:
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def _player_power_findings(state_text: str) -> list[SanityFinding]:
    powers_text = _prefixed_line(state_text, "Player powers:")
    if powers_text is None or powers_text == "none":
        return []

    relics_text = _prefixed_line(state_text, "Relics:")
    relics = (
        {
            item.strip().split(" [", 1)[0].strip()
            for item in relics_text.split(",")
            if item.strip()
        }
        if relics_text is not None
        else set()
    )
    findings: list[SanityFinding] = []
    for field in powers_text.split(","):
        parsed = _parse_power_field(field)
        if parsed is None:
            continue
        power, amount = parsed
        if power == "Thorns" and "Bronze Scales" not in relics:
            findings.append(
                SanityFinding(
                    side="player",
                    power=power,
                    amount=amount,
                    reason="player_thorns_without_source",
                )
            )
        elif power == "Buffer":
            findings.append(
                SanityFinding(
                    side="player",
                    power=power,
                    amount=amount,
                    reason="player_buffer_without_source",
                )
            )
    return findings


def validate_task_id(task_id: str) -> None:
    """Reject task identifiers without an encounter-specific sanity policy."""
    if task_id not in _SUPPORTED_TASK_IDS:
        raise ValueError(f"unsupported task_id: {task_id!r}")


def phantom_power_findings(
    state_text: str,
    *,
    task_id: str,
) -> list[SanityFinding]:
    """Return powers impossible in the named encounter, failing on unknown tasks."""
    validate_task_id(task_id)
    lines = [line.strip() for line in state_text.splitlines()]
    if "Enemies:" not in lines or not any(
        line.startswith("Player powers:") for line in lines
    ):
        raise ValueError("state_text_not_a_combat_state")
    return _enemy_power_findings(state_text) + _player_power_findings(state_text)
