"""Pure helpers for search-teacher provenance and public-state labels.

The simulator query itself lives in the native binding.  This module owns the
policy-facing boundary: map a native action back to the deduplicated displayed
menu, hash only model-visible state, and summarize repeated teacher queries.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable

from sts_ai.schemas import LegalAction


PUBLIC_OBSERVATION_VERSION = "combat_public_v2"
TEACHER_PRIVILEGE = "simulator_full_state"
SEARCH_TEACHER_LABEL_VERSION = 2
AGGREGATED_ROOT_VISITS = "aggregated_root_visits"
NATIVE_SELECTED_ACTION = "native_selected_action"
TEACHER_SELECTION_RULES = (AGGREGATED_ROOT_VISITS, NATIVE_SELECTED_ACTION)


def validate_search_teacher_rows(
    rows: Iterable[dict[str, Any]],
    *,
    expected_selection_rule: str | None = None,
    observation_version: str = PUBLIC_OBSERVATION_VERSION,
) -> str:
    """Validate one homogeneous, current-interface teacher collection.

    Observation and selection identity are repeated at row and query level on
    purpose.  A stale/mixed JSONL must fail before it can be summarized or turned
    into training data, even if its detached manifest is missing or incorrect.
    Returns the single observed teacher selection rule.
    """

    values = list(rows)
    if not values:
        raise ValueError("teacher collection is empty")
    rules: set[str] = set()
    for row_index, row in enumerate(values):
        observation = row.get("observation_version")
        if observation != observation_version:
            raise ValueError(
                f"teacher row {row_index} observation_version={observation!r}; "
                f"expected {observation_version!r}"
            )
        if row.get("teacher_privilege") != TEACHER_PRIVILEGE:
            raise ValueError(
                f"teacher row {row_index} has invalid teacher_privilege"
            )
        rule = row.get("teacher_selection_rule")
        if rule not in TEACHER_SELECTION_RULES:
            raise ValueError(
                f"teacher row {row_index} has invalid teacher_selection_rule={rule!r}"
            )
        rules.add(str(rule))
        queries = row.get("teacher_queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(
                f"teacher row {row_index} must contain non-empty teacher_queries"
            )
        for query_index, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(
                    f"teacher row {row_index} query {query_index} is not an object"
                )
            if query.get("observation_version") != observation:
                raise ValueError(
                    f"teacher row {row_index} query {query_index} observation_version "
                    "disagrees with its row"
                )
            if query.get("teacher_selection_rule") != rule:
                raise ValueError(
                    f"teacher row {row_index} query {query_index} selection rule "
                    "disagrees with its row"
                )
            if query.get("teacher_privilege") != TEACHER_PRIVILEGE:
                raise ValueError(
                    f"teacher row {row_index} query {query_index} has invalid "
                    "teacher_privilege"
                )
    if len(rules) != 1:
        raise ValueError(f"teacher rows mix selection rules: {sorted(rules)!r}")
    rule = next(iter(rules))
    if expected_selection_rule is not None and rule != expected_selection_rule:
        raise ValueError(
            f"teacher_selection_rule={rule!r}; expected {expected_selection_rule!r}"
        )
    return rule


def validate_search_teacher_manifest(
    manifest: dict[str, Any],
    *,
    n_rows: int,
    selection_rule: str,
    observation_version: str = PUBLIC_OBSERVATION_VERSION,
) -> None:
    """Fail closed when a detached label manifest disagrees with its JSONL."""

    expected = {
        "kind": "search_teacher_labels",
        "version": SEARCH_TEACHER_LABEL_VERSION,
        "observation_version": observation_version,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "teacher_selection_rule": selection_rule,
        "n_rows": n_rows,
    }
    mismatches = {
        key: {"stored": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "search-teacher label manifest disagrees with the current JSONL "
            f"contract: {json.dumps(mismatches, sort_keys=True)}"
        )


def public_observation_hash(
    state_text: str,
    legal_actions: Iterable[LegalAction | dict[str, Any]],
    *,
    observation_version: str = PUBLIC_OBSERVATION_VERSION,
) -> str:
    """Hash exactly the policy-visible state and displayed action menu.

    Native action bits are deliberately excluded: they encode engine identity,
    are not printed in the prompt, and can differ for display-equivalent card
    copies.  The action order and descriptions are included because the policy
    chooses a displayed index.
    """

    descriptions = [
        action.description if isinstance(action, LegalAction) else str(action["description"])
        for action in legal_actions
    ]
    payload = {
        "observation_version": observation_version,
        "state_text": state_text,
        "legal_action_descriptions": descriptions,
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def displayed_action_index(
    legal_actions: Iterable[LegalAction | dict[str, Any]],
    *,
    bits: int | None,
    description: str | None,
) -> int:
    """Map a native/source action to the policy's deduplicated action index.

    Description is authoritative for display-equivalent hand copies: the native
    searcher can choose a copy whose bits differ from the first copy retained by
    the Python menu.  An exact bits match is used as a consistency check/fallback.
    Ambiguous or absent matches fail closed instead of silently teaching index 0.
    """

    actions = list(legal_actions)
    by_description = [
        action for action in actions
        if description is not None
        and (action.description if isinstance(action, LegalAction) else action.get("description"))
        == description
    ]
    if len(by_description) == 1:
        action = by_description[0]
        return int(action.index if isinstance(action, LegalAction) else action["index"])
    if len(by_description) > 1:
        raise ValueError(f"teacher description is ambiguous in displayed menu: {description!r}")

    by_bits = [
        action for action in actions
        if bits is not None
        and int(action.bits if isinstance(action, LegalAction) else action["bits"]) == int(bits)
    ]
    if len(by_bits) == 1:
        action = by_bits[0]
        return int(action.index if isinstance(action, LegalAction) else action["index"])
    if len(by_bits) > 1:
        raise ValueError(f"teacher bits are ambiguous in displayed menu: {bits}")
    raise ValueError(
        "teacher action is absent from displayed legal actions: "
        f"bits={bits!r}, description={description!r}"
    )


def teacher_label(
    *,
    state_text: str,
    legal_actions: list[LegalAction | dict[str, Any]],
    search_result: dict[str, Any],
    observation_version: str = PUBLIC_OBSERVATION_VERSION,
    teacher_vote: dict[str, Any] | None = None,
    selection_rule: str = NATIVE_SELECTED_ACTION,
) -> dict[str, Any]:
    """Create the auditable portion of one action-only teacher row."""

    if teacher_vote is None:
        action_index: int | None = displayed_action_index(
            legal_actions,
            bits=_optional_int(search_result.get("bits")),
            description=_optional_str(search_result.get("description")),
        )
        action_bits = _optional_int(search_result.get("bits"))
        action_description = _optional_str(search_result.get("description"))
        vote = {
            "action_index": action_index,
            "abstained": False,
            "reason": None,
        }
    else:
        vote = _json_teacher_provenance(teacher_vote)
        raw_index = teacher_vote.get("action_index")
        action_index = None if raw_index is None else int(raw_index)
        if action_index is None:
            action_bits = None
            action_description = None
        else:
            matches = [
                action for action in legal_actions
                if int(
                    action.index
                    if isinstance(action, LegalAction)
                    else action["index"]
                ) == action_index
            ]
            if len(matches) != 1:
                raise ValueError(
                    "teacher vote action index is absent or ambiguous in displayed "
                    f"menu: {action_index}"
                )
            selected = matches[0]
            action_bits = int(
                selected.bits if isinstance(selected, LegalAction) else selected["bits"]
            )
            action_description = str(
                selected.description
                if isinstance(selected, LegalAction)
                else selected["description"]
            )
    return {
        "public_state_hash": public_observation_hash(
            state_text,
            legal_actions,
            observation_version=observation_version,
        ),
        "observation_version": observation_version,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "teacher_action_index": action_index,
        "teacher_action_bits": action_bits,
        "teacher_action_description": action_description,
        "teacher_selection_rule": selection_rule,
        "teacher_vote": vote,
        "target": (
            json.dumps({"action_index": action_index}, separators=(",", ":"))
            if action_index is not None
            else None
        ),
        # Native BattleAction objects also occur inside root_edges/best_sequence.
        # Drop every key named "action" recursively while retaining its bits and
        # description siblings; pybind action objects are neither stable nor JSON
        # serializable.
        "search": _json_teacher_provenance(search_result),
    }


def consensus_summary(labels: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated/budgeted search labels without claiming false certainty."""

    rows = list(labels)
    if not rows:
        raise ValueError("at least one teacher label is required")
    hashes = {str(row["public_state_hash"]) for row in rows}
    if len(hashes) != 1:
        raise ValueError("teacher consensus rows do not share one public state")
    actions = [
        int(row["teacher_action_index"])
        for row in rows
        if row.get("teacher_action_index") is not None
    ]
    counts = Counter(actions)
    best_count = max(counts.values(), default=0)
    consensus_actions = sorted(action for action, count in counts.items() if count == best_count)
    consensus_action = consensus_actions[0] if len(consensus_actions) == 1 else None
    return {
        "public_state_hash": next(iter(hashes)),
        "n_queries": len(rows),
        "action_counts": {str(action): counts[action] for action in sorted(counts)},
        "consensus_action_index": consensus_action,
        "consensus_fraction": best_count / len(rows),
        "unanimous": len(actions) == len(rows) and len(counts) == 1,
        "tied": len(consensus_actions) > 1,
        "n_abstentions": len(rows) - len(actions),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _json_teacher_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_teacher_provenance(item)
            for key, item in value.items()
            if str(key) != "action"
        }
    if isinstance(value, (list, tuple)):
        return [_json_teacher_provenance(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "search provenance contains a non-JSON native value outside an 'action' key: "
        f"{type(value).__name__}"
    )
