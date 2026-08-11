"""Visit-share regret and turn-set metrics for search-teacher audits."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class StateVisitStats:
    """Aggregated search evidence for one displayed public state."""

    public_state_hash: str
    window_id: str
    turn: int
    total_visits: int
    visit_share: dict[int, float]
    consensus_action_index: int
    top_set: tuple[int, ...]
    turn_set_descriptions: frozenset[str]
    margin: float


def _nonempty_string(value: Any, *, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(reason)
    return value


def _integer(value: Any, *, reason: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(reason)
    if nonnegative and value < 0:
        raise ValueError(reason)
    return value


def _validated_tie_ratio(tie_ratio: Any) -> float:
    if isinstance(tie_ratio, bool) or not isinstance(tie_ratio, (int, float)):
        raise ValueError("tie_ratio_not_number")
    value = float(tie_ratio)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("tie_ratio_out_of_range")
    return value


def _legal_action_descriptions(audit_row: dict[str, Any]) -> dict[int, str]:
    legal_actions = audit_row.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("legal_actions_not_nonempty_list")
    descriptions: dict[int, str] = {}
    for action in legal_actions:
        if not isinstance(action, dict):
            raise ValueError("legal_action_not_object")
        index = _integer(
            action.get("index"),
            reason="legal_action_index_invalid",
            nonnegative=True,
        )
        _integer(action.get("bits"), reason="legal_action_bits_invalid")
        description = _nonempty_string(
            action.get("description"),
            reason="legal_action_description_invalid",
        )
        if index in descriptions:
            raise ValueError("duplicate_legal_action_index")
        descriptions[index] = description
    return descriptions


def _query_evidence(
    query: Any,
    *,
    turn: int,
    legal_indices: frozenset[int],
) -> tuple[dict[int, int], set[str]] | None:
    if not isinstance(query, dict):
        raise ValueError("teacher_query_not_object")
    vote = query.get("teacher_vote")
    if not isinstance(vote, dict):
        raise ValueError("teacher_vote_not_object")
    abstained = vote.get("abstained")
    if not isinstance(abstained, bool):
        raise ValueError("teacher_vote_abstained_not_boolean")
    if abstained:
        return None
    displayed_visits = vote.get("displayed_action_visits")
    if not isinstance(displayed_visits, dict):
        raise ValueError("displayed_action_visits_not_object")
    visits: dict[int, int] = {}
    for raw_index, raw_visits in displayed_visits.items():
        if (
            not isinstance(raw_index, str)
            or not raw_index.isdigit()
            or str(int(raw_index)) != raw_index
        ):
            raise ValueError("displayed_action_visit_index_invalid")
        index = int(raw_index)
        count = _integer(
            raw_visits,
            reason="displayed_action_visit_count_invalid",
            nonnegative=True,
        )
        if index in visits:
            raise ValueError("duplicate_displayed_action_visit_index")
        visits[index] = count
    if frozenset(visits) != legal_indices:
        raise ValueError("displayed_action_visits_do_not_match_legal_actions")

    search = query.get("search")
    if not isinstance(search, dict):
        raise ValueError("search_not_object")
    best_sequence = search.get("best_sequence")
    if not isinstance(best_sequence, list):
        raise ValueError("best_sequence_not_list")
    descriptions: set[str] = set()
    for entry in best_sequence:
        if not isinstance(entry, dict):
            raise ValueError("best_sequence_entry_not_object")
        _integer(entry.get("bits"), reason="best_sequence_bits_invalid")
        entry_turn = _integer(
            entry.get("turn"),
            reason="best_sequence_turn_invalid",
            nonnegative=True,
        )
        description = _nonempty_string(
            entry.get("description"),
            reason="best_sequence_description_invalid",
        )
        if entry_turn == turn:
            descriptions.add(description)
    return visits, descriptions


def state_visit_stats(
    audit_row: dict,
    *,
    tie_ratio: float = 0.8,
) -> StateVisitStats:
    """Aggregate displayed-action visits and best-sequence actions for a state."""

    if not isinstance(audit_row, dict):
        raise ValueError("audit_row_not_object")
    ratio = _validated_tie_ratio(tie_ratio)
    public_state_hash = _nonempty_string(
        audit_row.get("public_state_hash"),
        reason="public_state_hash_invalid",
    )
    window_id = _nonempty_string(
        audit_row.get("window_id"),
        reason="window_id_invalid",
    )
    turn = _integer(
        audit_row.get("turn"),
        reason="turn_invalid",
        nonnegative=True,
    )
    action_descriptions = _legal_action_descriptions(audit_row)
    legal_indices = frozenset(action_descriptions)
    queries = audit_row.get("teacher_queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("teacher_queries_not_nonempty_list")

    aggregated_visits = {index: 0 for index in action_descriptions}
    turn_descriptions: set[str] = set()
    for query in queries:
        evidence = _query_evidence(
            query,
            turn=turn,
            legal_indices=legal_indices,
        )
        if evidence is None:
            continue
        visits, descriptions = evidence
        for index, count in visits.items():
            aggregated_visits[index] += count
        turn_descriptions.update(descriptions)

    total_visits = sum(aggregated_visits.values())
    if total_visits <= 0:
        raise ValueError("zero_non_abstaining_visits")
    visit_share = {
        index: count / total_visits
        for index, count in sorted(aggregated_visits.items())
    }
    maximum = max(visit_share.values())
    consensus_action_index = min(
        index for index, share in visit_share.items() if share == maximum
    )
    maximum_visits = max(aggregated_visits.values())
    top_set_threshold = math.nextafter(
        ratio * maximum_visits,
        -math.inf,
    )
    top_set = tuple(
        index
        for index, count in sorted(aggregated_visits.items())
        if count >= top_set_threshold
    )
    ordered_shares = sorted(visit_share.values(), reverse=True)
    margin = (
        ordered_shares[0] - ordered_shares[1]
        if len(ordered_shares) > 1
        else 0.0
    )
    stats = StateVisitStats(
        public_state_hash=public_state_hash,
        window_id=window_id,
        turn=turn,
        total_visits=total_visits,
        visit_share=visit_share,
        consensus_action_index=consensus_action_index,
        top_set=top_set,
        turn_set_descriptions=frozenset(turn_descriptions),
        margin=margin,
    )
    # The specified public dataclass does not expose legal-action descriptions,
    # while score_choice's default path must resolve them.  Preserve that exact
    # public field contract and retain the source-row lookup as private metadata.
    object.__setattr__(stats, "_action_descriptions", action_descriptions)
    return stats


def score_choice(
    stats: StateVisitStats,
    chosen_index: int,
    chosen_description: str | None = None,
) -> dict[str, Any]:
    """Score one displayed model choice against aggregated search evidence."""

    if not isinstance(stats, StateVisitStats):
        raise ValueError("stats_invalid")
    index = _integer(
        chosen_index,
        reason="chosen_index_invalid",
        nonnegative=True,
    )
    if index not in stats.visit_share:
        raise ValueError("chosen_index_not_legal")
    try:
        descriptions = getattr(stats, "_action_descriptions")
        resolved_description = descriptions[index]
    except (AttributeError, KeyError, TypeError):
        resolved_description = None
    if chosen_description is None:
        if resolved_description is None:
            raise ValueError("chosen_description_unavailable")
        description = resolved_description
    else:
        description = _nonempty_string(
            chosen_description,
            reason="chosen_description_invalid",
        )
        if (
            resolved_description is not None
            and description != resolved_description
        ):
            raise ValueError(
                "chosen_description_mismatch:"
                f"{description!r}!={resolved_description!r}"
            )
    maximum = max(stats.visit_share.values())
    chosen_share = stats.visit_share[index]
    return {
        "strict_top1": index == stats.consensus_action_index,
        "in_top_set": index in stats.top_set,
        "in_turn_set": description in stats.turn_set_descriptions,
        "regret_visit_share": maximum - chosen_share,
        "chosen_share": chosen_share,
        "margin": stats.margin,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "n": 0,
            "mean_regret_visit_share": None,
            "top_set_rate": None,
            "turn_set_rate": None,
            "strict_top1_rate": None,
            "mean_margin": None,
        }
    return {
        "n": count,
        "mean_regret_visit_share": sum(
            float(row["regret_visit_share"]) for row in rows
        )
        / count,
        "top_set_rate": sum(bool(row["in_top_set"]) for row in rows) / count,
        "turn_set_rate": sum(bool(row["in_turn_set"]) for row in rows) / count,
        "strict_top1_rate": sum(bool(row["strict_top1"]) for row in rows) / count,
        "mean_margin": sum(float(row["margin"]) for row in rows) / count,
    }


def build_metrics_report(
    per_row_choices: list[dict],
    audit_rows_by_hash: dict[str, dict],
    *,
    quarantined_windows: frozenset[str] = frozenset(),
    tie_ratio: float = 0.8,
) -> dict[str, Any]:
    """Join model choices to audit rows and summarize regret metrics."""

    if not isinstance(per_row_choices, list):
        raise ValueError("per_row_choices_not_list")
    if not isinstance(audit_rows_by_hash, dict):
        raise ValueError("audit_rows_by_hash_not_object")
    if not isinstance(quarantined_windows, frozenset) or any(
        not isinstance(value, str) or not value for value in quarantined_windows
    ):
        raise ValueError("quarantined_windows_invalid")
    ratio = _validated_tie_ratio(tie_ratio)

    scored_rows: list[dict[str, Any]] = []
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for choice in per_row_choices:
        if not isinstance(choice, dict):
            raise ValueError("per_row_choice_not_object")
        public_state_hash = _nonempty_string(
            choice.get("public_state_hash"),
            reason="choice_public_state_hash_invalid",
        )
        chosen_index = _integer(
            choice.get("chosen_index"),
            reason="chosen_index_invalid",
            nonnegative=True,
        )
        if public_state_hash not in audit_rows_by_hash:
            raise ValueError(f"audit_row_missing_for_hash:{public_state_hash}")
        audit_row = audit_rows_by_hash[public_state_hash]
        stats = state_visit_stats(audit_row, tie_ratio=ratio)
        if stats.public_state_hash != public_state_hash:
            raise ValueError(f"audit_hash_key_mismatch:{public_state_hash}")
        chosen_description = choice.get("chosen_description")
        score = score_choice(
            stats,
            chosen_index,
            chosen_description=chosen_description,
        )
        row = {
            "public_state_hash": public_state_hash,
            "window_id": stats.window_id,
            "turn": stats.turn,
            "chosen_index": chosen_index,
            **score,
        }
        scored_rows.append(row)
        by_window[stats.window_id].append(row)

    clean_rows = [
        row
        for row in scored_rows
        if row["window_id"] not in quarantined_windows
    ]
    return {
        "tie_ratio": ratio,
        "overall": {
            "all": _aggregate(scored_rows),
            "clean": _aggregate(clean_rows),
        },
        "per_window": {
            window_id: _aggregate(window_rows)
            for window_id, window_rows in sorted(by_window.items())
        },
        "rows": scored_rows,
    }
