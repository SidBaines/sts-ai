"""Pure policy helpers for comparing native search root-action selectors."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from sts_ai.schemas import LegalAction
from sts_ai.teacher import displayed_action_index


def root_visit_vote(
    search_result: dict[str, Any],
    legal_actions: Iterable[LegalAction | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate native root-edge visits into the deduplicated displayed menu.

    A valid edge that cannot be mapped makes the whole query abstain: silently
    dropping its visits could change the winning displayed action. Invalid
    native edges are excluded because the searcher did not consider them legal.
    """
    actions = list(legal_actions)
    visits: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    invalid_edges = 0
    unmapped_edges: list[dict[str, Any]] = []
    for edge in search_result.get("root_edges") or []:
        if not bool(edge.get("valid", False)):
            invalid_edges += 1
            continue
        try:
            action_index = displayed_action_index(
                actions,
                bits=_optional_int(edge.get("bits")),
                description=_optional_str(edge.get("description")),
            )
        except ValueError as exc:
            unmapped_edges.append(
                {
                    "bits": _optional_int(edge.get("bits")),
                    "description": _optional_str(edge.get("description")),
                    "visits": int(edge.get("visits", 0)),
                    "error": str(exc),
                }
            )
            continue
        edge_visits = int(edge.get("visits", 0))
        if edge_visits < 0:
            raise ValueError("root-edge visits must be non-negative")
        visits[action_index] += edge_visits
        edge_counts[action_index] += 1

    ordered_visits = {str(index): visits[index] for index in sorted(visits)}
    ordered_edge_counts = {
        str(index): edge_counts[index] for index in sorted(edge_counts)
    }
    if unmapped_edges:
        return {
            "action_index": None,
            "abstained": True,
            "reason": "unmapped_valid_root_edge",
            "displayed_action_visits": ordered_visits,
            "displayed_action_edge_counts": ordered_edge_counts,
            "n_invalid_edges": invalid_edges,
            "unmapped_edges": unmapped_edges,
        }
    if not visits:
        return {
            "action_index": None,
            "abstained": True,
            "reason": "no_valid_root_edges",
            "displayed_action_visits": {},
            "displayed_action_edge_counts": {},
            "n_invalid_edges": invalid_edges,
            "unmapped_edges": [],
        }

    max_visits = max(visits.values())
    winners = sorted(index for index, value in visits.items() if value == max_visits)
    tied = len(winners) > 1
    return {
        "action_index": None if tied else winners[0],
        "abstained": tied,
        "reason": "tied_max_visits" if tied else None,
        "displayed_action_visits": ordered_visits,
        "displayed_action_edge_counts": ordered_edge_counts,
        "n_invalid_edges": invalid_edges,
        "unmapped_edges": [],
        "max_visits": max_visits,
        "tied_action_indices": winners if tied else [],
    }


def winning_sequence_vote(
    search_result: dict[str, Any],
    legal_actions: Iterable[LegalAction | dict[str, Any]],
) -> dict[str, Any]:
    """Map the searcher's selected winning-sequence first action to the menu."""
    if not bool(search_result.get("winning_sequence_found", False)):
        return {
            "action_index": None,
            "abstained": True,
            "reason": "no_winning_sequence",
        }
    try:
        action_index = displayed_action_index(
            legal_actions,
            bits=_optional_int(search_result.get("bits")),
            description=_optional_str(search_result.get("description")),
        )
    except ValueError as exc:
        return {
            "action_index": None,
            "abstained": True,
            "reason": "unmapped_winning_sequence_action",
            "error": str(exc),
        }
    return {
        "action_index": action_index,
        "abstained": False,
        "reason": None,
    }


def action_vote_consensus(
    votes: Iterable[int | None],
    *,
    min_fraction: float = 2 / 3,
) -> dict[str, Any]:
    """Require a unique action with the threshold fraction of all seed votes.

    Abstentions stay in the denominator, so two matching votes plus one
    abstention pass at exactly 2/3, while one vote plus two abstentions fails.
    """
    vote_list = list(votes)
    if not vote_list:
        raise ValueError("at least one vote is required")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must be in (0, 1]")
    counts = Counter(int(vote) for vote in vote_list if vote is not None)
    best_count = max(counts.values(), default=0)
    winners = sorted(action for action, count in counts.items() if count == best_count)
    tied = len(winners) > 1
    consensus_action = winners[0] if len(winners) == 1 else None
    fraction = best_count / len(vote_list)
    eligible = consensus_action is not None and fraction >= min_fraction
    if not counts:
        reason = "all_seeds_abstained"
    elif tied:
        reason = "tied_seed_vote"
    elif fraction < min_fraction:
        reason = "below_consensus_threshold"
    else:
        reason = None
    return {
        "action_index": consensus_action if eligible else None,
        "leading_action_index": consensus_action,
        "eligible": eligible,
        "reason": reason,
        "n_queries": len(vote_list),
        "n_abstentions": sum(vote is None for vote in vote_list),
        "action_counts": {str(action): counts[action] for action in sorted(counts)},
        "consensus_fraction": fraction,
        "unanimous": best_count == len(vote_list),
        "tied": tied,
        "min_fraction": min_fraction,
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
