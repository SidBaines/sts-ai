"""Vote rules over canonical combat-search diagnostics."""
from collections import Counter
import math

RULES = ("best_line", "most_visited", "mean_value", "pooled_mean")


def _merge_edges(edges):
    visits, evaluations = Counter(), Counter()
    for edge in edges:
        visits[edge["index"]] += edge["visits"]
        evaluations[edge["index"]] += edge["evaluation_sum"]
    return [{"index": index, "visits": count, "evaluation_sum": evaluations[index],
             "mean_evaluation": evaluations[index] / count if count else 0.}
            for index, count in visits.items()]


def _mean_winner(edges):
    maximum = max(edge["visits"] for edge in edges)
    return max((edge for edge in edges if edge["visits"] * 10 >= maximum),
               key=lambda edge: (edge["mean_evaluation"], edge["visits"], -edge["index"]))["index"]


def pooled_edges(details):
    return _merge_edges(edge for search in details for edge in _merge_edges(search["edges"]))


def pooled_value_target(details, scale) -> dict[int, float]:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("value_scale must be finite and positive")
    edges = [edge for edge in pooled_edges(details) if edge["visits"] > 0]
    if not edges:
        return {}
    maximum = max(edge["visits"] for edge in edges)
    eligible = [edge for edge in edges if edge["visits"] * 10 >= maximum]
    best = max(edge["mean_evaluation"] for edge in eligible)
    weights = {edge["index"]: math.exp((edge["mean_evaluation"] - best) / scale) for edge in eligible}
    total = sum(weights.values())
    return {index: weight / total for index, weight in weights.items()}


def votes_from_details(details, rule) -> dict[int, int]:
    if rule not in RULES:
        raise ValueError(f"Unknown teacher vote rule: {rule}")
    if not details:
        return {}
    if rule == "best_line":
        return dict(Counter(search["chosen"] for search in details))
    searches = [_merge_edges(search["edges"]) for search in details]
    if rule == "pooled_mean":
        edges = _merge_edges(edge for edges in searches for edge in edges)
        return {_mean_winner(edges): len(details)} if edges else {}
    votes = Counter()
    for edges in searches:
        if not edges:
            return {}
        index = (max(edges, key=lambda edge: (edge["visits"], -edge["index"]))["index"]
                 if rule == "most_visited" else _mean_winner(edges))
        votes[index] += 1
    return dict(votes)
