from __future__ import annotations

from typing import Dict, Iterable


STRICT_SCORE_EPSILON = 0.01


def _strict_unit_interval(value: float) -> float:
    if value <= 0.0:
        return STRICT_SCORE_EPSILON
    if value >= 1.0:
        return 1.0 - STRICT_SCORE_EPSILON
    return value


def score_batch(per_ticket_scores: Iterable[Dict[str, float]], unresolved_urgent_count: int) -> float:
    scores = list(per_ticket_scores)
    if not scores:
        return STRICT_SCORE_EPSILON

    avg = sum(item["total"] for item in scores) / len(scores)
    sla_penalty = min(0.20, 0.05 * max(0, unresolved_urgent_count))
    final_score = avg - sla_penalty
    return _strict_unit_interval(final_score)
