from __future__ import annotations

from typing import Dict, Iterable


def score_batch(per_ticket_scores: Iterable[Dict[str, float]], unresolved_urgent_count: int) -> float:
    scores = list(per_ticket_scores)
    if not scores:
        return 0.0

    avg = sum(item["total"] for item in scores) / len(scores)
    sla_penalty = min(0.20, 0.05 * max(0, unresolved_urgent_count))
    final_score = avg - sla_penalty
    if final_score < 0.0:
        return 0.0
    if final_score > 1.0:
        return 1.0
    return final_score
