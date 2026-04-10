from __future__ import annotations

from typing import Dict, List

from ticket_triage_env.models import TicketDecision


WEIGHTS = {
    "category": 0.25,
    "priority": 0.20,
    "queue": 0.20,
    "next_action": 0.20,
    "response": 0.15,
}

MIN_RESPONSE_LENGTH = 50
STRICT_SCORE_EPSILON = 0.01


def _strict_unit_interval(value: float) -> float:
    if value < STRICT_SCORE_EPSILON:
        return STRICT_SCORE_EPSILON
    if value > 1.0 - STRICT_SCORE_EPSILON:
        return 1.0 - STRICT_SCORE_EPSILON
    return value


def _binary_match(actual: object, expected: object) -> float:
    return 1.0 if actual == expected else 0.0


def _response_component(response_text: str | None, answer: Dict[str, object]) -> float:
    text = (response_text or "").strip().lower()
    if not text or len(text) < MIN_RESPONSE_LENGTH:
        return _strict_unit_interval(0.0)

    required_keywords: List[str] = [
        str(value).lower() for value in answer.get("response_required_keywords", [])
    ]
    prohibited_phrases: List[str] = [
        str(value).lower() for value in answer.get("response_prohibited_phrases", [])
    ]

    required_hits = 0
    if required_keywords:
        for keyword in required_keywords:
            if keyword in text:
                required_hits += 1
        required_score = required_hits / len(required_keywords)
    else:
        required_score = 1.0

    prohibited_ok = 1.0
    for phrase in prohibited_phrases:
        if phrase and phrase in text:
            prohibited_ok = 0.0
            break

    max_chars = int(answer.get("response_max_chars", 600))
    length_ok = 1.0 if len(text) <= max_chars else 0.0

    contains_ack = 1.0 if any(k in text for k in ["sorry", "thanks", "thank you", "understand"]) else 0.0
    contains_next_step = 1.0 if any(k in text for k in ["next", "please", "we will", "you can"]) else 0.0

    # Weighted deterministic compliance rubric.
    score = (
        0.45 * required_score
        + 0.20 * prohibited_ok
        + 0.10 * length_ok
        + 0.125 * contains_ack
        + 0.125 * contains_next_step
    )
    return _strict_unit_interval(score)


def score_ticket(decision: TicketDecision, answer: Dict[str, object]) -> Dict[str, float]:
    category_score = _binary_match(
        decision.category.value if decision.category else None,
        answer.get("category"),
    )
    priority_score = _binary_match(
        decision.priority.value if decision.priority else None,
        answer.get("priority"),
    )
    queue_score = _binary_match(
        decision.queue.value if decision.queue else None,
        answer.get("queue"),
    )
    next_action_score = _binary_match(
        decision.next_action.value if decision.next_action else None,
        answer.get("next_action"),
    )
    response_score = _response_component(decision.response_text, answer)

    total = (
        WEIGHTS["category"] * category_score
        + WEIGHTS["priority"] * priority_score
        + WEIGHTS["queue"] * queue_score
        + WEIGHTS["next_action"] * next_action_score
        + WEIGHTS["response"] * response_score
    )

    return {
        "category": _strict_unit_interval(category_score),
        "priority": _strict_unit_interval(priority_score),
        "queue": _strict_unit_interval(queue_score),
        "next_action": _strict_unit_interval(next_action_score),
        "response": _strict_unit_interval(response_score),
        "total": _strict_unit_interval(total),
    }
