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


def _response_component(response_text: str | None, answer: Dict[str, object]) -> float:
    text = (response_text or "").strip().lower()
    if not text:
        return 0.0

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
    return (
        0.45 * required_score
        + 0.20 * prohibited_ok
        + 0.10 * length_ok
        + 0.125 * contains_ack
        + 0.125 * contains_next_step
    )


def score_ticket(decision: TicketDecision, answer: Dict[str, object]) -> Dict[str, float]:
    category_score = 1.0 if decision.category and decision.category.value == answer.get("category") else 0.0
    priority_score = 1.0 if decision.priority and decision.priority.value == answer.get("priority") else 0.0
    queue_score = 1.0 if decision.queue and decision.queue.value == answer.get("queue") else 0.0
    next_action_score = (
        1.0 if decision.next_action and decision.next_action.value == answer.get("next_action") else 0.0
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
        "category": category_score,
        "priority": priority_score,
        "queue": queue_score,
        "next_action": next_action_score,
        "response": response_score,
        "total": total,
    }
