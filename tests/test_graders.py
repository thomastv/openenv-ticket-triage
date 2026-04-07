from ticket_triage_env.graders import score_batch, score_ticket
from ticket_triage_env.models import (
    NextAction,
    Priority,
    Queue,
    TicketCategory,
    TicketDecision,
)


def test_score_ticket_range() -> None:
    decision = TicketDecision(
        category=TicketCategory.BILLING_DISPUTE,
        priority=Priority.HIGH,
        queue=Queue.BILLING_OPS,
        next_action=NextAction.REFUND_REVIEW,
        response_text="Thanks for reporting this billing issue. Next we will review your refund.",
    )
    answer = {
        "category": "billing_dispute",
        "priority": "high",
        "queue": "billing_ops",
        "next_action": "refund_review",
        "response_required_keywords": ["billing", "refund", "next"],
        "response_prohibited_phrases": ["your fault"],
        "response_max_chars": 500,
    }

    result = score_ticket(decision, answer)
    assert 0.0 <= result["total"] <= 1.0


def test_score_batch_bounds() -> None:
    score = score_batch([{"total": 0.9}, {"total": 0.8}], unresolved_urgent_count=1)
    assert 0.0 < score < 1.0


def test_score_batch_is_strictly_inside_unit_interval_at_edges() -> None:
    assert 0.0 < score_batch([{"total": 0.0}], unresolved_urgent_count=0) < 1.0
    assert 0.0 < score_batch([{"total": 1.0}], unresolved_urgent_count=0) < 1.0


def test_score_ticket_deterministic() -> None:
    decision = TicketDecision(
        category=TicketCategory.BILLING_DISPUTE,
        priority=Priority.HIGH,
        queue=Queue.BILLING_OPS,
        next_action=NextAction.REFUND_REVIEW,
        response_text=(
            "Thank you for reporting this billing issue. "
            "We understand the concern and next we will review your refund request carefully."
        ),
    )
    answer = {
        "category": "billing_dispute",
        "priority": "high",
        "queue": "billing_ops",
        "next_action": "refund_review",
        "response_required_keywords": ["billing", "refund", "next"],
        "response_prohibited_phrases": ["your fault"],
        "response_max_chars": 500,
    }
    assert score_ticket(decision, answer) == score_ticket(decision, answer)


def test_score_ticket_empty_decision_is_zero() -> None:
    decision = TicketDecision()
    answer = {
        "category": "billing_dispute",
        "priority": "high",
        "queue": "billing_ops",
        "next_action": "refund_review",
        "response_required_keywords": ["billing"],
        "response_prohibited_phrases": [],
        "response_max_chars": 500,
    }
    result = score_ticket(decision, answer)
    assert result["total"] == 0.0


def test_score_ticket_near_perfect_with_strong_response() -> None:
    decision = TicketDecision(
        category=TicketCategory.BILLING_DISPUTE,
        priority=Priority.HIGH,
        queue=Queue.BILLING_OPS,
        next_action=NextAction.REFUND_REVIEW,
        response_text=(
            "Thank you for contacting us about this billing issue. "
            "We understand the impact and next we will complete a refund review and share an update shortly."
        ),
    )
    answer = {
        "category": "billing_dispute",
        "priority": "high",
        "queue": "billing_ops",
        "next_action": "refund_review",
        "response_required_keywords": ["billing", "refund", "next"],
        "response_prohibited_phrases": ["your fault"],
        "response_max_chars": 500,
    }
    result = score_ticket(decision, answer)
    assert result["total"] >= 0.95
