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
    assert 0.0 <= score <= 1.0
