from ticket_triage_env.models import ActionType, TicketCategory, TicketTriageAction


def test_action_model_accepts_enum_fields() -> None:
    action = TicketTriageAction(
        action_type=ActionType.SET_FIELDS,
        ticket_id="E-001",
        category=TicketCategory.BILLING_DISPUTE,
    )
    assert action.ticket_id == "E-001"
    assert action.category == TicketCategory.BILLING_DISPUTE
