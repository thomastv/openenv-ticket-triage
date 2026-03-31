import pytest

from ticket_triage_env.models import ActionType, TicketCategory, TicketTriageAction


def test_action_model_accepts_enum_fields() -> None:
    action = TicketTriageAction(
        action_type=ActionType.SET_FIELDS,
        ticket_id="E-001",
        category=TicketCategory.BILLING_DISPUTE,
    )
    assert action.ticket_id == "E-001"
    assert action.category == TicketCategory.BILLING_DISPUTE


def test_action_noop_allows_minimal_payload() -> None:
    action = TicketTriageAction(action_type=ActionType.NOOP)
    assert action.ticket_id is None


def test_invalid_action_type_rejected() -> None:
    with pytest.raises(Exception):
        TicketTriageAction(action_type="invalid_type")
