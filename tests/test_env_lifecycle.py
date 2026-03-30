from ticket_triage_env.models import ActionType, TicketTriageAction
from ticket_triage_env.server.environment import TicketTriageEnvironment


def test_reset_step_state_lifecycle() -> None:
    env = TicketTriageEnvironment()
    result = env.reset("easy")
    assert result["done"] is False

    ticket_id = result["observation"]["ticket_view"]["ticket"]["ticket_id"]
    step_result = env.step(TicketTriageAction(action_type=ActionType.INSPECT_TICKET, ticket_id=ticket_id))
    assert "observation" in step_result
    assert "reward" in step_result
    assert "done" in step_result
    assert "info" in step_result

    state = env.state()
    assert state["task_id"] == "easy"
    assert state["step_count"] >= 1
