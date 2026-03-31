from ticket_triage_env.models import (
    ActionType,
    NextAction,
    Priority,
    Queue,
    TicketCategory,
    TicketTriageAction,
)
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


def test_missing_ticket_id_returns_error() -> None:
    env = TicketTriageEnvironment()
    env.reset("easy")
    result = env.step(TicketTriageAction(action_type=ActionType.SET_FIELDS))
    assert result["observation"]["last_action_result"]["success"] is False
    assert result["reward"] < 0


def test_step_after_done_returns_done() -> None:
    env = TicketTriageEnvironment()
    env.reset("easy")
    env.step(TicketTriageAction(action_type=ActionType.SUBMIT_BATCH))
    result = env.step(TicketTriageAction(action_type=ActionType.NOOP))
    assert result["done"] is True
    assert result["reward"] == 0.0


def test_full_episode_easy_reaches_done_with_good_score() -> None:
    env = TicketTriageEnvironment()
    env.reset("easy")

    scripted = {
        "E-001": {
            "category": TicketCategory.BILLING_DISPUTE,
            "priority": Priority.HIGH,
            "queue": Queue.BILLING_OPS,
            "next_action": NextAction.REFUND_REVIEW,
            "response": (
                "Thank you for reporting this billing issue. "
                "We understand the concern and next we will review your refund request and share an update."
            ),
        },
        "E-002": {
            "category": TicketCategory.ACCESS_ISSUE,
            "priority": Priority.MEDIUM,
            "queue": Queue.TECH_SUPPORT,
            "next_action": NextAction.PROVIDE_STEPS,
            "response": (
                "Thank you for reporting this login issue. "
                "We understand the impact and next you can follow these troubleshooting steps while we monitor."
            ),
        },
    }

    terminal_result = None
    for ticket_id, plan in scripted.items():
        env.step(TicketTriageAction(action_type=ActionType.INSPECT_TICKET, ticket_id=ticket_id))
        env.step(
            TicketTriageAction(
                action_type=ActionType.SET_FIELDS,
                ticket_id=ticket_id,
                category=plan["category"],
                priority=plan["priority"],
                queue=plan["queue"],
                next_action=plan["next_action"],
            )
        )
        env.step(
            TicketTriageAction(
                action_type=ActionType.DRAFT_RESPONSE,
                ticket_id=ticket_id,
                response_text=plan["response"],
            )
        )
        terminal_result = env.step(
            TicketTriageAction(action_type=ActionType.SUBMIT_TICKET, ticket_id=ticket_id)
        )

    assert terminal_result is not None
    assert terminal_result["done"] is True
    assert terminal_result["info"]["final_score"] >= 0.85
