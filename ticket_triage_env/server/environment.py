from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ticket_triage_env.graders import score_batch, score_ticket
from ticket_triage_env.models import (
    ActionType,
    LastActionResult,
    Queue,
    QueueSnapshot,
    ProgressMetrics,
    TicketDecision,
    TicketRecord,
    TicketStatus,
    TicketTriageAction,
    TicketTriageObservation,
    TicketTriageState,
    TicketView,
)


STRICT_SCORE_EPSILON = 0.01


def _strict_unit_interval(value: float) -> float:
    if value < STRICT_SCORE_EPSILON:
        return STRICT_SCORE_EPSILON
    if value > 1.0 - STRICT_SCORE_EPSILON:
        return 1.0 - STRICT_SCORE_EPSILON
    return value


class TicketTriageEnvironment:
    def __init__(self, scenarios_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.scenarios_dir = scenarios_dir or (base_dir / "data" / "scenarios")

        self.task_id: str = "easy"
        self.max_steps: int = 20
        self.episode_id: str = ""
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.current_ticket_id: Optional[str] = None
        self._tickets: Dict[str, TicketRecord] = {}
        self._decisions: Dict[str, TicketDecision] = {}
        self._answer_key: Dict[str, Dict[str, object]] = {}
        self._submitted_ticket_ids: List[str] = []
        self._last_action_result = LastActionResult(success=True, message="Environment initialized")
        self._done: bool = False
        self._validation_feedback: List[str] = []

    def reset(self, task_id: Optional[str] = None) -> Dict[str, object]:
        selected_task_id = task_id or "easy"
        scenario = self._load_scenario(selected_task_id)
        self.task_id = scenario["task_id"]
        self.max_steps = int(scenario.get("max_steps", 20))
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.cumulative_reward = 0.0
        self._done = False
        self._validation_feedback = []

        self._tickets = {}
        self._decisions = {}
        for ticket_payload in scenario["tickets"]:
            ticket = TicketRecord(**ticket_payload)
            self._tickets[ticket.ticket_id] = ticket
            self._decisions[ticket.ticket_id] = TicketDecision()

        self._answer_key = scenario["answer_key"]
        self._submitted_ticket_ids = []
        self.current_ticket_id = next(iter(self._tickets.keys()), None)
        self._last_action_result = LastActionResult(
            success=True,
            message=f"Loaded task '{self.task_id}' with {len(self._tickets)} tickets",
        )

        observation = self._build_observation(self.current_ticket_id)
        return {
            "observation": observation.model_dump(mode="json"),
            "reward": STRICT_SCORE_EPSILON,
            "done": False,
            "info": {"task_id": self.task_id},
        }

    def step(self, action: TicketTriageAction) -> Dict[str, object]:
        if self._done:
            observation = self._build_observation(self.current_ticket_id)
            return {
                "observation": observation.model_dump(mode="json"),
                "reward": STRICT_SCORE_EPSILON,
                "done": True,
                "info": {"message": "Episode already done"},
            }

        self.step_count += 1
        reward_delta = 0.0
        info: Dict[str, object] = {}
        self._validation_feedback = []

        success, message, action_reward, action_info = self._apply_action(action)
        reward_delta += action_reward
        info.update(action_info)

        if self.step_count > int(self.max_steps * 0.7):
            reward_delta -= 0.01
            self._validation_feedback.append("Step budget pressure: small efficiency penalty applied")

        reward_delta = max(-0.2, min(0.2, reward_delta))
        self.cumulative_reward += reward_delta

        if action.action_type == ActionType.SUBMIT_BATCH:
            self._done = True

        if len(self._submitted_ticket_ids) == len(self._tickets):
            self._done = True

        if self.step_count >= self.max_steps:
            self._done = True
            self._validation_feedback.append("Max steps reached")

        final_score_for_step: float | None = None
        if self._done:
            final_score, breakdown = self._final_score()
            final_score_for_step = final_score
            info["final_score"] = final_score
            info["per_ticket_breakdown"] = breakdown
            info["score"] = final_score
            info["task_score"] = final_score
            if final_score >= 0.85:
                reward_delta += 0.2
                self.cumulative_reward += 0.2

        reported_reward = (
            _strict_unit_interval(final_score_for_step)
            if final_score_for_step is not None
            else _strict_unit_interval(reward_delta)
        )

        self._last_action_result = LastActionResult(success=success, message=message)

        observation = self._build_observation(self.current_ticket_id)
        return {
            "observation": observation.model_dump(mode="json"),
            "reward": reported_reward,
            "done": self._done,
            "info": info,
        }

    def state(self) -> Dict[str, object]:
        state = TicketTriageState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            current_ticket_id=self.current_ticket_id,
            submitted_ticket_ids=list(self._submitted_ticket_ids),
            cumulative_reward=self.cumulative_reward,
        )
        return state.model_dump(mode="json")

    def _load_scenario(self, task_id: str) -> Dict[str, object]:
        scenario_file = self.scenarios_dir / f"{task_id}.json"
        if not scenario_file.exists():
            raise ValueError(f"Unknown task_id '{task_id}'")

        with scenario_file.open("r", encoding="utf-8") as fh:
            scenario = json.load(fh)

        self._validate_scenario(scenario)
        return scenario

    def _validate_scenario(self, scenario: Dict[str, object]) -> None:
        tickets = scenario.get("tickets", [])
        answer_key = scenario.get("answer_key", {})

        if not isinstance(tickets, list) or not tickets:
            raise ValueError("Scenario must contain a non-empty 'tickets' list")

        if not isinstance(answer_key, dict):
            raise ValueError("Scenario must contain an 'answer_key' object")

        ticket_ids = set()
        for item in tickets:
            ticket_id = item.get("ticket_id")
            if not ticket_id:
                raise ValueError("Every ticket must include a non-empty 'ticket_id'")
            if ticket_id in ticket_ids:
                raise ValueError(f"Duplicate ticket_id in scenario: {ticket_id}")
            ticket_ids.add(ticket_id)

        missing = sorted(ticket_ids - set(answer_key.keys()))
        if missing:
            raise ValueError(f"Missing answer_key entries for tickets: {', '.join(missing)}")

    def _apply_action(self, action: TicketTriageAction) -> Tuple[bool, str, float, Dict[str, object]]:
        reward = 0.0
        info: Dict[str, object] = {}

        if action.action_type == ActionType.NOOP:
            return True, "No operation", -0.005, info

        if action.action_type == ActionType.SUBMIT_BATCH:
            return True, "Batch submitted", 0.0, info

        if not action.ticket_id:
            return False, "ticket_id is required for this action", -0.03, info

        if action.ticket_id not in self._tickets:
            return False, f"ticket_id '{action.ticket_id}' not found", -0.03, info

        self.current_ticket_id = action.ticket_id
        decision = self._decisions[action.ticket_id]
        answer = self._answer_key[action.ticket_id]

        if action.action_type == ActionType.INSPECT_TICKET:
            return True, f"Inspected {action.ticket_id}", 0.0, info

        if action.action_type == ActionType.SET_FIELDS:
            if decision.status == TicketStatus.SUBMITTED:
                reward -= 0.05
                self._validation_feedback.append("Edited a submitted ticket, loop penalty applied")

            updated_any = False
            if action.category is not None:
                reward += self._field_reward("category", decision.category, action.category.value, answer)
                decision.category = action.category
                updated_any = True
            if action.priority is not None:
                reward += self._field_reward("priority", decision.priority, action.priority.value, answer)
                decision.priority = action.priority
                updated_any = True
            if action.queue is not None:
                queue_reward = self._field_reward("queue", decision.queue, action.queue.value, answer)
                if answer.get("high_risk_misroute", False) and action.queue.value != answer.get("queue"):
                    queue_reward -= 0.08
                reward += queue_reward
                decision.queue = action.queue
                updated_any = True
            if action.next_action is not None:
                reward += self._field_reward("next_action", decision.next_action, action.next_action.value, answer)
                decision.next_action = action.next_action
                updated_any = True

            if not updated_any:
                return False, "SET_FIELDS requires at least one field", -0.03, info

            decision.status = TicketStatus.IN_PROGRESS
            return True, f"Updated triage fields for {action.ticket_id}", reward, info

        if action.action_type == ActionType.DRAFT_RESPONSE:
            if not action.response_text:
                return False, "response_text is required", -0.03, info

            previous = decision.response_text
            decision.response_text = action.response_text
            decision.status = TicketStatus.IN_PROGRESS
            compliance = score_ticket(decision, answer)["response"]
            if compliance >= 0.8:
                reward += 0.03
            elif compliance >= 0.5:
                reward += 0.01
            else:
                reward -= 0.01

            if previous and previous.strip().lower() == action.response_text.strip().lower():
                reward -= 0.01

            info["response_compliance"] = compliance
            return True, f"Drafted response for {action.ticket_id}", reward, info

        if action.action_type == ActionType.SUBMIT_TICKET:
            required_missing = []
            if decision.category is None:
                required_missing.append("category")
            if decision.priority is None:
                required_missing.append("priority")
            if decision.queue is None:
                required_missing.append("queue")
            if decision.next_action is None:
                required_missing.append("next_action")
            if not decision.response_text:
                required_missing.append("response_text")

            if required_missing:
                self._validation_feedback.append(
                    f"Cannot submit ticket. Missing fields: {', '.join(required_missing)}"
                )
                return False, "Ticket submission blocked: incomplete fields", -0.03, info

            decision.status = TicketStatus.SUBMITTED
            if action.ticket_id not in self._submitted_ticket_ids:
                self._submitted_ticket_ids.append(action.ticket_id)

            if len(self._submitted_ticket_ids) < len(self._tickets):
                for ticket_id in self._tickets:
                    if ticket_id not in self._submitted_ticket_ids:
                        self.current_ticket_id = ticket_id
                        break

            return True, f"Submitted ticket {action.ticket_id}", 0.02, info

        return False, f"Unsupported action_type: {action.action_type.value}", -0.03, info

    def _field_reward(self, field_name: str, previous_value: object, new_value: str, answer: Dict[str, object]) -> float:
        expected = answer.get(field_name)
        if expected is None:
            return 0.0

        prev_str = None
        if previous_value is not None:
            prev_str = previous_value.value if hasattr(previous_value, "value") else str(previous_value)

        if new_value == expected and prev_str is None:
            return 0.05

        if new_value == expected:
            return 0.02 if prev_str != expected else 0.0

        return -0.01

    def _final_score(self) -> Tuple[float, Dict[str, Dict[str, float]]]:
        breakdown: Dict[str, Dict[str, float]] = {}
        scores: List[Dict[str, float]] = []
        unresolved_urgent = 0

        for ticket_id in sorted(self._tickets.keys()):
            decision = self._decisions[ticket_id]
            answer = self._answer_key[ticket_id]
            item_score = score_ticket(decision, answer)
            item_score["total"] = _strict_unit_interval(float(item_score.get("total", 0.0)))
            breakdown[ticket_id] = item_score
            scores.append(item_score)

            if answer.get("priority") == "urgent" and decision.status != TicketStatus.SUBMITTED:
                unresolved_urgent += 1

        return score_batch(scores, unresolved_urgent_count=unresolved_urgent), breakdown

    def _build_observation(self, ticket_id: Optional[str]) -> TicketTriageObservation:
        queue_counts: Dict[str, int] = {queue.value: 0 for queue in Queue}
        untriaged = 0
        urgent_remaining = 0

        for tid, decision in self._decisions.items():
            if decision.status != TicketStatus.SUBMITTED:
                untriaged += 1
            if decision.queue is not None:
                queue_counts[decision.queue.value] += 1

            answer = self._answer_key.get(tid, {})
            if answer.get("priority") in {"high", "urgent"} and decision.status != TicketStatus.SUBMITTED:
                urgent_remaining += 1

        queue_snapshot = QueueSnapshot(
            total_tickets=len(self._tickets),
            submitted_tickets=len(self._submitted_ticket_ids),
            untriaged_tickets=untriaged,
            high_or_urgent_remaining=urgent_remaining,
            by_queue=queue_counts,
        )

        current_ticket = self._tickets.get(ticket_id) if ticket_id else None
        current_decision = self._decisions.get(ticket_id) if ticket_id else None
        ticket_view = TicketView(ticket=current_ticket, decision=current_decision)

        provisional_score, _ = self._final_score()
        progress = ProgressMetrics(
            current_task=self.task_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            cumulative_reward=self.cumulative_reward,
            provisional_score=provisional_score,
        )

        return TicketTriageObservation(
            queue_snapshot=queue_snapshot,
            ticket_view=ticket_view,
            validation_feedback=list(self._validation_feedback),
            progress_metrics=progress,
            last_action_result=self._last_action_result,
        )
