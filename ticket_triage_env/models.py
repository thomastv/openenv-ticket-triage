from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    BILLING_DISPUTE = "billing_dispute"
    ACCESS_ISSUE = "access_issue"
    OUTAGE_REPORT = "outage_report"
    ABUSE_REPORT = "abuse_report"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Queue(str, Enum):
    BILLING_OPS = "billing_ops"
    TECH_SUPPORT = "tech_support"
    CSM = "csm"
    TRUST_SAFETY = "trust_safety"


class NextAction(str, Enum):
    REQUEST_INFO = "request_info"
    PROVIDE_STEPS = "provide_steps"
    ESCALATE_ENGINEERING = "escalate_engineering"
    ESCALATE_SECURITY = "escalate_security"
    REFUND_REVIEW = "refund_review"
    CLOSE = "close"


class TicketStatus(str, Enum):
    UNTRIAGED = "untriaged"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"


class ActionType(str, Enum):
    INSPECT_TICKET = "inspect_ticket"
    SET_FIELDS = "set_fields"
    DRAFT_RESPONSE = "draft_response"
    SUBMIT_TICKET = "submit_ticket"
    SUBMIT_BATCH = "submit_batch"
    NOOP = "noop"


class TicketRecord(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    product_area: str
    sentiment: str
    sla_hours_remaining: int
    language: str = "en"
    account_health_flags: List[str] = Field(default_factory=list)
    prior_interactions: int = 0


class TicketDecision(BaseModel):
    category: Optional[TicketCategory] = None
    priority: Optional[Priority] = None
    queue: Optional[Queue] = None
    next_action: Optional[NextAction] = None
    response_text: Optional[str] = None
    status: TicketStatus = TicketStatus.UNTRIAGED


class TicketTriageAction(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    category: Optional[TicketCategory] = None
    priority: Optional[Priority] = None
    queue: Optional[Queue] = None
    next_action: Optional[NextAction] = None
    response_text: Optional[str] = None


class QueueSnapshot(BaseModel):
    total_tickets: int
    submitted_tickets: int
    untriaged_tickets: int
    high_or_urgent_remaining: int
    by_queue: Dict[str, int] = Field(default_factory=dict)


class LastActionResult(BaseModel):
    success: bool
    message: str


class TicketView(BaseModel):
    ticket: Optional[TicketRecord] = None
    decision: Optional[TicketDecision] = None


class ProgressMetrics(BaseModel):
    current_task: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    provisional_score: float


class TicketTriageObservation(BaseModel):
    queue_snapshot: QueueSnapshot
    ticket_view: TicketView
    validation_feedback: List[str] = Field(default_factory=list)
    progress_metrics: ProgressMetrics
    last_action_result: LastActionResult


class TicketTriageState(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    current_ticket_id: Optional[str] = None
    submitted_ticket_ids: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
