from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI, RateLimitError

from ticket_triage_env.client import TicketTriageEnvClient
from ticket_triage_env.models import ActionType, TicketTriageAction

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a support triage agent. Return JSON only with keys:
action_type, ticket_id, category, priority, queue, next_action, response_text.
Valid action_type values: inspect_ticket, set_fields, draft_response, submit_ticket, submit_batch, noop.
When setting fields, include only relevant keys. Keep response_text concise and policy-safe.
""".strip()

LOGGER = logging.getLogger("inference")

CATEGORY_MAP = {
    "billing": "billing_dispute",
    "billing_dispute": "billing_dispute",
    "invoice": "billing_dispute",
    "payment": "billing_dispute",
    "access": "access_issue",
    "access_issue": "access_issue",
    "login": "access_issue",
    "auth": "access_issue",
    "outage": "outage_report",
    "outage_report": "outage_report",
    "incident": "outage_report",
    "abuse": "abuse_report",
    "abuse_report": "abuse_report",
    "security": "abuse_report",
    "feature": "feature_request",
    "feature_request": "feature_request",
    "bug": "bug_report",
    "bug_report": "bug_report",
    "other": "other",
}

PRIORITY_MAP = {
    "low": "low",
    "medium": "medium",
    "med": "medium",
    "high": "high",
    "urgent": "urgent",
    "critical": "urgent",
    "p1": "urgent",
    "p2": "high",
    "p3": "medium",
    "p4": "low",
}

QUEUE_MAP = {
    "billing": "billing_ops",
    "billing_ops": "billing_ops",
    "billing team": "billing_ops",
    "tech": "tech_support",
    "tech_support": "tech_support",
    "engineering": "tech_support",
    "support": "tech_support",
    "csm": "csm",
    "customer_success": "csm",
    "trust": "trust_safety",
    "trust_safety": "trust_safety",
    "security": "trust_safety",
}

NEXT_ACTION_MAP = {
    "request_info": "request_info",
    "ask_for_info": "request_info",
    "provide_steps": "provide_steps",
    "troubleshoot": "provide_steps",
    "escalate_engineering": "escalate_engineering",
    "escalate_to_engineering": "escalate_engineering",
    "escalate_security": "escalate_security",
    "escalate_to_security": "escalate_security",
    "refund_review": "refund_review",
    "refund": "refund_review",
    "close": "close",
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return text


def _normalize_plan(raw_plan: Dict[str, Any], ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize potentially non-canonical model outputs into valid enum values."""
    fallback = _heuristic_plan_from_ticket(ticket)

    category = CATEGORY_MAP.get(_norm(raw_plan.get("category")), fallback["category"])
    priority = PRIORITY_MAP.get(_norm(raw_plan.get("priority")), fallback["priority"])
    queue = QUEUE_MAP.get(_norm(raw_plan.get("queue")), "")
    next_action = NEXT_ACTION_MAP.get(_norm(raw_plan.get("next_action")), fallback["next_action"])
    response_text = raw_plan.get("response_text") or fallback["response_text"]

    # Common LLM mistake: put priority into queue (e.g., queue='urgent').
    queue_norm = _norm(raw_plan.get("queue"))
    if queue_norm in PRIORITY_MAP and _norm(raw_plan.get("priority")) not in PRIORITY_MAP:
        priority = PRIORITY_MAP[queue_norm]

    if not queue:
        if category == "billing_dispute":
            queue = "billing_ops"
        elif category in {"access_issue", "outage_report", "bug_report"}:
            queue = "tech_support"
        elif category == "abuse_report":
            queue = "trust_safety"
        else:
            queue = "csm"

    return {
        "category": category,
        "priority": priority,
        "queue": queue,
        "next_action": next_action,
        "response_text": response_text,
    }


def load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into process env.

    Existing environment variables are preserved and will not be overwritten.
    """
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_config() -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

    if provider == "gemini":
        default_api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
    elif provider == "ollama":
        default_api_base = "http://localhost:11434/v1"
    else:
        default_api_base = "https://api.openai.com/v1"

    api_base = os.getenv("LLM_API_BASE_URL") or os.getenv("API_BASE_URL") or default_api_base
    env_base = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    max_steps = int(os.getenv("MAX_STEPS", "70"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    seed = int(os.getenv("SEED", "42"))

    retry_max = int(os.getenv("LLM_RETRY_MAX", "2"))
    retry_seconds = float(os.getenv("LLM_RETRY_SECONDS", "35"))
    max_llm_calls_per_task = int(os.getenv("MAX_LLM_CALLS_PER_TASK", "8"))
    verbose = os.getenv("INFERENCE_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}
    inference_log_to_file = os.getenv("INFERENCE_LOG_TO_FILE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    inference_log_file_path = os.getenv("INFERENCE_LOG_FILE_PATH", "logs/inference.log")

    if provider == "ollama" and not api_key:
        # OpenAI-compatible Ollama endpoints typically accept any non-empty key.
        api_key = "ollama"

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "env_base": env_base,
        "max_steps": max_steps,
        "temperature": temperature,
        "seed": seed,
        "retry_max": retry_max,
        "retry_seconds": retry_seconds,
        "max_llm_calls_per_task": max_llm_calls_per_task,
        "verbose": verbose,
        "inference_log_to_file": inference_log_to_file,
        "inference_log_file_path": inference_log_file_path,
    }


def _extract_retry_seconds(exc: Exception, default_wait: float) -> float:
    text = str(exc)
    match = re.search(r"retry in\s+([0-9.]+)s", text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return default_wait
    return default_wait


def _heuristic_plan_from_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    subject = (ticket.get("subject") or "").lower()
    body = (ticket.get("body") or "").lower()
    text = f"{subject} {body}"

    category = "other"
    priority = "medium"
    queue = "csm"
    next_action = "request_info"

    if any(k in text for k in ["refund", "invoice", "charged", "billing", "vat", "payment"]):
        category = "billing_dispute"
        queue = "billing_ops"
        next_action = "refund_review"
        priority = "high"
    elif any(k in text for k in ["login", "password", "token", "access"]):
        category = "access_issue"
        queue = "tech_support"
        next_action = "provide_steps"
        priority = "medium"
    elif any(k in text for k in ["outage", "down", "timeout", "500", "unavailable", "incident"]):
        category = "outage_report"
        queue = "tech_support"
        next_action = "escalate_engineering"
        priority = "urgent"
    elif any(k in text for k in ["abuse", "harassment", "suspicious", "takeover", "unknown ip", "security"]):
        category = "abuse_report"
        queue = "trust_safety"
        next_action = "escalate_security"
        priority = "high"
    elif any(k in text for k in ["bug", "error", "fails", "failure"]):
        category = "bug_report"
        queue = "tech_support"
        next_action = "provide_steps"
        priority = "medium"
    elif any(k in text for k in ["feature", "request", "roadmap"]):
        category = "feature_request"
        queue = "csm"
        next_action = "request_info"
        priority = "low"

    if ticket.get("sla_hours_remaining", 999) <= 3 and priority in {"medium", "high"}:
        priority = "urgent" if priority == "high" else "high"

    response_text = (
        "Thank you for reaching out. We understand the issue and will help right away. "
        f"This has been triaged as {category.replace('_', ' ')} and routed to {queue.replace('_', ' ')}. "
        "Next, our team will follow up with concrete steps and an update shortly."
    )

    return {
        "category": category,
        "priority": priority,
        "queue": queue,
        "next_action": next_action,
        "response_text": response_text,
    }


def call_model_for_ticket(
    client: OpenAI,
    ticket: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    user_payload = {
        "instruction": (
            "Return triage fields only as JSON with keys: "
            "category, priority, queue, next_action, response_text"
        ),
        "ticket": ticket,
    }

    last_error: Exception | None = None
    for attempt in range(config["retry_max"] + 1):
        try:
            if config["verbose"]:
                LOGGER.info(
                    "llm_request ticket_id=%s attempt=%s/%s",
                    ticket.get("ticket_id"),
                    attempt + 1,
                    config["retry_max"] + 1,
                )
            completion = client.chat.completions.create(
                model=config["model"],
                temperature=config["temperature"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.error(
                    "llm_response_parse_failed ticket_id=%s error=%s raw_preview=%s",
                    ticket.get("ticket_id"),
                    str(exc),
                    content[:200],
                )
                break
            raw_plan = {
                "category": parsed.get("category"),
                "priority": parsed.get("priority"),
                "queue": parsed.get("queue"),
                "next_action": parsed.get("next_action"),
                "response_text": parsed.get("response_text"),
            }
            normalized_plan = _normalize_plan(raw_plan, ticket)
            if config["verbose"] and normalized_plan != raw_plan:
                LOGGER.info(
                    "llm_plan_normalized ticket_id=%s raw=%s normalized=%s",
                    ticket.get("ticket_id"),
                    raw_plan,
                    normalized_plan,
                )
            return normalized_plan
        except RateLimitError as exc:
            last_error = exc
            wait_seconds = _extract_retry_seconds(exc, config["retry_seconds"])
            LOGGER.error(
                "llm_rate_limited ticket_id=%s attempt=%s/%s wait_seconds=%.2f error=%s",
                ticket.get("ticket_id"),
                attempt + 1,
                config["retry_max"] + 1,
                wait_seconds,
                str(exc),
            )
            if attempt >= config["retry_max"]:
                break
            time.sleep(wait_seconds)
        except Exception as exc:
            last_error = exc
            LOGGER.error(
                "llm_request_failed ticket_id=%s error=%s",
                ticket.get("ticket_id"),
                str(exc),
            )
            break

    # Fallback guarantees progress when rate-limited or response malformed.
    LOGGER.error(
        "llm_fallback_heuristic ticket_id=%s reason=%s",
        ticket.get("ticket_id"),
        type(last_error).__name__ if last_error else "unknown",
    )
    return _heuristic_plan_from_ticket(ticket)


def run_task(client: OpenAI, task_id: str, config: Dict[str, Any]) -> float:
    env_client = TicketTriageEnvClient(base_url=config["env_base"])
    if config["verbose"]:
        LOGGER.info("env_request method=POST path=/reset task_id=%s", task_id)
    result = env_client.reset(task_id=task_id)

    ticket_plans: Dict[str, Dict[str, Any]] = {}
    llm_calls = 0
    final_score = 0.0
    for step_idx in range(config["max_steps"]):
        if result.get("done"):
            final_score = float(result.get("info", {}).get("final_score", 0.0))
            if config["verbose"]:
                LOGGER.info("task_done task_id=%s step=%s final_score=%.3f", task_id, step_idx, final_score)
            break

        observation = result["observation"]
        ticket = observation.get("ticket_view", {}).get("ticket") or {}
        decision = observation.get("ticket_view", {}).get("decision") or {}
        ticket_id = ticket.get("ticket_id")

        if not ticket_id:
            typed_action = TicketTriageAction(action_type=ActionType.SUBMIT_BATCH)
        elif ticket_id not in ticket_plans:
            if llm_calls < config["max_llm_calls_per_task"]:
                ticket_plans[ticket_id] = call_model_for_ticket(client, ticket, config)
                llm_calls += 1
            else:
                if config["verbose"]:
                    LOGGER.info(
                        "llm_budget_exceeded task_id=%s ticket_id=%s max_llm_calls_per_task=%s",
                        task_id,
                        ticket_id,
                        config["max_llm_calls_per_task"],
                    )
                ticket_plans[ticket_id] = _heuristic_plan_from_ticket(ticket)

            typed_action = TicketTriageAction(action_type=ActionType.INSPECT_TICKET, ticket_id=ticket_id)
        else:
            plan = ticket_plans[ticket_id]
            if not decision.get("category"):
                try:
                    typed_action = TicketTriageAction(
                        action_type=ActionType.SET_FIELDS,
                        ticket_id=ticket_id,
                        category=plan.get("category"),
                        priority=plan.get("priority"),
                        queue=plan.get("queue"),
                        next_action=plan.get("next_action"),
                    )
                except Exception:
                    LOGGER.exception("set_fields_plan_invalid ticket_id=%s plan=%s", ticket_id, plan)
                    fallback_plan = _heuristic_plan_from_ticket(ticket)
                    typed_action = TicketTriageAction(
                        action_type=ActionType.SET_FIELDS,
                        ticket_id=ticket_id,
                        category=fallback_plan.get("category"),
                        priority=fallback_plan.get("priority"),
                        queue=fallback_plan.get("queue"),
                        next_action=fallback_plan.get("next_action"),
                    )
            elif not decision.get("response_text"):
                typed_action = TicketTriageAction(
                    action_type=ActionType.DRAFT_RESPONSE,
                    ticket_id=ticket_id,
                    response_text=plan.get("response_text"),
                )
            elif decision.get("status") != "submitted":
                typed_action = TicketTriageAction(action_type=ActionType.SUBMIT_TICKET, ticket_id=ticket_id)
            else:
                typed_action = TicketTriageAction(action_type=ActionType.NOOP)

        try:
            if config["verbose"]:
                LOGGER.info(
                    "env_request method=POST path=/step task_id=%s step=%s action_type=%s ticket_id=%s",
                    task_id,
                    step_idx + 1,
                    typed_action.action_type.value,
                    typed_action.ticket_id,
                )
            result = env_client.step(typed_action)
            if config["verbose"]:
                LOGGER.info(
                    "env_response task_id=%s step=%s reward=%.3f done=%s",
                    task_id,
                    step_idx + 1,
                    float(result.get("reward", 0.0)),
                    bool(result.get("done", False)),
                )
        except Exception:
            LOGGER.exception(
                "env_step_failed task_id=%s step=%s action_type=%s",
                task_id,
                step_idx + 1,
                typed_action.action_type.value,
            )
            result = env_client.step(TicketTriageAction(action_type=ActionType.NOOP))

    if not result.get("done"):
        if config["verbose"]:
            LOGGER.info("env_request method=POST path=/step task_id=%s action_type=submit_batch", task_id)
        batch_result = env_client.step(TicketTriageAction(action_type=ActionType.SUBMIT_BATCH))
        final_score = float(batch_result.get("info", {}).get("final_score", 0.0))
        if config["verbose"]:
            LOGGER.info("task_done_forced task_id=%s final_score=%.3f", task_id, final_score)

    return float(final_score)


def check_environment_server(config: Dict[str, Any]) -> None:
    """Validate environment server connectivity before inference runs."""
    base_url = config["env_base"].rstrip("/")
    health_url = f"{base_url}/health"
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to reach environment server at "
            f"{base_url}. Start it first with: "
            "`uvicorn ticket_triage_env.server.app:app --host 0.0.0.0 --port 8000` "
            "or run the Docker container and set ENV_BASE_URL accordingly. "
            f"Original error: {exc}"
        ) from exc


def main() -> None:
    load_env_file()
    config = resolve_config()

    default_level = "INFO" if config["verbose"] else "ERROR"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if config["inference_log_to_file"]:
        log_path = config["inference_log_file_path"]
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", default_level).upper(), logging.ERROR),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=handlers,
        force=True,
    )

    if not config["api_key"]:
        raise RuntimeError("LLM_API_KEY is required (or set OPENAI_API_KEY for backward compatibility)")

    client = OpenAI(base_url=config["api_base"], api_key=config["api_key"])
    if config["verbose"]:
        LOGGER.info(
            "inference_start provider=%s model=%s api_base=%s env_base=%s",
            config["provider"],
            config["model"],
            config["api_base"],
            config["env_base"],
        )
    check_environment_server(config)

    scores: Dict[str, float] = {}
    for task in TASKS:
        scores[task] = run_task(client, task, config)

    overall = sum(scores.values()) / len(scores)
    print("Baseline Scores")
    print("==============")
    for task in TASKS:
        print(f"{task:>6}: {scores[task]:.3f}")
    print(f"overall: {overall:.3f}")
    print(
        "metadata: "
        f"provider={config['provider']}, model={config['model']}, api_base={config['api_base']}, "
        f"seed={config['seed']}, temp={config['temperature']}, env={config['env_base']}, "
        f"max_llm_calls_per_task={config['max_llm_calls_per_task']}"
    )


if __name__ == "__main__":
    main()
