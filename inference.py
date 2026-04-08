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

DEFAULT_TASK_NAME = "easy"
DEFAULT_BASELINE_TASKS = ["easy", "medium", "hard"]
DEFAULT_BENCHMARK = "ticket_triage"
DEFAULT_ENV_BASE_URL = "http://localhost:8000"
DEFAULT_HF_SPACE_BASE_URL = "https://thomastv-openenv-customer-ticket-triage.hf.space"
DEFAULT_MAX_STEPS = 70
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_RETRY_MAX = 2
DEFAULT_RETRY_SECONDS = 35.0
DEFAULT_MAX_LLM_CALLS_PER_TASK = 8
DEFAULT_INFERENCE_LOG_TO_FILE = False
DEFAULT_INFERENCE_LOG_FILE_PATH = "logs/inference.log"
STRICT_SCORE_EPSILON = 1e-6


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _parse_csv_env(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_space_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if "huggingface.co/spaces/" in base:
        parts = base.split("huggingface.co/spaces/", 1)[1].split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"https://{parts[0]}-{parts[1]}.hf.space"
    return base


def _parse_bool_env(*names: str, default: bool = False) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _strict_unit_interval(value: float) -> float:
    if value <= 0.0:
        return STRICT_SCORE_EPSILON
    if value >= 1.0:
        return 1.0 - STRICT_SCORE_EPSILON
    return value

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
    model = _first_non_empty(
        os.getenv("MODEL_NAME"),
        os.getenv("LLM_MODEL"),
        os.getenv("OPENAI_MODEL"),
        "Qwen/Qwen2.5-72B-Instruct",
    )
    api_key = _first_non_empty(
        os.getenv("HF_TOKEN"),
        os.getenv("API_KEY"),
        os.getenv("LLM_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    )
    local_image_name = _first_non_empty(os.getenv("LOCAL_IMAGE_NAME"), os.getenv("IMAGE_NAME"))

    missing: List[str] = []
    if not api_key:
        missing.append("HF_TOKEN (or LLM_API_KEY / OPENAI_API_KEY)")

    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    provider = _first_non_empty(os.getenv("LLM_PROVIDER"), "openai").lower()

    if provider == "gemini":
        default_api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
    elif provider == "ollama":
        default_api_base = "http://localhost:11434/v1"
    else:
        default_api_base = "https://router.huggingface.co/v1"

    # Explicit override wins, then primary API_BASE_URL, then provider-derived default.
    api_base = _first_non_empty(
        os.getenv("LLM_API_BASE_URL"),
        os.getenv("API_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
        default_api_base,
    )

    env_base = _first_non_empty(os.getenv("ENV_BASE_URL"), os.getenv("OPENENV_ENV_BASE_URL"), DEFAULT_ENV_BASE_URL)
    hf_space_url = _normalize_space_url(
        _first_non_empty(os.getenv("HF_SPACE_URL"), DEFAULT_HF_SPACE_BASE_URL)
    )
    env_base_fallbacks = _parse_csv_env(_first_non_empty(os.getenv("ENV_BASE_URL_FALLBACKS"), ""))
    if hf_space_url and hf_space_url not in env_base_fallbacks and hf_space_url != env_base:
        env_base_fallbacks.append(hf_space_url)

    task_name = _first_non_empty(os.getenv("TASK_NAME"), os.getenv("OPENENV_TASK"), DEFAULT_TASK_NAME)
    tasks_csv = (os.getenv("TASKS") or ",".join(DEFAULT_BASELINE_TASKS)).strip()
    tasks = [item.strip() for item in tasks_csv.split(",") if item.strip()]
    run_all_tasks = _parse_bool_env("RUN_ALL_TASKS", default=True)
    benchmark = _first_non_empty(os.getenv("BENCHMARK"), os.getenv("OPENENV_BENCHMARK"), DEFAULT_BENCHMARK)
    max_steps = DEFAULT_MAX_STEPS
    temperature = DEFAULT_TEMPERATURE
    seed = DEFAULT_SEED
    retry_max = DEFAULT_RETRY_MAX
    retry_seconds = DEFAULT_RETRY_SECONDS
    max_llm_calls_per_task = DEFAULT_MAX_LLM_CALLS_PER_TASK
    verbose = _parse_bool_env("INFERENCE_VERBOSE", default=False)
    inference_log_to_file = DEFAULT_INFERENCE_LOG_TO_FILE
    inference_log_file_path = DEFAULT_INFERENCE_LOG_FILE_PATH

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "local_image_name": local_image_name,
        "env_base": env_base,
        "env_base_fallbacks": env_base_fallbacks,
        "task_name": task_name,
        "tasks": tasks,
        "run_all_tasks": run_all_tasks,
        "benchmark": benchmark,
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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def log_start(task_name: str, benchmark: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={benchmark} model={model_name}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    reward_text = _format_strict_score(reward)
    print(
        f"[STEP] step={step} action={action} reward={reward_text} done={_bool_text(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(_format_strict_score(r) for r in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} rewards={rewards_text}", flush=True)


def _format_strict_score(value: float) -> str:
    """Format scores without rounding strict in-range values to 0.000/1.000."""
    return f"{_strict_unit_interval(value):.6f}"


def log_baseline(scores: Dict[str, float], seed: int, temperature: float) -> None:
    ordered_tasks = [task for task in DEFAULT_BASELINE_TASKS if task in scores]
    extras = sorted(task for task in scores.keys() if task not in ordered_tasks)
    ordered_tasks.extend(extras)
    normalized_scores = {task: _strict_unit_interval(float(value)) for task, value in scores.items()}
    overall = (
        _strict_unit_interval(sum(normalized_scores.values()) / len(normalized_scores))
        if normalized_scores
        else STRICT_SCORE_EPSILON
    )
    parts = [f"{task}={_format_strict_score(normalized_scores[task])}" for task in ordered_tasks]
    parts.append(f"overall={_format_strict_score(overall)}")
    print(f"[BASELINE] {' '.join(parts)}", flush=True)


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
                if attempt >= config["retry_max"]:
                    break
                time.sleep(config["retry_seconds"])
                continue
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
    log_start(task_name=task_id, benchmark=config["benchmark"], model_name=config["model"])

    if config["verbose"]:
        LOGGER.info("env_request method=POST path=/reset task_id=%s", task_id)
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        result = env_client.reset(task_id=task_id)

        ticket_plans: Dict[str, Dict[str, Any]] = {}
        llm_calls = 0
        for step_idx in range(config["max_steps"]):
            if result.get("done"):
                final_score = _strict_unit_interval(float(result.get("info", {}).get("final_score", 0.0)))
                if config["verbose"]:
                    LOGGER.info(
                        "task_done task_id=%s step=%s final_score=%s",
                        task_id,
                        step_idx,
                        _format_strict_score(final_score),
                    )
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

            step_error: str | None = None
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
            except Exception as exc:
                LOGGER.exception(
                    "env_step_failed task_id=%s step=%s action_type=%s",
                    task_id,
                    step_idx + 1,
                    typed_action.action_type.value,
                )
                result = env_client.step(TicketTriageAction(action_type=ActionType.NOOP))
                step_error = str(exc)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken = step_idx + 1

            if step_error is None:
                step_error = result.get("observation", {}).get("last_action_error")

            action_repr = typed_action.action_type.value
            if typed_action.ticket_id:
                action_repr = f"{action_repr}:{typed_action.ticket_id}"
            log_step(
                step=steps_taken,
                action=action_repr,
                reward=reward,
                done=done,
                error=step_error,
            )

            if config["verbose"]:
                LOGGER.info(
                    "env_response task_id=%s step=%s reward=%.3f done=%s",
                    task_id,
                    steps_taken,
                    reward,
                    done,
                )

            if done:
                final_score = float(result.get("info", {}).get("final_score", 0.0))
                break

        if not result.get("done"):
            if config["verbose"]:
                LOGGER.info("env_request method=POST path=/step task_id=%s action_type=submit_batch", task_id)
            batch_result = env_client.step(TicketTriageAction(action_type=ActionType.SUBMIT_BATCH))
            final_score = _strict_unit_interval(float(batch_result.get("info", {}).get("final_score", 0.0)))
            forced_reward = float(batch_result.get("reward", 0.0))
            rewards.append(forced_reward)
            steps_taken += 1
            forced_done = bool(batch_result.get("done", False))
            log_step(
                step=steps_taken,
                action=ActionType.SUBMIT_BATCH.value,
                reward=forced_reward,
                done=forced_done,
                error=None,
            )
            if config["verbose"]:
                LOGGER.info(
                    "task_done_forced task_id=%s final_score=%s",
                    task_id,
                    _format_strict_score(final_score),
                )

        final_score = _strict_unit_interval(float(final_score))
        success_threshold = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
        success = final_score >= success_threshold
        return final_score
    finally:
        env_client.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


def check_environment_server(config: Dict[str, Any]) -> tuple[bool, str | None]:
    """Validate environment server connectivity before inference runs."""
    base_url = config["env_base"].rstrip("/")
    health_url = f"{base_url}/health"
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        return True, None
    except requests.RequestException as exc:
        return False, (
            "Unable to reach environment server at "
            f"{base_url}. Start it first with: "
            "`uvicorn ticket_triage_env.server.app:app --host 0.0.0.0 --port 8000` "
            "or run the Docker container and set ENV_BASE_URL accordingly. "
            f"Original error: {exc}"
        )


def select_reachable_env_base(config: Dict[str, Any]) -> tuple[str | None, str | None]:
    candidates = [config["env_base"], *config.get("env_base_fallbacks", [])]
    seen = set()
    errors: List[str] = []

    for raw in candidates:
        candidate = _normalize_space_url(raw)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        probe_cfg = {**config, "env_base": candidate}
        ok, err = check_environment_server(probe_cfg)
        if ok:
            return candidate, None
        if err:
            errors.append(f"{candidate}: {err}")

    if errors:
        return None, " | ".join(errors)
    return None, "No environment base URL candidates configured"


def main() -> None:
    load_env_file()
    config: Dict[str, Any]
    try:
        config = resolve_config()
    except Exception as exc:
        task_name = os.getenv("TASK_NAME") or DEFAULT_TASK_NAME
        benchmark = os.getenv("BENCHMARK") or DEFAULT_BENCHMARK
        model_name = os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL") or "unknown"
        log_start(task_name=task_name, benchmark=benchmark, model_name=model_name)
        log_end(success=False, steps=0, rewards=[])
        LOGGER.error("inference_config_failed error=%s", str(exc))
        return

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
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)

    try:
        client = OpenAI(base_url=config["api_base"], api_key=config["api_key"])
    except Exception as exc:
        log_start(task_name=config["task_name"], benchmark=config["benchmark"], model_name=config["model"])
        log_end(success=False, steps=0, rewards=[])
        LOGGER.error("inference_client_init_failed error=%s", str(exc))
        return
    if config["verbose"]:
        LOGGER.info(
            "inference_start provider=%s model=%s api_base=%s env_base=%s",
            config["provider"],
            config["model"],
            config["api_base"],
            config["env_base"],
        )
    selected_env_base, env_error = select_reachable_env_base(config)
    if selected_env_base:
        config["env_base"] = selected_env_base
    elif env_error:
        LOGGER.error("environment_preflight_failed error=%s", env_error)

    if config["run_all_tasks"]:
        baseline_scores: Dict[str, float] = {}
        for task in config["tasks"]:
            try:
                baseline_scores[task] = run_task(client, task, config)
            except Exception as exc:
                LOGGER.error("task_run_failed task=%s error=%s", task, str(exc))
                baseline_scores[task] = STRICT_SCORE_EPSILON
        log_baseline(baseline_scores, seed=config["seed"], temperature=config["temperature"])
    else:
        try:
            run_task(client, config["task_name"], config)
        except Exception as exc:
            LOGGER.error("task_run_failed task=%s error=%s", config["task_name"], str(exc))


if __name__ == "__main__":
    main()
