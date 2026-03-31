from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
from openai import OpenAI

from ticket_triage_env.client import TicketTriageEnvClient
from ticket_triage_env.models import ActionType, TicketTriageAction

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a support triage agent. Return JSON only with keys:
action_type, ticket_id, category, priority, queue, next_action, response_text.
Valid action_type values: inspect_ticket, set_fields, draft_response, submit_ticket, submit_batch, noop.
When setting fields, include only relevant keys. Keep response_text concise and policy-safe.
""".strip()


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
    else:
        default_api_base = "https://api.openai.com/v1"

    api_base = os.getenv("LLM_API_BASE_URL") or os.getenv("API_BASE_URL") or default_api_base
    env_base = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    max_steps = int(os.getenv("MAX_STEPS", "70"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    seed = int(os.getenv("SEED", "42"))

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "env_base": env_base,
        "max_steps": max_steps,
        "temperature": temperature,
        "seed": seed,
    }


def call_model(client: OpenAI, observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_payload = {
        "instruction": "Choose the next best triage action.",
        "observation": observation,
    }

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
        return json.loads(content)
    except json.JSONDecodeError:
        return {"action_type": "noop"}


def run_task(client: OpenAI, task_id: str, config: Dict[str, Any]) -> float:
    env_client = TicketTriageEnvClient(base_url=config["env_base"])
    result = env_client.reset(task_id=task_id)

    final_score = 0.0
    for _ in range(config["max_steps"]):
        if result.get("done"):
            final_score = float(result.get("info", {}).get("final_score", 0.0))
            break

        observation = result["observation"]
        action = call_model(client, observation, config)
        try:
            typed_action = TicketTriageAction(**action)
        except Exception:
            typed_action = TicketTriageAction(action_type=ActionType.NOOP)

        try:
            result = env_client.step(typed_action)
        except Exception:
            result = env_client.step(TicketTriageAction(action_type=ActionType.NOOP))

    if not result.get("done"):
        batch_result = env_client.step(TicketTriageAction(action_type=ActionType.SUBMIT_BATCH))
        final_score = float(batch_result.get("info", {}).get("final_score", 0.0))

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

    if not config["api_key"]:
        raise RuntimeError("LLM_API_KEY is required (or set OPENAI_API_KEY for backward compatibility)")

    client = OpenAI(base_url=config["api_base"], api_key=config["api_key"])
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
        f"seed={config['seed']}, temp={config['temperature']}, env={config['env_base']}"
    )


if __name__ == "__main__":
    main()
