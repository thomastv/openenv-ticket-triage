from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
from openai import OpenAI

from ticket_triage_env.models import ActionType, TicketTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", "70"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
SEED = int(os.getenv("SEED", "42"))

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a support triage agent. Return JSON only with keys:
action_type, ticket_id, category, priority, queue, next_action, response_text.
Valid action_type values: inspect_ticket, set_fields, draft_response, submit_ticket, submit_batch, noop.
When setting fields, include only relevant keys. Keep response_text concise and policy-safe.
""".strip()


def call_model(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    user_payload = {
        "instruction": "Choose the next best triage action.",
        "observation": observation,
    }

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
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


def run_task(client: OpenAI, task_id: str) -> float:
    reset_response = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    reset_response.raise_for_status()
    result = reset_response.json()

    final_score = 0.0
    for _ in range(MAX_STEPS):
        if result.get("done"):
            final_score = float(result.get("info", {}).get("final_score", 0.0))
            break

        observation = result["observation"]
        action = call_model(client, observation)
        try:
            safe_action = TicketTriageAction(**action).model_dump(mode="json")
        except Exception:
            safe_action = TicketTriageAction(action_type=ActionType.NOOP).model_dump(mode="json")

        step_response = requests.post(f"{ENV_BASE_URL}/step", json=safe_action, timeout=30)
        if step_response.status_code >= 400:
            noop_action = TicketTriageAction(action_type=ActionType.NOOP).model_dump(mode="json")
            step_response = requests.post(
                f"{ENV_BASE_URL}/step", json=noop_action, timeout=30
            )
        step_response.raise_for_status()
        result = step_response.json()

    if not result.get("done"):
        batch_action = TicketTriageAction(action_type=ActionType.SUBMIT_BATCH).model_dump(mode="json")
        batch_response = requests.post(f"{ENV_BASE_URL}/step", json=batch_action, timeout=30)
        batch_response.raise_for_status()
        batch_result = batch_response.json()
        final_score = float(batch_result.get("info", {}).get("final_score", 0.0))

    return float(final_score)


def main() -> None:
    api_key = OPENAI_API_KEY or HF_TOKEN
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    scores: Dict[str, float] = {}
    for task in TASKS:
        scores[task] = run_task(client, task)

    overall = sum(scores.values()) / len(scores)
    print("Baseline Scores")
    print("==============")
    for task in TASKS:
        print(f"{task:>6}: {scores[task]:.3f}")
    print(f"overall: {overall:.3f}")
    print(
        f"metadata: model={MODEL_NAME}, seed={SEED}, temp={TEMPERATURE}, env={ENV_BASE_URL}"
    )


if __name__ == "__main__":
    main()
