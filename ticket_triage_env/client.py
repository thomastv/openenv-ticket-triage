from __future__ import annotations

from typing import Any, Dict

import requests

from .models import TicketTriageAction


class TicketTriageEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_id: str | None = None) -> Dict[str, Any]:
        payload = {"task_id": task_id} if task_id else {}
        response = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def step(self, action: TicketTriageAction) -> Dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}/step", json=action.model_dump(mode="json"), timeout=30
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        response = self._session.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._session.close()
