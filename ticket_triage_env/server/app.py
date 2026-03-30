from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ticket_triage_env.models import TicketTriageAction
from ticket_triage_env.server.environment import TicketTriageEnvironment


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


def create_app() -> FastAPI:
    env = TicketTriageEnvironment()
    app = FastAPI(title="Customer Support Ticket Triage OpenEnv", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/reset")
    def reset(payload: ResetRequest | None = None) -> dict:
        task_id = payload.task_id if payload else None
        return env.reset(task_id=task_id)

    @app.post("/step")
    def step(action: TicketTriageAction) -> dict:
        return env.step(action)

    @app.get("/state")
    def state() -> dict:
        return env.state()

    return app


app = create_app()
