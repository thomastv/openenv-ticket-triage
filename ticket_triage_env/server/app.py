from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ticket_triage_env.models import TicketTriageAction
from ticket_triage_env.server.environment import TicketTriageEnvironment
from ticket_triage_env.server.logging_config import configure_logging


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


def create_app() -> FastAPI:
    env = TicketTriageEnvironment()
    app = FastAPI(title="Customer Support Ticket Triage OpenEnv", version="0.1.0")
    logger = configure_logging()

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning(
            "request validation error",
            extra={"event": "value_error", "path": str(request.url.path), "status_code": 400},
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "message": str(exc),
                "path": str(request.url.path),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled server exception",
            extra={"event": "unexpected_error", "path": str(request.url.path), "status_code": 500},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Unexpected server error",
                "path": str(request.url.path),
            },
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)

    @app.get("/web")
    def web() -> RedirectResponse:
        # Hugging Face Spaces may probe /web when frontmatter has base_path: /web.
        return RedirectResponse(url="/docs", status_code=307)

    @app.post("/reset")
    def reset(payload: ResetRequest | None = None) -> dict:
        task_id = payload.task_id if payload else None
        logger.info("reset called", extra={"event": "reset", "task_id": task_id})
        return env.reset(task_id=task_id)

    @app.post("/step")
    def step(action: TicketTriageAction) -> dict:
        logger.info(
            "step called",
            extra={
                "event": "step",
                "action_type": action.action_type.value,
                "ticket_id": action.ticket_id,
            },
        )
        return env.step(action)

    @app.get("/state")
    def state() -> dict:
        logger.info("state called", extra={"event": "state"})
        return env.state()

    return app


app = create_app()
