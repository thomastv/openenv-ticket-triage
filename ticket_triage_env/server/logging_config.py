from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


def _load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE from local .env without overriding existing process env."""
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


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ["event", "path", "action_type", "ticket_id", "task_id", "status_code"]:
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging() -> logging.Logger:
    _load_env_file()

    logger = logging.getLogger("ticket_triage_env")

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_to_file = os.getenv("LOG_TO_FILE", "0").strip().lower() in {"1", "true", "yes", "on"}
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/server.jsonl")

    logger.setLevel(level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

        if log_to_file:
            os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)

    logger.propagate = False

    # Mirror handlers to uvicorn loggers so framework/access logs are also persisted.
    for uvicorn_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uv_logger = logging.getLogger(uvicorn_logger_name)
        uv_logger.setLevel(level)
        uv_logger.handlers = list(logger.handlers)
        uv_logger.propagate = False

    return logger
