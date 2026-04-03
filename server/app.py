from __future__ import annotations

import uvicorn

from ticket_triage_env.server.app import app, create_app


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()