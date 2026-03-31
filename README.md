# Customer Support Ticket Triage OpenEnv

A production-oriented OpenEnv-style environment for evaluating tool-using agents on customer support ticket triage.

## Why This Environment

Real support teams routinely classify, prioritize, route, and respond to incoming tickets under SLA pressure. This environment models that workflow with deterministic scoring so agents can be trained and benchmarked reproducibly.

## API

- `POST /reset` -> initialize an episode (`task_id`: easy, medium, hard)
- `POST /step` -> submit one action and receive observation, reward, done, info
- `GET /state` -> retrieve episode state

## Action Space

`action_type` values:
- `inspect_ticket`
- `set_fields` (supports `category`, `priority`, `queue`, `next_action`)
- `draft_response` (requires `response_text`)
- `submit_ticket`
- `submit_batch`
- `noop`

## Observation Space

Each step returns:
- queue snapshot (submitted, remaining, high/urgent counters)
- current ticket view and current decision state
- validation feedback
- progress metrics (step count, cumulative reward, provisional score)
- last action result

## Tasks and Difficulty

- Easy: explicit billing and access issues
- Medium: mixed outages, billing and security with SLA pressure
- Hard: multi-ticket batch with outages, abuse/security, feature and bug mix

## Deterministic Grading

Per-ticket weighted score:
- category: 0.25
- priority: 0.20
- queue: 0.20
- next action: 0.20
- response compliance: 0.15

Batch score is the average per-ticket score minus unresolved urgent-ticket penalties, clamped to [0.0, 1.0].

## Reward Design

- Positive shaping for first-time correct triage fields and compliant responses
- Penalties for invalid actions, contradictory loops, risky misroutes, and inefficient step usage
- Terminal bonus for high final score

## Setup

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn ticket_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t ticket-triage-env:latest .
docker run --rm -p 8000:8000 ticket-triage-env:latest
```

## Baseline Inference

Set env vars (recommended via `.env.example`):
- `LLM_PROVIDER` (recommended: `gemini` or `openai`)
- `LLM_API_KEY` (required)
- `LLM_MODEL` (recommended)
- `LLM_API_BASE_URL` (optional)
- Backward compatibility: `OPENAI_API_KEY`, `MODEL_NAME`, `API_BASE_URL`
- `HF_TOKEN` (optional fallback token)
- `ENV_BASE_URL` (optional)
- `MAX_STEPS` (optional)
- `TEMPERATURE` (optional)
- `SEED` (optional)
- `LOG_LEVEL` (optional server logging level)

Gemini example (OpenAI-compatible endpoint):

```bash
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-2.5-flash-lite
# Optional, auto-selected when provider=gemini:
# LLM_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

Quick setup:

```bash
cp .env.example .env
# Fill .env values, then export/load in your shell
```

Run baseline:

```bash
python inference.py
```

## Validation

```bash
openenv validate
pytest -q
```

## Baseline Scores

Fill after first reproducible run:

- easy: TBD
- medium: TBD
- hard: TBD
- overall: TBD
