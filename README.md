---
title: Customer Support Ticket Triage OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

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

## Hugging Face Space

Live Space:

https://huggingface.co/spaces/thomastv/openenv-customer-ticket-triage

Space endpoint checks:

```bash
curl -s -o /dev/null -w "%{http_code}" https://thomastv-openenv-customer-ticket-triage.hf.space/health
curl -X POST https://thomastv-openenv-customer-ticket-triage.hf.space/reset -H "Content-Type: application/json" -d '{}'
```

## Baseline Inference

Set env vars (recommended via `.env.example`):
- `API_BASE_URL` (recommended; provider default is used if unset)
- `MODEL_NAME` (required for deterministic model selection)
- `HF_TOKEN` (required; aliases: `LLM_API_KEY`, `OPENAI_API_KEY`)
- `LOCAL_IMAGE_NAME` (required only for implementations using `from_docker_image()`)
- `LLM_PROVIDER` (optional: `openai`, `gemini`, `ollama`)
- `LLM_API_BASE_URL` (optional override; highest precedence)
- `TASK_NAME` (optional; default `easy`)
- `BENCHMARK` (optional; default `ticket_triage`)
- `LOG_LEVEL` (optional server logging level)
- `LOG_TO_FILE` and `LOG_FILE_PATH` (optional server file logging)
- `INFERENCE_VERBOSE` (optional inference verbosity)
- `INFERENCE_LOG_TO_FILE` and `INFERENCE_LOG_FILE_PATH` (optional inference file logging)

Gemini example (OpenAI-compatible endpoint):

```bash
LLM_PROVIDER=gemini
HF_TOKEN=your_hf_token
MODEL_NAME=google/gemini-2.5-flash
# Optional; auto-selected when provider=gemini and API_BASE_URL is unset:
# API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

Ollama local example (`llama3.2`):

```bash
LLM_PROVIDER=ollama
HF_TOKEN=ollama
MODEL_NAME=llama3.2
API_BASE_URL=http://localhost:11434/v1
```

Free-tier quota controls:

```bash
MAX_LLM_CALLS_PER_TASK=8
LLM_RETRY_MAX=2
LLM_RETRY_SECONDS=35
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

File logging examples:

```bash
# Server JSON logs to file
LOG_TO_FILE=1
LOG_FILE_PATH=logs/server.jsonl

# Inference logs to file (respects INFERENCE_VERBOSE level)
INFERENCE_LOG_TO_FILE=1
INFERENCE_LOG_FILE_PATH=logs/inference.log
```

## Validation

```bash
openenv validate
pytest -q
```

## Baseline Scores

- easy: 0.850
- medium: 0.855
- hard: 0.623
- overall: 0.776
