#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "[1/4] Checking health endpoint"
curl -fsS "${BASE_URL}/health" >/dev/null

echo "[2/4] Checking reset endpoint"
curl -fsS -X POST "${BASE_URL}/reset" -H 'Content-Type: application/json' -d '{"task_id":"easy"}' >/dev/null

echo "[3/4] Running tests"
pytest -q

echo "[4/4] Running OpenEnv validation"
if command -v openenv >/dev/null 2>&1; then
  openenv validate
else
  echo "openenv CLI not found; skipping openenv validate"
fi

echo "Validation completed"
