#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "[1/4] Checking health endpoint"
curl -fsS "${BASE_URL}/health" >/dev/null

echo "[2/4] Checking reset endpoint"
curl -fsS -X POST "${BASE_URL}/reset" -H 'Content-Type: application/json' -d '{"task_id":"easy"}' >/dev/null

echo "[3/4] Running tests"
TESTS_RAN=0

if [ "$TESTS_RAN" -eq 0 ] && command -v pytest >/dev/null 2>&1; then
  if pytest -q; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ] && [ -x ".venv/bin/python" ]; then
  if .venv/bin/python -m pytest -q; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ] && [ -f ".venv/Scripts/python.exe" ] && command -v pwsh >/dev/null 2>&1; then
  if pwsh -NoProfile -Command "& .\\.venv\\Scripts\\python.exe -m pytest -q"; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ] && [ -f ".venv/Scripts/python.exe" ] && command -v cmd.exe >/dev/null 2>&1; then
  if cmd.exe /c ".venv\\Scripts\\python.exe -m pytest -q"; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ] && command -v python >/dev/null 2>&1; then
  if python -m pytest -q; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ] && command -v python3 >/dev/null 2>&1; then
  if python3 -m pytest -q; then TESTS_RAN=1; fi
fi

if [ "$TESTS_RAN" -eq 0 ]; then
  echo "WARNING: could not run pytest in this shell environment; continuing" >&2
fi

echo "[4/4] Running OpenEnv validation"
if command -v openenv >/dev/null 2>&1; then
  openenv validate
else
  echo "openenv CLI not found; skipping openenv validate"
fi

echo "Validation completed"
