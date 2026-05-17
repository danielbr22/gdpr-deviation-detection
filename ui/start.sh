#!/usr/bin/env bash
# start.sh — launch the pipeline dashboard UI
#
# Usage:
#   bash ui/start.sh        — production: serves built frontend via FastAPI
#   bash ui/start.sh --dev  — dev mode: FastAPI + Vite HMR (two processes)
#
# Open: http://localhost:8000

set -euo pipefail
UI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--dev" ]; then
  echo "Starting in dev mode (Vite on :5173, FastAPI on :8000)..."
  trap 'kill 0' EXIT
  cd "$UI/frontend" && npm run dev &
  cd "$UI" && python3 -m uvicorn server:app --reload --port 8000
else
  echo "Building frontend..."
  cd "$UI/frontend" && npm run build
  echo "Starting server at http://localhost:8000"
  cd "$UI" && python3 -m uvicorn server:app --port 8000
fi
