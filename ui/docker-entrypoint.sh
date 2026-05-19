#!/usr/bin/env bash
# Pull the LLM model if not already cached, then start the dashboard UI.
set -euo pipefail

MODEL="qwen3.5:9b"

echo "[ui] Pre-warming model cache (non-fatal if network is unavailable)..."
python3 - <<'PYEOF'
import json, os, sys
import requests

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
model = "qwen3.5:9b"

try:
    tags = requests.get(f"{base}/api/tags", timeout=10).json()
    if any(m["name"] == model or m["name"].startswith(model.split(":")[0])
           for m in tags.get("models", [])):
        print(f"[ui] Model {model} already cached — skipping pull.")
        sys.exit(0)

    print(f"[ui] Pulling {model} (~5.5 GB — this may take several minutes on first run)...")
    r = requests.post(
        f"{base}/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=1800,
    )
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            print(f"[ui] Pull warning: {data['error']} — model will be pulled on first run.", file=sys.stderr)
            sys.exit(0)
        status = data.get("status", "")
        if "total" in data and "completed" in data and data["total"] > 0:
            pct = int(data["completed"] / data["total"] * 100)
            print(f"\r[ui]   {status}: {pct}%", end="", flush=True)
        elif status:
            print(f"[ui]   {status}", flush=True)
    print("\n[ui] Model ready.")
except Exception as e:
    print(f"[ui] Could not pre-warm model ({e}) — model will be pulled on first run.", file=sys.stderr)
PYEOF

echo "[ui] Starting dashboard at http://0.0.0.0:8000"
cd /app/ui
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
