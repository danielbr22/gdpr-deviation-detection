#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="qwen3.5:9b"

# ── Wait for Ollama ────────────────────────────────────────────────────────────
echo "[entrypoint] Waiting for Ollama at $OLLAMA_BASE..."
for i in $(seq 1 36); do
    if curl -sf "$OLLAMA_BASE/api/tags" >/dev/null 2>&1; then
        echo "[entrypoint] Ollama is ready."
        break
    fi
    if [ "$i" -eq 36 ]; then
        echo "[entrypoint] ERROR: Ollama never became ready after 3 minutes."
        exit 1
    fi
    echo "[entrypoint]   attempt $i/36 — retrying in 5s..."
    sleep 5
done

# ── Pull model if not already cached ──────────────────────────────────────────
python3 - <<'PYEOF'
import json, os, sys
import requests

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
model = "qwen3.5:9b"

tags = requests.get(f"{base}/api/tags", timeout=10).json()
if any(m["name"] == model or m["name"].startswith(model.split(":")[0])
       for m in tags.get("models", [])):
    print(f"[entrypoint] Model {model} already cached — skipping pull.")
    sys.exit(0)

print(f"[entrypoint] Pulling {model} (~5.5 GB, this may take several minutes on first run)...")
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
        print(f"[entrypoint] Pull error: {data['error']}", file=sys.stderr)
        sys.exit(1)
    status = data.get("status", "")
    if "total" in data and "completed" in data and data["total"] > 0:
        pct = int(data["completed"] / data["total"] * 100)
        print(f"\r[entrypoint]   {status}: {pct}%", end="", flush=True)
    elif status:
        print(f"[entrypoint]   {status}", flush=True)
print("\n[entrypoint] Model pull complete.")
PYEOF

# ── Run pipeline ───────────────────────────────────────────────────────────────
exec bash run.sh "$@"
