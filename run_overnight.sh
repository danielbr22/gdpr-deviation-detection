#!/usr/bin/env bash
# Overnight pipeline: LLM judge → classify → evaluate (all 3 use cases)
#
# Assumes extraction and embedding are already done (topk_candidates.json exists
# for all 3 use cases). Runs everything from the judge step onward.
#
# Usage:
#   bash run_overnight.sh
#
# On completion: results are printed to the terminal and saved to
#   data/evaluation/results.json
#   logs/overnight_<timestamp>.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_$(date '+%Y%m%d_%H%M%S').log"

log() {
  local msg="[$(date '+%H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

die() { log "ERROR: $*"; exit 1; }

# ── Keep Mac awake ────────────────────────────────────────────────────────────
if command -v caffeinate &>/dev/null && [ "${CAFFEINATED:-0}" != "1" ]; then
  log "Re-launching under caffeinate to prevent sleep…"
  CAFFEINATED=1 exec caffeinate -d -i -s bash "$0" "$@"
fi

# ── Check Ollama ──────────────────────────────────────────────────────────────
log "Checking Ollama…"
python3 -c "
import requests, sys, os
base = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
try:
    r = requests.get(f'{base}/api/tags', timeout=5)
    r.raise_for_status()
    models = [m['name'] for m in r.json().get('models', [])]
    print(f'  Ollama reachable at {base}')
    print(f'  Available models: {models}')
    if not any('qwen3.5' in m for m in models):
        sys.exit('qwen3.5:9b not found — run: ollama pull qwen3.5:9b')
except Exception as e:
    sys.exit(f'Ollama not reachable at {base}: {e}')
" || die "Start Ollama and ensure qwen3.5:9b is available before running this script."

log "Starting overnight run — log: $LOG_FILE"
log ""

# ── Step 1: LLM judge — all 3 use cases ─────────────────────────────────────
for uc in hetzner zalando traderepublic; do
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "JUDGE: $uc"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  topk="$ROOT/data/retrieval/${uc}_hybrid/topk_candidates.json"
  [ -f "$topk" ] || die "Missing topk file for $uc: $topk — run embed_and_match.py first."

  python3 -m src.retrieval.llm_judge \
    --topk       "$topk" \
    --output-dir "$ROOT/data/retrieval/${uc}_hybrid"

  n_m=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/${uc}_hybrid/matched_pairs.json'))))")
  n_u=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/${uc}_hybrid/unmapped_gdpr.json'))))")
  log "Done — matched=$n_m  unmapped=$n_u"
  log ""
done

# ── Step 2: Classify — all 3 use cases ───────────────────────────────────────
# Clear stale classified files so classify.py's caching doesn't reuse pairs
# that may no longer exist after the updated judge run.
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "CLASSIFY — all 3 use cases"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for uc in hetzner zalando traderepublic; do
  out="$ROOT/data/classification/${uc}_hybrid_classified.json"
  [ -f "$out" ] && rm "$out" && log "Cleared stale: $(basename $out)"
done

for uc in hetzner zalando traderepublic; do
  log "Classifying: $uc"
  python3 -m src.classification.classify \
    --matched-pairs "$ROOT/data/retrieval/${uc}_hybrid/matched_pairs.json" \
    --unmapped      "$ROOT/data/retrieval/${uc}_hybrid/unmapped_gdpr.json" \
    --output        "$ROOT/data/classification/${uc}_hybrid_classified.json" \
    --use-case      "$uc"
  log "Done: $uc"
  log ""
done

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EVALUATE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m src.evaluation.evaluate \
  --use-case all \
  --output "$ROOT/data/evaluation/results.json"

log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ALL DONE"
log "Results : data/evaluation/results.json"
log "Log     : $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
