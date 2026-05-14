#!/usr/bin/env bash
# Resumes the pipeline from the LLM-judge step onward.
#
# Assumes extraction + embedding are already done (topk_candidates.json exists).
# Hetzner judge output is already present; this script runs:
#   1. LLM judge   — zalando, traderepublic
#   2. Classify    — all 3 use cases  (old classified files are cleared first)
#   3. Evaluate    — all 3 use cases  → data/evaluation/results.json
#
# Usage:
#   bash run_from_judge.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/from_judge_$(date '+%Y%m%d_%H%M%S').log"

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
    requests.get(f'{base}/api/tags', timeout=5).raise_for_status()
    print(f'  Ollama reachable at {base}')
except Exception as e:
    sys.exit(f'Ollama not reachable at {base}: {e}')
" || die "Start Ollama before running this script."

log "Pipeline (judge → classify → evaluate) starting — log: $LOG_FILE"

# ── Step 1: LLM judge — Zalando ──────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "JUDGE [1/2]: zalando"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m src.retrieval.llm_judge \
  --topk  "$ROOT/data/retrieval/zalando_hybrid/topk_candidates.json" \
  --output-dir "$ROOT/data/retrieval/zalando_hybrid"
n_m=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/zalando_hybrid/matched_pairs.json'))))")
n_u=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/zalando_hybrid/unmapped_gdpr.json'))))")
log "Done — matched=$n_m, unmapped=$n_u"

# ── Step 1: LLM judge — Trade Republic ───────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "JUDGE [2/2]: traderepublic"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m src.retrieval.llm_judge \
  --topk  "$ROOT/data/retrieval/traderepublic_hybrid/topk_candidates.json" \
  --output-dir "$ROOT/data/retrieval/traderepublic_hybrid"
n_m=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/traderepublic_hybrid/matched_pairs.json'))))")
n_u=$(python3 -c "import json; print(len(json.load(open('$ROOT/data/retrieval/traderepublic_hybrid/unmapped_gdpr.json'))))")
log "Done — matched=$n_m, unmapped=$n_u"

# ── Step 2: Classify — all 3 use cases ───────────────────────────────────────
# Remove stale classified files so the caching in classify.py doesn't reuse
# pairs that no longer exist after the updated judge run.
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "CLASSIFY — clearing stale outputs and reclassifying all 3 use cases"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for uc in hetzner zalando traderepublic; do
  out="$ROOT/data/classification/${uc}_hybrid_classified.json"
  [ -f "$out" ] && rm "$out" && log "Removed stale: $out"
done

for uc in hetzner zalando traderepublic; do
  log "Classifying: $uc"
  python3 -m src.classification.classify \
    --matched-pairs "$ROOT/data/retrieval/${uc}_hybrid/matched_pairs.json" \
    --unmapped      "$ROOT/data/retrieval/${uc}_hybrid/unmapped_gdpr.json" \
    --output        "$ROOT/data/classification/${uc}_hybrid_classified.json" \
    --use-case      "$uc"
  log "Done: $uc"
done

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EVALUATE — all 3 use cases"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m src.evaluation.evaluate \
  --use-case all \
  --output "$ROOT/data/evaluation/results.json"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ALL DONE — results at data/evaluation/results.json"
log "Log saved to $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
