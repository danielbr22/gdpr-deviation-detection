#!/usr/bin/env bash
# Full hybrid pipeline: extraction → embed+top-k → LLM judge → classify
# Usage:
#   bash run_pipeline.sh             # full overnight run, all 3 use cases
#   bash run_pipeline.sh --test      # dry-run with tiny slice to verify everything works

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"

TOP_K=5
EMBED_MODEL="nlpaueb/legal-bert-base-uncased"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() {
  local msg="[$(date '+%H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

die() { log "ERROR: $*"; exit 1; }

check_ollama() {
  python3 -c "
import requests, sys, os
base = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
try:
    requests.get(f'{base}/api/tags', timeout=5).raise_for_status()
except Exception as e:
    sys.exit(f'Ollama not reachable at {base}: {e}')
" || die "Ollama must be running before starting the pipeline."
}

run_use_case() {
  local name="$1"
  local policy_file="$2"
  local constraints_out="$3"
  local retrieval_dir="$4"
  local classification_out="$5"
  local judge_limit="${6:-}"      # optional --limit N for judge (testing)
  local classify_limit="${7:-}"   # optional --limit N for classifier (testing)

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "USE CASE: $name"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Step 1: Hybrid LLM extraction
  log "[1/4] Hybrid extraction  →  $constraints_out"
  python3 "$ROOT/src/preprocessing/hybrid_extract.py" \
    --policy "$policy_file" \
    --output "$constraints_out"
  n_constraints=$(python3 -c "import json; d=json.load(open('$constraints_out')); print(len(d))")
  log "[1/4] Done — $n_constraints constraints extracted"

  # Step 2: Legal-S-BERT embed + top-k
  mkdir -p "$retrieval_dir"
  log "[2/4] Embed + top-$TOP_K  →  $retrieval_dir"
  python3 "$ROOT/src/retrieval/embed_and_match.py" \
    --policy-constraints "$constraints_out" \
    --output-dir "$retrieval_dir" \
    --model "$EMBED_MODEL" \
    --top-k "$TOP_K"
  log "[2/4] Done"

  # Step 3: LLM-as-judge
  log "[3/4] LLM judge  →  $retrieval_dir"
  local judge_limit_arg=""
  [ -n "$judge_limit" ] && judge_limit_arg="--limit $judge_limit"
  python3 "$ROOT/src/retrieval/llm_judge.py" \
    --topk "$retrieval_dir/topk_candidates.json" \
    --output-dir "$retrieval_dir" \
    $judge_limit_arg
  n_matched=$(python3 -c "import json; print(len(json.load(open('$retrieval_dir/matched_pairs.json'))))")
  n_unmapped=$(python3 -c "import json; print(len(json.load(open('$retrieval_dir/unmapped_gdpr.json'))))")
  log "[3/4] Done — $n_matched matched, $n_unmapped unmapped"

  # Step 4: Classify
  mkdir -p "$(dirname "$classification_out")"
  log "[4/4] Classify  →  $classification_out"
  local limit_arg=""
  [ -n "$classify_limit" ] && limit_arg="--limit $classify_limit"
  python3 "$ROOT/src/classification/classify.py" \
    --matched-pairs "$retrieval_dir/matched_pairs.json" \
    --unmapped      "$retrieval_dir/unmapped_gdpr.json" \
    --output        "$classification_out" \
    --use-case      "$name" \
    $limit_arg
  log "[4/4] Done"
}

# ── Prevent sleep (macOS caffeinate) ─────────────────────────────────────────
if command -v caffeinate &>/dev/null && [ "${CAFFEINATED:-0}" != "1" ]; then
  log "Re-launching under caffeinate to prevent sleep…"
  CAFFEINATED=1 exec caffeinate -d -i -s bash "$0" "$@"
fi

# ── Test mode vs full run ─────────────────────────────────────────────────────
TEST_MODE=0
for arg in "$@"; do [ "$arg" = "--test" ] && TEST_MODE=1; done

check_ollama
log "Pipeline starting — log: $LOG_FILE"
log "Test mode: $TEST_MODE"

if [ "$TEST_MODE" = "1" ]; then
  log "──── TEST MODE: tiny slice of Hetzner only ────"
  TMP="$ROOT/logs/test_run"
  mkdir -p "$TMP"

  # Write a small policy slice (~1500 chars, first 2 sections)
  python3 -c "
from pathlib import Path
text = Path('$ROOT/data/policy/hetzner_policy_modified.txt').read_text()
Path('$TMP/hetzner_test_policy.txt').write_text(text[:1500])
print('Test slice: 1500 chars')
"

  # judge_limit=5, classify_limit=3
  run_use_case \
    "hetzner_test" \
    "$TMP/hetzner_test_policy.txt" \
    "$TMP/hetzner_test_constraints.json" \
    "$TMP/hetzner_test_retrieval" \
    "$TMP/hetzner_test_classified.json" \
    "5" \
    "3"

  log "──── TEST MODE complete — verifying outputs ────"
  python3 -c "
import json, sys
from pathlib import Path
tmp = '$TMP'

constraints = json.load(open(f'{tmp}/hetzner_test_constraints.json'))
topk        = json.load(open(f'{tmp}/hetzner_test_retrieval/topk_candidates.json'))
matched     = json.load(open(f'{tmp}/hetzner_test_retrieval/matched_pairs.json'))
unmapped    = json.load(open(f'{tmp}/hetzner_test_retrieval/unmapped_gdpr.json'))
classified  = json.load(open(f'{tmp}/hetzner_test_classified.json'))

assert all('id' in c and 'text' in c for c in constraints), 'constraints schema broken'
assert all('candidates' in e and len(e['candidates']) > 0 for e in topk), 'topk schema broken'
for p in matched:
    assert {'gdpr_id','policy_id','gdpr_text','policy_text'} <= set(p.keys()), f'matched missing fields: {p}'
for u in unmapped:
    assert u['deviation_type'] == 'missing_coverage', 'unmapped wrong type'

pairs = classified.get('pairs', [])
for p in pairs:
    assert 'deviation_type' in p and 'reasoning' in p, f'classified missing fields: {p}'

print(f'  constraints : {len(constraints)}')
print(f'  topk entries: {len(topk)}')
print(f'  matched     : {len(matched)}')
print(f'  unmapped    : {len(unmapped)}')
print(f'  classified  : {len(pairs)} pairs + {len(classified.get(\"unmapped\",[]))} unmapped')
print('ALL SCHEMA CHECKS PASSED')
"
  log "TEST PASSED — safe to run full pipeline"

else
  log "──── FULL PIPELINE — all 3 use cases ────"

  run_use_case \
    "hetzner" \
    "$ROOT/data/policy/hetzner_policy_modified.txt" \
    "$ROOT/data/constraints/hetzner_hybrid_constraints.json" \
    "$ROOT/data/retrieval/hetzner_hybrid" \
    "$ROOT/data/classification/hetzner_hybrid_classified.json"

  run_use_case \
    "zalando" \
    "$ROOT/data/policy/zalando_policy_modified.txt" \
    "$ROOT/data/constraints/zalando_hybrid_constraints.json" \
    "$ROOT/data/retrieval/zalando_hybrid" \
    "$ROOT/data/classification/zalando_hybrid_classified.json"

  run_use_case \
    "traderepublic" \
    "$ROOT/data/policy/traderepublic_policy_modified.txt" \
    "$ROOT/data/constraints/traderepublic_hybrid_constraints.json" \
    "$ROOT/data/retrieval/traderepublic_hybrid" \
    "$ROOT/data/classification/traderepublic_hybrid_classified.json"

  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "ALL USE CASES COMPLETE"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

log "Pipeline finished — $(date)"
