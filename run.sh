#!/usr/bin/env bash
# run.sh — full GDPR deviation detection pipeline (single entry point)
#
# Phases:
#   0  Scope detection   extract + embed + judge on ORIGINAL policies (once per use case)
#   1  Extraction        hybrid LLM extraction on MODIFIED policies
#   2  Retrieval         Legal-BERT embed + top-k + LLM judge on modified policies
#   3  Remapping         fill policy_constraint_id in gold standard manifests
#   4  Classification    LLM deviation classifier on matched pairs
#   5  Evaluation        precision/recall/F1 vs gold standard → data/evaluation/results.json
#
# Phases 0–2 are skipped when their output files already exist (idempotent by default).
# Use --force to re-run all phases from scratch.
#
# Usage:
#   bash run.sh           full pipeline (resume-safe, skip already-done phases)
#   bash run.sh --test    smoke test: tiny Hetzner slice to verify the full chain works
#   bash run.sh --force   ignore skip guards, re-run all phases

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"

# Load .env if present (does not override already-exported vars).
if [ -f "$ROOT/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +o allexport
fi

TOP_K=5
EMBED_MODEL="nlpaueb/legal-bert-base-uncased"

# ── Parse flags (must be first — used by caffeinate re-exec and log setup) ────
TEST_MODE=0
FORCE=0
for arg in "$@"; do
  [ "$arg" = "--test"  ] && TEST_MODE=1
  [ "$arg" = "--force" ] && FORCE=1
done
[ "$TEST_MODE" = "1" ] && FORCE=1   # always re-run all phases in test mode

# ── Keep Mac awake (must come before tee redirect — re-exec restarts script) ──
if command -v caffeinate &>/dev/null && [ "${CAFFEINATED:-0}" != "1" ]; then
  echo "[$(date '+%H:%M:%S')] Re-launching under caffeinate to prevent sleep…"
  CAFFEINATED=1 exec caffeinate -d -i -s bash "$0" "$@"
fi

# ── Logging: tee ALL output (including Python) to terminal + log file ─────────
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1

PIPELINE_START=$(date +%s)

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

elapsed() {
  local s=$(( $(date +%s) - $1 ))
  printf "%dm%02ds" $(( s / 60 )) $(( s % 60 ))
}

# Safe JSON list-length counter — passes path as argv so spaces in path are fine.
jcount() {
  python3 - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(len(d) if isinstance(d, list) else "?")
except Exception:
    print("-")
EOF
}

jcount_key() {
  python3 - "$1" "$2" <<'EOF'
import json, sys
try:
    print(len(json.load(open(sys.argv[1])).get(sys.argv[2], [])))
except Exception:
    print("-")
EOF
}

# ── Pre-flight: verify all required input files exist ─────────────────────────
preflight() {
  log "Pre-flight checks…"
  local ok=1
  local required=(
    "$ROOT/data/constraints/gdpr_constraints.json"
    "$ROOT/data/policy/hetzner_privacy_policy.txt"
    "$ROOT/data/policy/hetzner_policy_modified.txt"
    "$ROOT/data/policy/zalando_privacy_policy.txt"
    "$ROOT/data/policy/zalando_policy_modified.txt"
    "$ROOT/data/policy/traderepublic_privacy_policy.txt"
    "$ROOT/data/policy/traderepublic_policy_modified.txt"
  )
  for f in "${required[@]}"; do
    if [ ! -f "$f" ]; then
      log "  MISSING: $f"
      ok=0
    fi
  done
  [ "$ok" = "0" ] && die "Pre-flight failed — missing required input files"
  log "  All required input files present ✓"
}

# ── Provider check ─────────────────────────────────────────────────────────────
check_provider() {
  local provider="${LLM_PROVIDER:-ollama}"
  log "Checking LLM provider (${provider})…"
  python3 -c "from src.utils import llm_client; llm_client.check_provider()"
}

# ── Skip guard ─────────────────────────────────────────────────────────────────
# Returns 0 (true → caller should skip) when output exists and FORCE=0.
already_done() {
  local path="$1"
  local label="$2"
  if [ "$FORCE" = "0" ] && [ -f "$path" ]; then
    local n
    n=$(jcount "$path")
    log "  SKIP $label — already done ($n entries); use --force to re-run"
    return 0
  fi
  return 1
}

# ── Phase functions ────────────────────────────────────────────────────────────

# Phase 0 — run extract+embed+judge on the ORIGINAL policy.
# Output: data/retrieval/{uc}_original/matched_pairs.json
# Used by evaluate.py to filter structural constraint_coverage FPs.
scope_detection() {
  local name="$1" policy="$2" constraints_out="$3" retrieval_dir="$4"

  already_done "$retrieval_dir/matched_pairs.json" "scope/$name" && return

  log "  [scope/$name] Extracting policy obligations from original policy…"
  python3 "$ROOT/src/preprocessing/hybrid_extract.py" \
    --policy "$policy" \
    --output "$constraints_out"
  log "  [scope/$name] Extracted $(jcount "$constraints_out") constraints"

  log "  [scope/$name] Embedding + top-$TOP_K retrieval…"
  mkdir -p "$retrieval_dir"
  python3 "$ROOT/src/retrieval/embed_and_match.py" \
    --policy-constraints "$constraints_out" \
    --output-dir "$retrieval_dir" \
    --model "$EMBED_MODEL" \
    --top-k "$TOP_K"

  log "  [scope/$name] LLM judge filtering top-$TOP_K candidates…"
  python3 "$ROOT/src/retrieval/llm_judge.py" \
    --topk "$retrieval_dir/topk_candidates.json" \
    --output-dir "$retrieval_dir"

  log "  [scope/$name] Done — $(jcount "$retrieval_dir/matched_pairs.json") in-scope GDPR articles"
}

# Phase 1 — hybrid LLM extraction on the MODIFIED policy.
extraction() {
  local name="$1" policy="$2" constraints_out="$3"

  already_done "$constraints_out" "extraction/$name" && return

  log "  [extraction/$name] Extracting policy obligations…"
  python3 "$ROOT/src/preprocessing/hybrid_extract.py" \
    --policy "$policy" \
    --output "$constraints_out"
  log "  [extraction/$name] Done — $(jcount "$constraints_out") constraints extracted"
}

# Phase 2 — embed + top-k + LLM judge on the MODIFIED policy.
retrieval() {
  local name="$1" constraints_out="$2" retrieval_dir="$3"
  local judge_limit="${4:-}"

  already_done "$retrieval_dir/matched_pairs.json" "retrieval/$name" && return

  log "  [retrieval/$name] Embedding + top-$TOP_K retrieval…"
  mkdir -p "$retrieval_dir"
  python3 "$ROOT/src/retrieval/embed_and_match.py" \
    --policy-constraints "$constraints_out" \
    --output-dir "$retrieval_dir" \
    --model "$EMBED_MODEL" \
    --top-k "$TOP_K"

  log "  [retrieval/$name] LLM judge filtering top-$TOP_K candidates…"
  local limit_arg=""
  [ -n "$judge_limit" ] && limit_arg="--limit $judge_limit"
  # shellcheck disable=SC2086
  python3 "$ROOT/src/retrieval/llm_judge.py" \
    --topk "$retrieval_dir/topk_candidates.json" \
    --output-dir "$retrieval_dir" \
    $limit_arg

  log "  [retrieval/$name] Done — matched=$(jcount "$retrieval_dir/matched_pairs.json")  unmapped=$(jcount "$retrieval_dir/unmapped_gdpr.json")"
}

# ── Final summary ──────────────────────────────────────────────────────────────
print_summary() {
  log ""
  log "━━━ PIPELINE SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "  $(printf '%-16s  %8s  %8s  %8s  %8s' 'use case' 'in-scope' 'matched' 'unmapped' 'classified')"
  log "  $(printf '%-16s  %8s  %8s  %8s  %8s' '────────────────' '────────' '────────' '────────' '──────────')"
  for uc in hetzner zalando traderepublic; do
    local in_scope matched unmapped classified
    in_scope=$(jcount  "$ROOT/data/retrieval/${uc}_original/matched_pairs.json"   2>/dev/null || echo "-")
    matched=$(jcount   "$ROOT/data/retrieval/${uc}_hybrid/matched_pairs.json"     2>/dev/null || echo "-")
    unmapped=$(jcount  "$ROOT/data/retrieval/${uc}_hybrid/unmapped_gdpr.json"     2>/dev/null || echo "-")
    classified=$(jcount_key "$ROOT/data/classification/${uc}_hybrid_classified.json" "pairs" 2>/dev/null || echo "-")
    log "  $(printf '%-16s  %8s  %8s  %8s  %8s' "$uc" "$in_scope" "$matched" "$unmapped" "$classified")"
  done
  log ""
  log "  Total elapsed: $(elapsed $PIPELINE_START)"
  log "  Results file : data/evaluation/results.json"
  log "  Log file     : $LOG_FILE"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ══════════════════════════════════════════════════════════════════════════════
# TEST MODE
# ══════════════════════════════════════════════════════════════════════════════
if [ "$TEST_MODE" = "1" ]; then
  check_provider
  log "TEST MODE — tiny Hetzner slice — log: $LOG_FILE"

  TMP="$ROOT/logs/test_run"
  rm -rf "$TMP" && mkdir -p "$TMP"

  python3 - "$ROOT/data/policy/hetzner_policy_modified.txt" "$TMP/hetzner_test_policy.txt" <<'EOF'
import sys
text = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(text[:1500])
print(f"Test slice: 1500 chars written to {sys.argv[2]}")
EOF

  log "── Phase 1: Extraction ──────────────────────────────────────────────────"
  extraction "hetzner_test" "$TMP/hetzner_test_policy.txt" "$TMP/hetzner_test_constraints.json"

  log "── Phase 2: Retrieval ───────────────────────────────────────────────────"
  retrieval "hetzner_test" "$TMP/hetzner_test_constraints.json" "$TMP/hetzner_test_retrieval" "5"

  log "── Phase 4: Classification (limit 3) ───────────────────────────────────"
  python3 "$ROOT/src/classification/classify.py" \
    --matched-pairs "$TMP/hetzner_test_retrieval/matched_pairs.json" \
    --unmapped      "$TMP/hetzner_test_retrieval/unmapped_gdpr.json" \
    --output        "$TMP/hetzner_test_classified.json" \
    --use-case      "hetzner" \
    --limit 3

  log "── Schema verification ──────────────────────────────────────────────────"
  python3 - "$TMP" <<'EOF'
import json, sys
tmp = sys.argv[1]
constraints = json.load(open(f"{tmp}/hetzner_test_constraints.json"))
topk        = json.load(open(f"{tmp}/hetzner_test_retrieval/topk_candidates.json"))
matched     = json.load(open(f"{tmp}/hetzner_test_retrieval/matched_pairs.json"))
unmapped    = json.load(open(f"{tmp}/hetzner_test_retrieval/unmapped_gdpr.json"))
classified  = json.load(open(f"{tmp}/hetzner_test_classified.json"))

assert all("id" in c and "text" in c for c in constraints), "constraints schema broken"
assert all("candidates" in e and len(e["candidates"]) > 0 for e in topk), "topk schema broken"
for p in matched:
    assert {"gdpr_id","policy_id","gdpr_text","policy_text"} <= set(p.keys()), f"matched missing fields: {p}"
for u in unmapped:
    assert u["deviation_type"] == "missing_coverage", "unmapped wrong type"
for p in classified.get("pairs", []):
    assert "deviation_type" in p and "reasoning" in p, f"classified missing fields: {p}"

print(f"  constraints : {len(constraints)}")
print(f"  topk entries: {len(topk)}")
print(f"  matched     : {len(matched)}")
print(f"  unmapped    : {len(unmapped)}")
print(f"  classified  : {len(classified.get('pairs', []))} pairs + {len(classified.get('unmapped', []))} unmapped")
print("ALL SCHEMA CHECKS PASSED")
EOF

  log ""
  log "TEST PASSED in $(elapsed $PIPELINE_START) — safe to run full pipeline"
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
check_provider
preflight
log ""
log "Pipeline starting — log: $LOG_FILE"
[ "$FORCE" = "1" ] && log "FORCE mode: skip guards disabled"

# ── Phase 0 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 0: Original policy scope detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  Purpose: identify which GDPR articles are substantively covered by each"
log "           original policy, so structural missing_coverage FPs are excluded."

scope_detection \
  "hetzner" \
  "$ROOT/data/policy/hetzner_privacy_policy.txt" \
  "$ROOT/data/constraints/hetzner_original_constraints.json" \
  "$ROOT/data/retrieval/hetzner_original"

scope_detection \
  "zalando" \
  "$ROOT/data/policy/zalando_privacy_policy.txt" \
  "$ROOT/data/constraints/zalando_original_constraints.json" \
  "$ROOT/data/retrieval/zalando_original"

scope_detection \
  "traderepublic" \
  "$ROOT/data/policy/traderepublic_privacy_policy.txt" \
  "$ROOT/data/constraints/traderepublic_original_constraints.json" \
  "$ROOT/data/retrieval/traderepublic_original"

log "  Phase 0 done in $(elapsed $T)"

# ── Phase 1 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 1: Modified policy extraction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

extraction \
  "hetzner" \
  "$ROOT/data/policy/hetzner_policy_modified.txt" \
  "$ROOT/data/constraints/hetzner_hybrid_constraints.json"

extraction \
  "zalando" \
  "$ROOT/data/policy/zalando_policy_modified.txt" \
  "$ROOT/data/constraints/zalando_hybrid_constraints.json"

extraction \
  "traderepublic" \
  "$ROOT/data/policy/traderepublic_policy_modified.txt" \
  "$ROOT/data/constraints/traderepublic_hybrid_constraints.json"

log "  Phase 1 done in $(elapsed $T)"

# ── Phase 2 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 2: Modified policy retrieval ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

retrieval \
  "hetzner" \
  "$ROOT/data/constraints/hetzner_hybrid_constraints.json" \
  "$ROOT/data/retrieval/hetzner_hybrid"

retrieval \
  "zalando" \
  "$ROOT/data/constraints/zalando_hybrid_constraints.json" \
  "$ROOT/data/retrieval/zalando_hybrid"

retrieval \
  "traderepublic" \
  "$ROOT/data/constraints/traderepublic_hybrid_constraints.json" \
  "$ROOT/data/retrieval/traderepublic_hybrid"

log "  Phase 2 done in $(elapsed $T)"

# ── Phase 3 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 3: Manifest remapping ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  Filling null policy_constraint_id fields in gold standard manifests…"
python3 "$ROOT/src/utils/remap_manifests.py"
log "  Phase 3 done in $(elapsed $T)"

# ── Phase 4 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 4: Classification ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for uc in hetzner zalando traderepublic; do
  out="$ROOT/data/classification/${uc}_hybrid_classified.json"
  [ -f "$out" ] && rm "$out" && log "  Cleared stale: $(basename "$out")"
done

for uc in hetzner zalando traderepublic; do
  log "  [classify/$uc] Running LLM classifier…"
  python3 "$ROOT/src/classification/classify.py" \
    --matched-pairs "$ROOT/data/retrieval/${uc}_hybrid/matched_pairs.json" \
    --unmapped      "$ROOT/data/retrieval/${uc}_hybrid/unmapped_gdpr.json" \
    --output        "$ROOT/data/classification/${uc}_hybrid_classified.json" \
    --use-case      "$uc"
  log "  [classify/$uc] Done — $(jcount_key "$ROOT/data/classification/${uc}_hybrid_classified.json" "pairs") pairs classified"
done

log "  Phase 4 done in $(elapsed $T)"

# ── Phase 5 ───────────────────────────────────────────────────────────────────
T=$(date +%s)
log "━━━ PHASE 5: Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$ROOT/src/evaluation/evaluate.py" \
  --use-case all \
  --output "$ROOT/data/evaluation/results.json"
log "  Phase 5 done in $(elapsed $T)"

# ── Summary ───────────────────────────────────────────────────────────────────
print_summary
