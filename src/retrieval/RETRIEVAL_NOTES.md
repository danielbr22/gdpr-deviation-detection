# Retrieval Design — Notes

## Current Implementation: Hybrid Pipeline

The retrieval step is a two-stage hybrid:

1. **Stage 1 — Embedding top-k** (`embed_and_match.py`): Legal-BERT encodes all
   GDPR constraints and policy passages. For each GDPR constraint, the top-k most
   similar policy passages (k=5) are retrieved by cosine similarity. This reduces
   the LLM judge to 279 calls (one per GDPR constraint, all 5 candidates in one
   prompt) rather than ~16,000+ naïve pairwise checks (279 × policy length).

2. **Stage 2 — LLM-as-judge** (`llm_judge.py`): For each GDPR constraint, its
   top-k candidates are sent to Qwen3.5 9B (via Ollama) with a structured prompt
   (`judge_prompt.py`). The LLM decides which candidate(s) substantively address
   the GDPR obligation — or none if no match exists. This produces:
   - `matched_pairs.json` — confirmed (GDPR, policy) pairs
   - `unmapped_gdpr.json` — GDPR constraints with no matching policy passage
                            (→ `missing_coverage` candidates for classification)

Model: `nlpaueb/legal-bert-base-uncased` (Legal-BERT), following Sai et al.'s
best-performing retrieval configuration.

**Context-enriched embedding (±1 sentence window):** Policy constraints are embedded
using an `embed_text` field that prepends/appends the immediately adjacent sentence to
the target sentence. The bare `text` field is kept unchanged for the LLM judge and
classifier downstream. Rationale: a single isolated policy sentence is often semantically
ambiguous (e.g. "You cannot request deletion…" scores high for purpose-limitation GDPR
constraints without context). Using the full ±5 extraction window would dilute the
embedding and risk hitting Legal-BERT's 512-token limit; ±1 provides enough
disambiguation without those costs.

Run via: `bash run_pipeline.sh` (all 3 use cases) or individual script calls.

---

## Why Hybrid over Embedding-Only

Embedding similarity captures surface-level semantic overlap well but misses
legally equivalent but differently phrased constraints. Example:

  GDPR:   "the controller shall document all processing activities"
  Policy: "NovaTech maintains a register of data processing operations"

These score below a fixed threshold even though they address the same obligation.
The paper's best model (Legal-S-BERT) reached only 70.6% recall with embedding-only —
roughly 30% of real matches were missed.

The LLM judge handles paraphrase and legal equivalence that embeddings cannot.

---

## Comparison: v1 (embedding-only) vs. current (hybrid)

| Aspect | v1 | Current |
|--------|----|---------|
| Policy extraction | Signal words (28 sentences) | Hybrid LLM (100–200+ passages) |
| Retrieval model | all-MiniLM-L6-v2 | Legal-BERT |
| Matching | γ threshold (cosine ≥ 0.5) | top-k + LLM judge |
| Coverage (Hetzner) | 65.9% | Higher recall, lower noise |

The v1 run data has been removed; all current results are in
`data/retrieval/{hetzner,zalando,traderepublic}_hybrid/`.
