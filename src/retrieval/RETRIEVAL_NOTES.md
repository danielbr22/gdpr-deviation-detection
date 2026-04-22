# Retrieval Design — Notes and Future Directions

## Current Implementation

Embedding-based cosine similarity matching using `all-MiniLM-L6-v2`
(sentence-transformers). Each GDPR constraint is mapped to its single
best-matching policy constraint. Pairs below threshold γ are classified
as `missing_coverage` candidates without further LLM involvement.

Threshold: γ = 0.7 (same as reported in Sai et al. Case Study 1).

Run: `python3 src/retrieval/embed_and_match.py [--gamma 0.7] [--model ...]`

## Why Embedding-Only for Now

- Fast and cheap: 279 × 47 similarity matrix computed in seconds locally
- Directly comparable to the paper's baseline (same model, same γ)
- Reproducible and threshold-tunable — easy to sweep γ as an experiment

## Known Limitation: Paraphrase Gap

Embedding similarity captures lexical and surface-level semantic overlap
well, but can miss constraints that are legally equivalent but phrased very
differently. Example:

  GDPR:   "the controller shall document all processing activities"
  Policy: "NovaTech maintains a register of data processing operations"

These may score below γ even though they address the same obligation.
The paper's best model (Legal-S-BERT) reached only 70.6% recall, meaning
roughly 30% of real matches were missed — likely due to exactly this issue.

## Planned Improvement: Hybrid LLM Re-ranking (Future Work)

A two-stage hybrid approach is planned but not yet implemented:

1. **Stage 1 (embedding):** Use cosine similarity to retrieve top-K candidates
   (K = 3–5) per GDPR constraint. Fast, eliminates obvious non-matches.

2. **Stage 2 (LLM):** For each GDPR constraint, send its top-K candidates to
   an LLM with a prompt like:
   > "Which of these policy clauses, if any, addresses the following GDPR
   >  obligation? If none apply, say so."

   The LLM makes the final mapping decision based on legal semantics, not
   surface similarity.

**Why this is better:** The LLM handles paraphrase and legal equivalence;
the embedding step keeps the LLM call count to ~900 (279 × 3) rather than
13,113 (279 × 47 full pairwise).

**Experiment design:** Compare embedding-only (γ = 0.7) vs. hybrid on the
gold standard — precision, recall, F1 per deviation type. This is a direct,
quantifiable contribution over the paper's baseline.

This should be implemented after the full pipeline is validated end-to-end.
