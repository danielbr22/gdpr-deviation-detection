# Detecting Deviations Between External and Internal Regulatory Requirements
**TUM Master Praktikum — NLP for Information Systems, SS26**

## Project Overview

This project extends Sai et al. (2023) by replacing their classical NLP pipeline with LLM-based methods for detecting deviations between external regulatory documents and internal company privacy policies.

**External regulation:** GDPR (Articles 5–43)
**Internal policy:** Hetzner Online GmbH — publicly available privacy policy (retrieved April 2026, policy version April 16, 2025)

A gold standard is constructed by using an LLM to introduce deliberate deviations of known types into a modified version of this policy. The detection pipeline runs against the modified policy; the introduced deviations serve as ground truth for evaluation.

## Repository Structure

```
├── data/
│   ├── gdpr/              # GDPR source HTML and extracted plain text (Art. 5–43)
│   ├── policy/            # Hetzner privacy policy (plain text)
│   ├── constraints/       # Extracted constraint sentences (GDPR + policy)
│   └── retrieval/         # Embedding-based matching results
├── src/
│   ├── preprocessing/     # GDPR extraction + constraint extraction scripts
│   ├── retrieval/         # Embedding-based constraint matching
│   ├── classification/    # LLM-based deviation classification (upcoming)
│   └── evaluation/        # Analysis against annotated subset (upcoming)
├── notebooks/             # Exploratory analyses
└── report/                # Report PDF and figures
```

## Pipeline

1. **Preprocessing:** Extract constraint sentences from GDPR (signal words: *shall*, *should*, *must*) and from the internal policy.
2. **Retrieval:** Embed constraints with `all-MiniLM-L6-v2` → cosine similarity matching (γ = 0.5). Unmatched GDPR constraints → `missing_coverage` candidates.
3. **Classification** *(upcoming):* For each matched pair, an LLM classifies the deviation type: `responsibility`, `execution_style`, `data`, `negation`, or `none`.
4. **Evaluation** *(upcoming):* Qualitative analysis + quantitative metrics on a manually annotated subset (Art. 15–22 vs. Hetzner Section 5).

## Current Results (Steps 1–2)

| Metric | Value |
|--------|-------|
| GDPR constraints (Art. 5–43) | 279 |
| Policy constraints (Hetzner) | 28 |
| Matched pairs (γ = 0.5) | 184 (65.9%) |
| Unmapped GDPR constraints | 95 (34.1%) |

The Art. 15–22 (data subject rights) vs. Hetzner Section 5 analysis is the primary focus for manual review, as it is a well-bounded, interpretable subset.

## Reference Paper

Sai, C., Winter, K., Fernanda, E., & Rinderle-Ma, S. (2023). *Detecting Deviations Between External and Internal Regulatory Requirements for Improved Process Compliance Assessment*. CAiSE 2023, LNCS 13901, pp. 401–416.
