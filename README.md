# Detecting Deviations Between External and Internal Regulatory Requirements
**TUM Master Praktikum — NLP for Information Systems, SS26**
**Student:** Daniel Bier | **Supervisor:** Catherine Sai (catherine.sai@tum.de)

## Project Overview

This project extends Sai et al. (2023) by replacing their classical NLP pipeline with LLM-based methods for detecting deviations between external regulatory documents and internal company privacy policies.

**External regulation:** GDPR (Articles 5–43)
**Internal policy:** Hetzner Online GmbH — real-world privacy policy (retrieved April 2025)

Rather than a synthetically generated policy, this project uses a publicly available real-world document, making detected deviations genuine compliance gaps rather than planted ground truth.

## Key Deadlines

| Date | Event |
|------|-------|
| **Tue. 28.04** | **Intermediate Presentation** |
| **Tue. 19.05** | **Final Presentations + Submission** |

## Grading

- 40% Python implementation (this repo)
- 20% Report (6–7 pages PDF)
- 30% Presentations (10% intermediate + 20% final)
- 10% Participation

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
