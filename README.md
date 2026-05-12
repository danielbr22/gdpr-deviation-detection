# Detecting Deviations Between External and Internal Regulatory Requirements
**TUM Master Praktikum — NLP for Information Systems, SS26**

## Project Overview

This project extends Sai et al. (2023) by replacing their classical NLP pipeline with LLM-based methods for detecting deviations between external regulatory documents and internal company privacy policies.

**External regulation:** GDPR (Articles 5–43)
**Internal policies:** Hetzner Online GmbH, Zalando SE, Trade Republic Bank GmbH

A gold standard is constructed by introducing deliberate deviations of known types into a modified version of each policy. The detection pipeline runs against the modified policy; the introduced deviations serve as ground truth for evaluation.

## Repository Structure

```
├── data/
│   ├── gdpr/              # GDPR source HTML and extracted plain text (Art. 5–43)
│   ├── policy/            # Privacy policies: original + modified (gold standard)
│   ├── constraints/       # Extracted constraint sentences (GDPR + per-use-case hybrid)
│   ├── retrieval/         # Retrieval results per use case (*_hybrid/ subdirs)
│   └── classification/    # LLM deviation classification results per use case
├── gold_standard/         # Deviation manifests (ground truth) for all 3 use cases
├── src/
│   ├── preprocessing/     # GDPR extraction (extract_gdpr.py) + hybrid policy extraction
│   ├── retrieval/         # Legal-BERT embedding + LLM-as-judge matching
│   ├── classification/    # LLM-based deviation classification
│   └── evaluation/        # Precision/recall/F1 metrics against gold standard
├── notebooks/             # Exploratory analyses
├── run_pipeline.sh        # End-to-end pipeline for all 3 use cases
└── report/                # Report PDF and figures
```

## Pipeline

1. **Preprocessing — GDPR:** Extract constraint sentences using signal words (*shall*, *should*, *must*) from the official EUR-Lex XHTML → `data/constraints/gdpr_constraints.json` (279 constraints).

2. **Preprocessing — Policy (hybrid):** For each policy sentence, a local LLM (Qwen3.5 9B via Ollama) with ±5-sentence context decides whether the sentence describes a data-protection obligation → `data/constraints/<use_case>_hybrid_constraints.json`.

3. **Retrieval:** Legal-BERT encodes GDPR and policy constraints. Top-5 policy candidates are retrieved per GDPR constraint, then an LLM judge confirms which (if any) substantively addresses the obligation. Unmatched GDPR constraints → `missing_coverage`.

4. **Classification:** For each matched pair, an LLM classifies the deviation type: `responsibility`, `execution_style`, `data`, `negation`, or `none`. Unmatched → `missing_coverage` automatically.

5. **Evaluation:** Precision, recall, F1 per deviation type and overall, compared against the gold standard manifests.

Run the full pipeline: `bash run_pipeline.sh`

## Use Cases

| Use Case | Policy | Deviations introduced |
|----------|--------|----------------------|
| Hetzner Online GmbH | `hetzner_policy_modified.txt` | 5 |
| Zalando SE | `zalando_policy_modified.txt` | 5 |
| Trade Republic Bank GmbH | `traderepublic_policy_modified.txt` | 5 |

## Reference Paper

Sai, C., Winter, K., Fernanda, E., & Rinderle-Ma, S. (2023). *Detecting Deviations Between External and Internal Regulatory Requirements for Improved Process Compliance Assessment*. CAiSE 2023, LNCS 13901, pp. 401–416.
