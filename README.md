# Detecting Deviations Between External and Internal Regulatory Requirements
**TUM Master Praktikum — NLP for Information Systems, SS26**

## Project Overview

This project extends Sai et al. (2023) by replacing their classical NLP pipeline with LLM-based methods for detecting deviations between external regulatory documents and internal company privacy policies.

**External regulation:** GDPR (Articles 5–43)
**Internal policies:** Hetzner Online GmbH, Zalando SE, Trade Republic Bank GmbH

A gold standard is constructed by introducing deliberate deviations of known types into a modified version of each policy. The detection pipeline runs against the modified policy; the introduced deviations serve as ground truth for evaluation.

---

## Quick Start

The recommended way to run the project is via Docker — no manual Python or Node installation required. Everything runs through the web dashboard.

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd gdpr-deviation-detection
```

### Step 2 — Configure the LLM provider

**OpenAI is strongly recommended.** Local Ollama (CPU) is extremely slow — a single pipeline run takes many hours. OpenAI completes the same run in minutes.

Copy the example env file and fill in your key:

```bash
cp .env.example .env
# then open .env and set OPENAI_API_KEY=sk-...
```

The dashboard auto-detects the key and switches to OpenAI automatically. No other changes needed.

> **Local Ollama alternative:** if you prefer not to use OpenAI, leave `.env` as-is. Be aware that the 9B model requires ~7.9 GB of RAM inside Docker. On macOS/Windows you must first raise Docker Desktop's memory limit: **Settings → Resources → Advanced → Memory → 12 GB → Apply & Restart.** Expect each full pipeline run to take several hours.

### Step 3 — Start

```bash
bash start.sh
```

On macOS this wraps `docker compose up` with `caffeinate` so the machine does not sleep mid-run and interrupt API calls. On Linux/Windows it is equivalent to `docker compose up`.

On first run this will:
1. Build the image (Python deps + React frontend compiled inside Docker)
2. Start Ollama (model is downloaded on first pipeline run if using local inference)
3. Serve the dashboard at **http://localhost:8000**

### Step 4 — Run the pipeline

Open **http://localhost:8000** in your browser. From the **New Run** tab:

- Confirm the LLM provider shown matches your choice (OpenAI or Local)
- Toggle **Skip scope (Phase 0)** to reuse existing outputs and save time on re-runs
- Click **Start** — log output streams live in the browser

Results land in `data/evaluation/results.json` on your host machine. Past runs are browsable in the **History** tab.

> **Changing provider after startup:** edit `.env`, then run `docker compose restart ui` to pick up the new value.

---

## Pipeline

`run.sh` orchestrates the full detection chain across all three use cases.

### Phases

| # | Name | Description | Output |
|---|------|-------------|--------|
| 0 | Scope detection | Runs extraction + retrieval on the *original* policy to identify which GDPR articles are substantively in scope — used to filter false positives | `data/retrieval/<uc>_original/` |
| 1 | Policy extraction | LLM (±5-sentence context) extracts data-protection obligations from the *modified* policy | `data/constraints/<uc>_hybrid_constraints.json` |
| 2 | Retrieval | Legal-BERT top-20 candidates + LLM judge; unmatched GDPR constraints → `missing_coverage` | `data/retrieval/<uc>_hybrid/` |
| 3 | Manifest remapping | Fills null `policy_constraint_id` fields in the gold standard manifests | — |
| 4 | Classification | LLM labels each matched pair with a deviation type | `data/classification/<uc>_hybrid_classified.json` |
| 5 | Evaluation | Precision / recall / F1 vs. gold standard | `data/evaluation/results.json` |

Phases are skipped when their output already exists (resume-safe). The **Force re-run** option in the dashboard disables this.

## Use Cases

| Use Case | Policy | Deviations introduced |
|----------|--------|----------------------|
| Hetzner Online GmbH | `hetzner_policy_modified.txt` | 17 |
| Zalando SE | `zalando_policy_modified.txt` | 17 |
| Trade Republic Bank GmbH | `traderepublic_policy_modified.txt` | 17 |

17 deviations per use case: 3 per type × 6 RCASR types (`constraint_coverage`, `severity`, `execution_style`, `negation`, `responsibility`, `data`), minus one structurally undetectable Art. 77 deviation.

## Repository Structure

```
├── data/
│   ├── gdpr/              # GDPR source HTML and extracted plain text (Art. 5–43)
│   ├── policy/            # Privacy policies: original + modified (gold standard)
│   ├── constraints/       # Extracted constraint sentences (GDPR + per-use-case hybrid)
│   ├── retrieval/         # Retrieval results (*_hybrid/ and *_original/ per use case)
│   ├── classification/    # LLM deviation classification results
│   └── evaluation/        # results.json — precision/recall/F1
├── gold_standard/         # Deviation manifests (ground truth) for all 3 use cases
├── src/
│   ├── preprocessing/     # GDPR extraction + hybrid policy extraction
│   ├── retrieval/         # Legal-BERT embedding + LLM-as-judge
│   ├── classification/    # LLM deviation classifier
│   ├── evaluation/        # Metrics against gold standard
│   └── utils/             # LLM client, manifest remapping
├── ui/                    # Dashboard (FastAPI backend + React frontend)
├── run.sh                 # Pipeline entry point (invoked by the dashboard)
└── report/report.pdf      # Final report 
```

## Reference Paper

Sai, C., Winter, K., Fernanda, E., & Rinderle-Ma, S. (2023). *Detecting Deviations Between External and Internal Regulatory Requirements for Improved Process Compliance Assessment*. CAiSE 2023, LNCS 13901, pp. 401–416.
