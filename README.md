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
├── ui/                    # Pipeline dashboard (FastAPI + React)
├── run_pipeline.sh        # End-to-end pipeline for all 3 use cases
└── report/                # Report PDF and figures
```

## Quick Start (Docker)

The full pipeline — including the local LLM (Qwen3.5 9B via Ollama) and all Python dependencies — runs inside Docker containers. No manual installation required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows) **or** Docker Engine + Docker Compose plugin (Linux)
- ~15 GB free disk space (model ~5.5 GB + image ~3 GB + runtime outputs)
- 16 GB RAM recommended (8 GB minimum; CPU-only inference is slow — expect several hours for a full run)

> Works on **macOS, Linux, and Windows** (WSL2). No GPU required, but see GPU section below for faster inference.

### Run

```bash
git clone <repo-url>
cd project
docker compose up
```

On first run Docker will automatically:
1. Build the Python image and install all dependencies
2. Start Ollama and pull Qwen3.5:9b (~5.5 GB — cached in a Docker volume for subsequent runs)
3. Download Legal-BERT embeddings (~400 MB — also cached)
4. Execute the full pipeline across all 3 use cases

Results are written to `data/` and `logs/` on your host machine so they can be inspected directly.

### GPU acceleration (Linux / Windows with NVIDIA)

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

This passes all available GPUs into the Ollama container, reducing inference time significantly.

### Smoke test (~5 minutes)

Verifies the full pipeline is wired up correctly on a tiny policy slice:

```bash
docker compose run --rm pipeline bash run_pipeline.sh --test
```

### Running locally (without Docker)

Requires Python 3.11+, [Ollama](https://ollama.com) with `qwen3.5:9b` pulled, and:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash run_pipeline.sh
```

---

## Dashboard UI

A lightweight web dashboard lets you trigger pipeline runs, stream live log output, and browse results without touching the terminal.

### Start

```bash
bash ui/start.sh
```

Opens at **http://localhost:8000**. The script builds the React frontend and starts a FastAPI server. Requires Python 3.11+ and Node.js (for the one-time build).

### Features

- **New Run tab** — configure and launch a pipeline run with options:
  - *Smoke test* — tiny Hetzner slice to verify the full chain (~5 min)
  - *Force re-run* — ignore skip guards and re-run all phases from scratch
  - *LLM Provider* — switch between Local (Ollama) and API (OpenAI); API key read from `.env` automatically
- **Live output** — log lines stream in real time with phase progress indicator
- **History tab** — browse past runs, view their logs, and inspect saved result snapshots

### Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your values. At minimum, set `OPENAI_API_KEY` to use the OpenAI provider (auto-detected by the dashboard):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Leave it blank (or omit the file) to default to local Ollama inference.

---

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
