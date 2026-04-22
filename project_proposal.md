# Project Proposal — Daniel Bier

## Topic
Building on "Detecting Deviations Between External and Internal Regulatory Requirements for Improved Process Compliance Assessment", this project explores whether modern LLM-based methods can improve the detection of deviations between external regulatory documents and internal company policies. The paper's classical NLP approach provides the research context and motivation for applying more recent techniques.

## Data Input
- **External regulation:** GDPR (Articles 5–43)
- **Internal policy:** A real company privacy policy (Hetzner Online GmbH) used as the base, into which an LLM introduces deliberate deviations of known types to produce the final internal policy used for detection.

## Output Goal
A report identifying where an internal company policy fails to address, contradicts, or incompletely covers the requirements of an external regulation, classified by deviation type (responsibility, execution style, data, negation, missing coverage) with supporting evidence from both documents.

## Proposed Method
A two-stage pipeline is proposed: a retrieval step maps individual regulatory constraints to their most relevant counterparts in the company policy using embeddings, followed by an LLM-based classification step that determines the deviation type for each matched pair. Unmapped regulatory constraints are flagged as missing coverage.

The choice of retrieval model, LLM, prompting strategy, and constraint segmentation approach are open questions to be explored during the project.

## Proposed Gold Standard

A real company privacy policy (Hetzner Online GmbH) is taken as the starting point. An LLM then produces a modified version of that policy with deliberately introduced deviations of known types (e.g. responsibility deviation, data deviation, missing requirement, negation). Since the deviations are defined in advance against a realistic document, ground truth is exact and reproducible while the policy retains the structure and language of a real-world document. A different model is used for introducing deviations and for detection to avoid circularity.

## Evaluation
- Constraint mapping and deviation classification against the gold standard (precision, recall, F1 per deviation type)
- Cost and latency considerations for LLM-based components
