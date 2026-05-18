# Gold Standard — Deviation Manifests

## What this is

This directory contains the ground truth for evaluating the deviation detection pipeline.
Three JSON manifests document deliberately introduced deviations in the three modified
company privacy policies. The detection pipeline runs against the **modified** policies,
not the originals.

| Use case | Manifest | Modified policy |
|----------|----------|----------------|
| Hetzner Online GmbH | `deviation_manifest.json` | `data/policy/hetzner_policy_modified.txt` |
| Zalando SE | `zalando_deviation_manifest.json` | `data/policy/zalando_policy_modified.txt` |
| Trade Republic Bank GmbH | `traderepublic_deviation_manifest.json` | `data/policy/traderepublic_policy_modified.txt` |

---

## Gold standard construction

**17 deviations per use case** (3 per RCASR type × 6 types = 18 originally, minus one
Art. 77 constraint_coverage deviation removed from each manifest because Art. 77 is
outside the GDPR Art. 5–43 scope and has no counterpart in `gdpr_constraints.json`,
making those deviations structurally undetectable).

**6 deviation types (RCASR types):**

| RCASR | Type | Description |
|-------|------|-------------|
| RCASR3 | `constraint_coverage` | A GDPR topic is absent from the policy |
| RCASR4 | `severity` | Obligation strength is weakened (shall → may, must → should) |
| RCASR5 | `execution_style` | How an obligation is fulfilled differs |
| RCASR6 | `negation` | Policy states the opposite of what GDPR requires |
| RCASR7 | `responsibility` | Wrong party assigned to an obligation |
| RCASR8 | `data` | Wrong or narrower data scope referenced |

RCASR9/10 (time, task order) are left as future work per the original paper.

---

## How deviations were introduced

1. **Base document**: the real company privacy policy (retrieved from the company website).
2. **Scope**: whole policy, not just one section.
3. **Generation**: deviations were introduced in a supervised interactive session, reviewed
   and accepted by the developer before committing.
4. **Circularity guard**: the local model used for classification (Qwen3.5 9B via Ollama)
   is a different model from Claude Sonnet 4.6, which introduced the deviations. This
   ensures the model that introduced deviations does not detect them.

---

## Evaluation

Evaluation is computed by `src/evaluation/evaluate.py` against each manifest:

- **True positive (TP)**: pipeline flags a deviation of the correct type for a GDPR/policy
  pair whose article matches a manifest entry.
- **False positive (FP)**: pipeline flags a deviation where none was introduced.
- **False negative (FN)**: a manifest entry is not detected by the pipeline.

Results (precision/recall/F1 per type and overall, across all three use cases) are written
to `data/evaluation/results.json`.

Note: precision is a lower bound — the gold standard covers only the 17 deliberately
introduced deviations and does not constitute an exhaustive compliance audit. Pipeline
outputs beyond those 17 cases are counted as false positives, though some may represent
genuine compliance gaps in the original policy. Recall is the primary reliability metric.
