# Gold Standard — Deviation Manifest

## What this is

`deviation_manifest.json` is the ground truth for evaluating the deviation detection
pipeline. It documents five deliberately introduced deviations in
`data/policy/hetzner_policy_modified.txt` — a modified version of the real Hetzner
Online GmbH privacy policy (April 16, 2025).

The detection pipeline runs against the **modified** policy, not the original. The
manifest records exactly what was changed and why, enabling precise computation of
precision, recall, and F1 per deviation type.

---

## How the gold standard was created

1. **Base document**: `data/policy/hetzner_privacy_policy.txt` — real Hetzner policy
   retrieved 2026-04-22.

2. **Scope**: For now only Section 5 ("What rights do I have when it comes to my data?") was
   modified. This section maps to GDPR Art. 15–22 (data subject rights), making it
   the most bounded and verifiable subset for a focused evaluation.

3. **Deviation taxonomy**: The five deviation types follow the Regulatory Compliance
   Assessment Solution Requirements (RCASRs) defined in:

   > Sai et al. (2023). *Detecting Deviations Between External and Internal Regulatory
   > Requirements for Improved Process Compliance Assessment.* CAiSE 2023.

   Specifically: RCASR3 (constraint coverage), RCASR5 (execution style), RCASR6
   (negation), RCASR7 (responsibility), RCASR8 (data).

4. **Generation**: One deviation of each RCASR type was introduced by Claude Sonnet
   4.6 (claude-sonnet-4-6) in a supervised interactive session.
   The LLM selected the target sentences and authored both the modified text and the
   rationale field in the manifest. All changes were reviewed and accepted by the developer before committing.

5. **Circularity guard**: The local model used for classification (Step 3) is a
   different model from Claude Sonnet 4.6. This ensures the model that introduced
   deviations is not the same model that detects them.

---

## Deviation summary

| ID      | RCASR  | Type               | Target sentence    | GDPR article |
|---------|--------|--------------------|--------------------|--------------|
| dev_001 | RCASR3 | missing_coverage   | Right to data portability (deleted) | Art. 20 |
| dev_002 | RCASR5 | execution_style    | Right to restriction of processing  | Art. 18 |
| dev_003 | RCASR6 | negation           | Right to erasure                    | Art. 17 |
| dev_004 | RCASR7 | responsibility     | Contact for exercising rights       | Art. 12, 37 |
| dev_005 | RCASR8 | data               | Right to rectification              | Art. 16 |

Full rationale for each deviation is in `deviation_manifest.json`.

---

## Pipeline instructions

After generating the gold standard, re-run the pipeline on the modified policy:

```bash
# 1. Re-extract constraints from the modified policy
python src/preprocessing/extract_constraints.py \
    --policy data/policy/hetzner_policy_modified.txt \
    --output data/constraints/policy_constraints_modified.json

# 2. Re-run retrieval against modified policy constraints
python src/retrieval/embed_and_match.py \
    --gdpr data/constraints/gdpr_constraints.json \
    --policy data/constraints/policy_constraints_modified.json \
    --output-dir data/retrieval_modified/

# 3. Run classifier on matched pairs (Step 3)
# 4. Evaluate against this manifest (Step 4)
```

The `missing_coverage` deviation (dev_001) is detectable at the retrieval stage:
the GDPR Art. 20 constraint(s) will appear in `unmapped_gdpr.json` after Step 2.
Of the remaining four deviations, dev_002 and dev_003 survive to the classifier.
dev_004 is blocked at extraction (ZSC threshold) and dev_005 at retrieval (many-to-one
bottleneck) — see Retrieval Run Results below.

---

## Retrieval run results (2026-04-23)

Re-ran `extract_constraints.py` and `embed_and_match.py` on the modified policy
(γ=0.5, model=all-MiniLM-L6-v2). Outputs in `data/retrieval_modified/`.

| Metric | Original policy | Modified policy |
|--------|----------------|-----------------|
| Policy constraints extracted | 28 | 26 (−1: portability deleted; −1: dev_004 sentence below ZSC threshold) |
| GDPR constraints matched | 184 | 183 |
| GDPR constraints unmapped | 95 | 96 (+1: Art.20 newly unmapped) |

**Pipeline detectability per deviation:**

| ID | Type | Detectable? | Blocked at | Notes |
|----|------|------------|------------|-------|
| dev_001 | missing_coverage | ✓ | — | gdpr_122 Art.20 newly unmapped at retrieval |
| dev_002 | execution_style | ✓ | — | pol_022 matched to Art.18 (sim ≈ 0.73–0.77) |
| dev_003 | negation | ✓ | — | pol_020 matched to Art.17 (sim ≈ 0.65–0.69) |
| dev_004 | responsibility | ✗ | extraction | Modified sentence scores below ZSC threshold (< 0.5); never enters pipeline |
| dev_005 | data | ✗ | retrieval | pol_019 outcompeted by pol_022 for gdpr_099 Art.16 (sim 0.63 vs lower) |

**Finding**: The pipeline has two distinct bottlenecks.
- **Extraction bottleneck** (dev_004): the ZSC classifier does not score the modified sentence above the extraction threshold — non-legal-register language is not recognised as a legal obligation. The sentence never enters the pipeline at all.
- **Retrieval bottleneck** (dev_005): "account information" diverges enough from "personal data" to lower similarity, so a competing sentence wins the best-match slot. The deviation exists in the policy but is never paired with its GDPR counterpart.

Both bottlenecks are independent of classifier quality and motivate future work on extraction coverage and retrieval precision.

---

## Evaluation

Compare classifier output against this manifest for deviations that survive
retrieval (dev_001, dev_002, dev_003):

- **True positive**: classifier flags a deviation of the correct type for a pair
  whose GDPR/policy IDs match a manifest entry.
- **False positive**: classifier flags a deviation where none was introduced.
- **False negative**: a manifest entry (dev_001–003) is not detected.

Report precision, recall, and F1 per deviation type and overall, following the
format of Sai et al. Table 2 / Table 3. Note the retrieval-blocked deviations
(dev_004, dev_005) separately as evidence that retrieval quality caps recall.
