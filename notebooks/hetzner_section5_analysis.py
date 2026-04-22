#!/usr/bin/env python3
"""
One-off analysis: run the pipeline on the Hetzner policy, filtered to GDPR
Art. 15–22 (the data subject rights articles that map to Section 5).

Runs in-memory — does not overwrite any data/ files.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
GDPR_CONSTRAINTS = ROOT / "data/constraints/gdpr_constraints.json"
HETZNER_POLICY   = ROOT / "data/policy/hetzner_privacy_policy.txt"

ARTICLES_OF_INTEREST = {15, 16, 17, 18, 20, 21, 22}
SIGNAL_RE = re.compile(r"\b(shall|should|must|have the right|can request|may request)\b", re.IGNORECASE)
GAMMA = 0.5  # lower threshold so we can see best-effort matches

# ---------------------------------------------------------------------------
# 1. Load GDPR constraints filtered to Art. 15–22
# ---------------------------------------------------------------------------
print("=== GDPR constraints (Art. 15–22) ===")
all_gdpr = json.loads(GDPR_CONSTRAINTS.read_text())
gdpr = [c for c in all_gdpr if c.get("article") in ARTICLES_OF_INTEREST]
print(f"Total GDPR constraints (Art. 5–43): {len(all_gdpr)}")
print(f"Filtered to Art. 15/16/17/18/20/21/22: {len(gdpr)}\n")

for art in sorted(ARTICLES_OF_INTEREST):
    n = sum(1 for c in gdpr if c["article"] == art)
    print(f"  Art. {art}: {n} constraints")

# ---------------------------------------------------------------------------
# 2. Extract policy constraints from Hetzner policy
#    Extended signal words to capture rights language
# ---------------------------------------------------------------------------
print("\n=== Extracting Hetzner policy constraints ===")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

raw = HETZNER_POLICY.read_text(encoding="utf-8")
doc = nlp(raw)

policy = []
for sent in doc.sents:
    text = re.sub(r"\s+", " ", sent.text).strip()
    if not text or not SIGNAL_RE.search(text):
        continue
    if re.match(r"^[=\-]{5,}", text):
        continue
    policy.append({"id": f"pol_{len(policy)+1:03d}", "text": text})

print(f"Policy constraints extracted (extended signal words): {len(policy)}")

# Show which ones come from Section 5 area
section5_start = raw.find("5. What rights do I have")
section5_end   = raw.find("6. Making complaints")
section5_text  = raw[section5_start:section5_end] if section5_start > 0 else ""

sec5_constraints = [c for c in policy if c["text"][:60] in section5_text or
                    any(kw in c["text"].lower() for kw in
                        ["right of access","right to rectification","right to erasure",
                         "right to restriction","data portability","right to object",
                         "automated decision"])]
print(f"  → from Section 5 (rights): {len(sec5_constraints)}")
for c in sec5_constraints:
    print(f"     {c['text'][:100]}")

# ---------------------------------------------------------------------------
# 3. Embed and match
# ---------------------------------------------------------------------------
print("\n=== Embedding & matching (γ=0.5) ===")
model = SentenceTransformer("all-MiniLM-L6-v2")
gdpr_emb   = model.encode([c["text"] for c in gdpr],   show_progress_bar=False)
policy_emb = model.encode([c["text"] for c in policy], show_progress_bar=False)

scores = util.cos_sim(gdpr_emb, policy_emb).numpy()

matched, unmapped = [], []
for i, g in enumerate(gdpr):
    best_j     = int(np.argmax(scores[i]))
    best_score = float(scores[i][best_j])
    if best_score >= GAMMA:
        matched.append({
            "gdpr_article": g["article"],
            "similarity":   round(best_score, 4),
            "gdpr_text":    g["text"],
            "policy_text":  policy[best_j]["text"],
        })
    else:
        unmapped.append({
            "gdpr_article": g["article"],
            "best_sim":     round(best_score, 4),
            "gdpr_text":    g["text"],
            "best_policy":  policy[best_j]["text"],
        })

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
print(f"\nMatched (sim ≥ {GAMMA}): {len(matched)}")
print(f"Unmapped:               {len(unmapped)}")
print(f"Coverage rate:          {len(matched)/len(gdpr):.1%}\n")

print("─" * 70)
print("MATCHED PAIRS (by article)")
print("─" * 70)
for art in sorted(ARTICLES_OF_INTEREST):
    art_matches = [m for m in matched if m["gdpr_article"] == art]
    art_unmapped = [u for u in unmapped if u["gdpr_article"] == art]
    print(f"\nArt. {art}  — {len(art_matches)} matched, {len(art_unmapped)} unmapped")
    for m in art_matches:
        print(f"  sim={m['similarity']:.3f}")
        print(f"  GDPR:   {m['gdpr_text'][:90]}")
        print(f"  Policy: {m['policy_text'][:90]}")
    for u in art_unmapped:
        print(f"  [UNMAPPED sim={u['best_sim']:.3f}] {u['gdpr_text'][:90]}")
        print(f"           best: {u['best_policy'][:80]}")
