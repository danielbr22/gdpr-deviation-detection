#!/usr/bin/env python3
"""
Extract constraint sentences from the GDPR (Art. 5-43) and a company policy.

GDPR extraction:
  Signal-word based (shall/should/must). Parent clauses ending with ":" are
  joined with their sub-items "(a) ...", "(b) ..." to form complete constraints.

Policy extraction:
  Zero-shot classification via facebook/bart-large-mnli. Each sentence is
  scored against the label "legal obligation or right" — sentences above
  ZSC_THRESHOLD are kept. This captures real-world policy language ("you have
  the right to", "we will", "you can request") that signal words miss.

Output:
  data/constraints/gdpr_constraints.json
  data/constraints/policy_constraints.json

Each record:
  {
    "id":      "gdpr_042",
    "source":  "gdpr" | "policy",
    "article": 5,            # gdpr only
    "section": "2. DATA …",  # policy only
    "text":    "Personal data shall be: processed lawfully …"
  }
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import spacy
from transformers import pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
GDPR_TEXT = ROOT / "data/gdpr/gdpr_art5_43.txt"
POLICY_TEXT = ROOT / "data/policy/hetzner_privacy_policy.txt"
OUT_DIR = ROOT / "data/constraints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZSC_THRESHOLD = 0.50
ZSC_LABELS = ["legal obligation or right", "factual statement or other information"]

SIGNAL_RE = re.compile(r"\b(shall|should|must)\b", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^\s{2,}\([a-z0-9ivx]+\)\s+")  # "  (a) ..." or "  (ii) ..."
ARTICLE_RE = re.compile(r"^Article\s+(\d+)\s*$")
SEPARATOR_RE = re.compile(r"^={3,}")
DEVIATION_BLOCK_RE = re.compile(r"\[DEVIATION.*?\]", re.DOTALL)
SECTION_HEADER_RE = re.compile(r"^={10,}\n(.+?)\n={10,}", re.MULTILINE)
# Matches subsection headers like "2.2 Purpose Limitation" at the start of a sentence
SUBSECTION_PREFIX_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z\s]+?\s+(?=[A-Z])")


def _is_boilerplate(line: str) -> bool:
    """Skip section dividers and empty lines (NOT article headers — handled separately)."""
    stripped = line.strip()
    if not stripped:
        return True
    if SEPARATOR_RE.match(stripped):
        return True
    return False


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# GDPR extractor
# ---------------------------------------------------------------------------

def extract_gdpr_constraints(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()

    constraints: list[dict] = []
    article_num = None
    parent_clause: str | None = None  # last non-list line that ends with ":"

    for raw in lines:
        if _is_boilerplate(raw):
            continue

        m = ARTICLE_RE.match(raw.strip())
        if m:
            article_num = int(m.group(1))
            parent_clause = None
            continue

        if LIST_ITEM_RE.match(raw):
            # Sub-item: strip leading whitespace + marker, join with parent
            content = LIST_ITEM_RE.sub("", raw).strip()
            if parent_clause:
                # Combine: "Personal data shall be: processed lawfully..."
                text = _clean(parent_clause.rstrip(":") + ": " + content)
            else:
                text = _clean(content)
            if SIGNAL_RE.search(text):
                constraints.append({
                    "id": f"gdpr_{len(constraints) + 1:03d}",
                    "source": "gdpr",
                    "article": article_num,
                    "text": text,
                })
        else:
            stripped = raw.strip()
            # Remove leading paragraph numbering "1. ", "2. ", "(1) " etc.
            stripped = re.sub(r"^\d+\.\s+", "", stripped)
            stripped = re.sub(r"^\(\d+\)\s+", "", stripped)

            if stripped.endswith(":"):
                # This is a parent clause — store it, emit only if it has signal words
                parent_clause = stripped
                if SIGNAL_RE.search(stripped):
                    # The parent itself is a meaningful constraint header; still emit it
                    # but the real value comes from its joined children above.
                    pass
            else:
                parent_clause = None
                text = _clean(stripped)
                if text and SIGNAL_RE.search(text):
                    constraints.append({
                        "id": f"gdpr_{len(constraints) + 1:03d}",
                        "source": "gdpr",
                        "article": article_num,
                        "text": text,
                    })

    return constraints


# ---------------------------------------------------------------------------
# Policy extractor
# ---------------------------------------------------------------------------

def _strip_annotations(text: str) -> str:
    return DEVIATION_BLOCK_RE.sub("", text)


def _current_section(position: int, section_positions: list[tuple[int, str]]) -> str:
    section = ""
    for pos, name in section_positions:
        if pos <= position:
            section = name
        else:
            break
    return section


def extract_policy_constraints(path: Path, nlp, classifier) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8")
    clean_text = _strip_annotations(raw_text)

    # Strip separator lines first so all subsequent character positions are consistent
    clean_text = re.sub(r"^[=\-]{3,}\s*$", "", clean_text, flags=re.MULTILINE)

    # Add periods after bare subsection headers so spaCy treats them as sentence boundaries
    clean_text = re.sub(
        r"^(\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n.!?]+)$",
        r"\1.",
        clean_text,
        flags=re.MULTILINE,
    )

    # Locate section headers for metadata — detected after stripping so positions are valid
    SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", re.MULTILINE)
    section_positions: list[tuple[int, str]] = []
    for m in SECTION_RE.finditer(clean_text):
        section_positions.append((m.start(), m.group(0).strip()))

    doc = nlp(clean_text)

    # Collect candidate sentences, joining colon-terminated headers with the next sentence
    raw_candidates: list[tuple[str, int]] = []
    for sent in doc.sents:
        text = _clean(sent.text)
        if not text:
            continue
        if re.match(r"^[=\-]{5,}", text) or re.match(r"^\d+\.\s+[A-Z\s]+$", text):
            continue
        raw_candidates.append((text, sent.start_char))

    candidates: list[tuple[str, int]] = []
    i = 0
    while i < len(raw_candidates):
        text, start_char = raw_candidates[i]
        if text.endswith(":") and i + 1 < len(raw_candidates):
            next_text, _ = raw_candidates[i + 1]
            candidates.append((_clean(text + " " + next_text), start_char))
            i += 2
        else:
            candidates.append((text, start_char))
            i += 1

    if not candidates:
        return []

    # Zero-shot classify all candidates in one batch
    texts = [t for t, _ in candidates]
    print(f"  Running zero-shot classifier on {len(texts)} candidate sentences …")
    results = classifier(texts, candidate_labels=ZSC_LABELS, multi_label=False)

    seen: set[str] = set()
    constraints: list[dict] = []
    for (text, start_char), result in zip(candidates, results):
        # Filter 1: question headers are never constraints
        if text.endswith("?"):
            continue
        # Filter 2: fragments — starts with lowercase, ")", or "(" signals a mid-sentence split
        if text[0].islower() or text.startswith(")") or text.startswith("("):
            continue
        # Filter 3: separator artifacts
        if "===" in text or "---" in text:
            continue
        # Filter 4: deduplicate — same sentence repeated across sections
        if text in seen:
            continue

        top_label = result["labels"][0]
        top_score = result["scores"][0]
        # Lower threshold for sentences that explicitly name a right — the phrasing
        # "you can request" scores conservatively but the context is unambiguous
        threshold = 0.40 if re.match(r"^Right (to|of)\b", text) else ZSC_THRESHOLD
        if top_label == ZSC_LABELS[0] and top_score >= threshold:
            seen.add(text)
            section = _current_section(start_char, section_positions)
            constraints.append({
                "id": f"pol_{len(constraints) + 1:03d}",
                "source": "policy",
                "section": section,
                "zsc_score": round(top_score, 4),
                "text": text,
            })

    return constraints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY_TEXT,
                        help="Path to policy text file (default: hetzner_privacy_policy.txt)")
    parser.add_argument("--output-policy", type=Path, default=OUT_DIR / "policy_constraints.json",
                        help="Output path for policy constraints JSON")
    parser.add_argument("--skip-gdpr", action="store_true",
                        help="Skip GDPR extraction (reuse existing gdpr_constraints.json)")
    args = parser.parse_args()

    print("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    print("Loading zero-shot classifier (facebook/bart-large-mnli) ...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU
    )

    if not args.skip_gdpr:
        print(f"Extracting GDPR constraints from {GDPR_TEXT.name} ...")
        gdpr = extract_gdpr_constraints(GDPR_TEXT)
        out_gdpr = OUT_DIR / "gdpr_constraints.json"
        out_gdpr.write_text(json.dumps(gdpr, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {len(gdpr)} constraints saved to {out_gdpr.name}")

    policy_path = Path(args.policy)
    print(f"Extracting policy constraints from {policy_path.name} ...")
    print(f"  ZSC threshold: {ZSC_THRESHOLD}  |  label: '{ZSC_LABELS[0]}'")
    policy = extract_policy_constraints(policy_path, nlp, classifier)
    out_pol = Path(args.output_policy)
    out_pol.parent.mkdir(parents=True, exist_ok=True)
    out_pol.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {len(policy)} constraints saved to {out_pol.name}")

    if not args.skip_gdpr:
        print("\nDone. Sample GDPR constraints:")
        for c in gdpr[:3]:
            print(f"  [{c['id']}] Art.{c['article']}: {c['text'][:90]}...")
    print("\nPolicy constraints extracted:")
    for c in policy:
        print(f"  [{c['id']}] score={c['zsc_score']} | {c['text'][:90]}")


if __name__ == "__main__":
    main()
