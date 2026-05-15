#!/usr/bin/env python3
"""
Fill null policy_constraint_id fields in gold standard manifests.

After Phase 0 (scope detection) runs hybrid_extract on the original policies,
each extracted constraint gets a pol_XXX ID. This script matches each manifest
entry's original_text against those extracted constraints using fuzzy string
similarity and writes the best-matching ID back into the manifest.

Only null entries are touched; existing IDs are never overwritten.
"""

import difflib
import json
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "gold_standard"
CONSTRAINTS_DIR = ROOT / "data" / "constraints"

USE_CASES = {
    "hetzner": {
        "manifest": GOLD_DIR / "deviation_manifest.json",
        "constraints": CONSTRAINTS_DIR / "hetzner_original_constraints.json",
    },
    "zalando": {
        "manifest": GOLD_DIR / "zalando_deviation_manifest.json",
        "constraints": CONSTRAINTS_DIR / "zalando_original_constraints.json",
    },
    "traderepublic": {
        "manifest": GOLD_DIR / "traderepublic_deviation_manifest.json",
        "constraints": CONSTRAINTS_DIR / "traderepublic_original_constraints.json",
    },
}

MIN_RATIO = 0.45


def best_match(text: str, constraints: list) -> Tuple[Optional[str], float]:
    best_id, best_ratio = None, 0.0
    for c in constraints:
        ratio = difflib.SequenceMatcher(None, text, c["text"]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = c["id"]
    return (best_id, best_ratio) if best_ratio >= MIN_RATIO else (None, best_ratio)


def remap_use_case(name: str, cfg: dict) -> int:
    manifest_path: Path = cfg["manifest"]
    constraints_path: Path = cfg["constraints"]

    if not constraints_path.exists():
        print(f"  [{name}] {constraints_path.name} not found — run Phase 0 first, skipping")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(constraints_path) as f:
        constraints = json.load(f)

    updated = 0
    for entry in manifest:
        if entry.get("policy_constraint_id") is not None:
            continue
        original_text = entry.get("original_text", "")
        if not original_text:
            continue
        matched_id, ratio = best_match(original_text, constraints)
        if matched_id:
            entry["policy_constraint_id"] = matched_id
            updated += 1
            print(f"  [{name}] {entry['id']} ({entry['deviation_type']}) → {matched_id}  sim={ratio:.2f}")
        else:
            print(f"  [{name}] {entry['id']} ({entry['deviation_type']}) — no match (best sim={ratio:.2f})")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return updated


def main() -> None:
    total = 0
    for name, cfg in USE_CASES.items():
        print(f"\nRemapping: {name}")
        total += remap_use_case(name, cfg)
    print(f"\nTotal entries updated: {total}")


if __name__ == "__main__":
    main()
