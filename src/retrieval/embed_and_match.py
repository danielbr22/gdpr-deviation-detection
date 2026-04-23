#!/usr/bin/env python3
"""
Embed GDPR and policy constraints with sentence-transformers, then match each
GDPR constraint to its best policy counterpart via cosine similarity.

Input:
  data/constraints/gdpr_constraints.json
  data/constraints/policy_constraints.json

Output:
  data/retrieval/matched_pairs.json      — pairs above threshold γ
  data/retrieval/unmapped_gdpr.json      — GDPR constraints below threshold
                                            (= missing_coverage candidates for Step 3)

Usage:
  python3 src/retrieval/embed_and_match.py [--gamma 0.7] [--model all-MiniLM-L6-v2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS_DIR = ROOT / "data/constraints"
OUT_DIR = ROOT / "data/retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_constraints(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def embed(model: SentenceTransformer, constraints: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in constraints]
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=True)


def match(
    gdpr: list[dict],
    policy: list[dict],
    gdpr_emb: np.ndarray,
    policy_emb: np.ndarray,
    gamma: float,
) -> tuple[list[dict], list[dict]]:
    """
    For each GDPR constraint find the highest-similarity policy constraint.
    Returns (matched_pairs, unmapped_gdpr).

    One GDPR constraint maps to at most one policy constraint (best match).
    Multiple GDPR constraints can map to the same policy constraint.
    """
    # Full cosine similarity matrix: shape (n_gdpr, n_policy)
    scores = util.cos_sim(gdpr_emb, policy_emb).numpy()  # type: ignore[arg-type]

    matched: list[dict] = []
    unmapped: list[dict] = []

    for i, g in enumerate(gdpr):
        best_j = int(np.argmax(scores[i]))
        best_score = float(scores[i][best_j])

        if best_score >= gamma:
            matched.append({
                "gdpr_id": g["id"],
                "policy_id": policy[best_j]["id"],
                "similarity": round(best_score, 4),
                "gdpr_article": g.get("article"),
                "gdpr_text": g["text"],
                "policy_section": policy[best_j].get("section", ""),
                "policy_text": policy[best_j]["text"],
            })
        else:
            unmapped.append({
                "gdpr_id": g["id"],
                "best_similarity": round(best_score, 4),
                "best_policy_id": policy[best_j]["id"],
                "gdpr_article": g.get("article"),
                "gdpr_text": g["text"],
                "deviation_type": "missing_coverage",
            })

    return matched, unmapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Cosine similarity threshold (default: 0.7)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model name")
    parser.add_argument("--policy-constraints", type=Path,
                        default=CONSTRAINTS_DIR / "policy_constraints.json",
                        help="Path to policy constraints JSON")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help="Directory for matched_pairs.json, unmapped_gdpr.json, run_metadata.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}  |  γ = {args.gamma}")

    gdpr = load_constraints(CONSTRAINTS_DIR / "gdpr_constraints.json")
    policy = load_constraints(Path(args.policy_constraints))
    print(f"Loaded {len(gdpr)} GDPR constraints, {len(policy)} policy constraints")

    print("Encoding …")
    model = SentenceTransformer(args.model)
    gdpr_emb = embed(model, gdpr)
    policy_emb = embed(model, policy)

    print("Matching …")
    matched, unmapped = match(gdpr, policy, gdpr_emb, policy_emb, args.gamma)

    # Save results
    meta = {
        "model": args.model,
        "gamma": args.gamma,
        "n_gdpr": len(gdpr),
        "n_policy": len(policy),
        "n_matched": len(matched),
        "n_unmapped": len(unmapped),
        "coverage_rate": round(len(matched) / len(gdpr), 3),
    }

    out_matched = out_dir / "matched_pairs.json"
    out_unmapped = out_dir / "unmapped_gdpr.json"
    out_meta = out_dir / "run_metadata.json"

    out_matched.write_text(json.dumps(matched, indent=2, ensure_ascii=False))
    out_unmapped.write_text(json.dumps(unmapped, indent=2, ensure_ascii=False))
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"\n{'─'*50}")
    print(f"Matched (above γ={args.gamma}):  {len(matched):>4}  →  {out_matched.name}")
    print(f"Unmapped (missing coverage):    {len(unmapped):>4}  →  {out_unmapped.name}")
    print(f"Coverage rate: {meta['coverage_rate']:.1%}")
    print(f"{'─'*50}")

    print("\nSample matched pairs:")
    for p in matched[:3]:
        print(f"  [{p['gdpr_id']} Art.{p['gdpr_article']}] sim={p['similarity']}")
        print(f"    GDPR:   {p['gdpr_text'][:80]}…")
        print(f"    Policy: {p['policy_text'][:80]}…")

    print("\nSample unmapped GDPR constraints:")
    for u in unmapped[:3]:
        print(f"  [{u['gdpr_id']} Art.{u['gdpr_article']}] best_sim={u['best_similarity']}")
        print(f"    {u['gdpr_text'][:80]}…")


if __name__ == "__main__":
    main()
