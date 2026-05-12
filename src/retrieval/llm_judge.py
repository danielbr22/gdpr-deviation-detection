#!/usr/bin/env python3
"""
LLM-as-judge — Step 3 of the hybrid pipeline.

Reads top-k candidates per GDPR constraint (output of embed_and_match --top-k)
and asks Qwen3.5:9b to decide which candidate (if any) has a substantive
connection to the GDPR constraint.

Input:
  data/retrieval/<use-case>/topk_candidates.json

Output:
  data/retrieval/<use-case>/matched_pairs.json    — pairs where LLM found a match
  data/retrieval/<use-case>/unmapped_gdpr.json    — GDPR constraints with no match

Usage:
  python3 src/retrieval/llm_judge.py \\
      --topk data/retrieval/hetzner/topk_candidates.json \\
      --output-dir data/retrieval/hetzner
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from src.retrieval.judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"


def check_ollama() -> None:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR: Ollama unreachable at localhost:11434 — {e}")


def parse_judge_response(content: str, n_candidates: int):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    try:
        match = int(data["match"])
    except (KeyError, TypeError, ValueError):
        return None
    if match < 0 or match > n_candidates:
        return None
    if not data.get("reasoning"):
        return None
    return {"match": match, "reasoning": data["reasoning"]}


def call_ollama(gdpr_text: str, gdpr_article: int, candidates: list):
    user_msg = "/no_think\n" + build_judge_prompt(gdpr_text, gdpr_article, candidates)
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    for _ in range(2):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            result = parse_judge_response(r.json()["message"]["content"], len(candidates))
            if result:
                return result
        except Exception:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge: filter top-k retrieval candidates to genuine matches"
    )
    parser.add_argument("--topk", type=Path, required=True,
                        help="topk_candidates.json from embed_and_match --top-k")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write matched_pairs.json and unmapped_gdpr.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge first N GDPR constraints (for testing)")
    args = parser.parse_args()

    check_ollama()

    topk_data = json.loads(args.topk.read_text(encoding="utf-8"))
    if args.limit:
        topk_data = topk_data[: args.limit]
    print(f"Loaded {len(topk_data)} GDPR constraints with top-k candidates")
    print(f"Model: {MODEL}")

    matched = []
    unmapped = []

    for i, entry in enumerate(topk_data, 1):
        candidates = entry["candidates"]
        print(f"  [{i}/{len(topk_data)}] {entry['gdpr_id']}", flush=True)

        result = call_ollama(entry["gdpr_text"], entry.get("gdpr_article", 0), candidates)

        if result and result["match"] > 0:
            chosen = candidates[result["match"] - 1]
            matched.append({
                "gdpr_id": entry["gdpr_id"],
                "policy_id": chosen["policy_id"],
                "similarity": chosen["similarity"],
                "gdpr_article": entry.get("gdpr_article"),
                "gdpr_text": entry["gdpr_text"],
                "policy_section": chosen.get("policy_section", ""),
                "policy_text": chosen["policy_text"],
            })
        else:
            best = candidates[0] if candidates else {}
            unmapped.append({
                "gdpr_id": entry["gdpr_id"],
                "best_similarity": best.get("similarity", 0.0),
                "best_policy_id": best.get("policy_id", ""),
                "gdpr_article": entry.get("gdpr_article"),
                "gdpr_text": entry["gdpr_text"],
                "deviation_type": "missing_coverage",
            })

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_pairs.json").write_text(json.dumps(matched, indent=2, ensure_ascii=False))
    (out_dir / "unmapped_gdpr.json").write_text(json.dumps(unmapped, indent=2, ensure_ascii=False))

    print(f"\n{'─' * 50}")
    print(f"Matched:  {len(matched)}")
    print(f"Unmapped: {len(unmapped)}")
    coverage = len(matched) / len(topk_data) if topk_data else 0.0
    print(f"Coverage: {coverage:.1%}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
