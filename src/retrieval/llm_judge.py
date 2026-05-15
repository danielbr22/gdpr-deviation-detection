#!/usr/bin/env python3
"""
LLM-as-judge — Step 3 of the hybrid pipeline.

Reads top-k candidates per GDPR constraint and asks the LLM which candidate
(if any) has a substantive connection to the constraint.

Provider selected via LLM_PROVIDER env var; concurrent when using OpenAI.

Input:
  data/retrieval/<use-case>/topk_candidates.json

Output:
  data/retrieval/<use-case>/matched_pairs.json
  data/retrieval/<use-case>/unmapped_gdpr.json

Usage:
  python3 src/retrieval/llm_judge.py \
      --topk data/retrieval/hetzner/topk_candidates.json \
      --output-dir data/retrieval/hetzner
"""

import argparse
import asyncio
import json
from pathlib import Path

from src.retrieval.judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.utils import llm_client


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


async def run_judge_async(topk_data: list) -> tuple:
    sem = asyncio.Semaphore(llm_client.concurrency())

    async def judge_one(entry: dict):
        candidates = entry["candidates"]
        user_msg = build_judge_prompt(entry["gdpr_text"], entry.get("gdpr_article", 0), candidates)
        async with sem:
            content = await llm_client.call_async(
                JUDGE_SYSTEM_PROMPT, user_msg, json_mode=True, timeout=120
            )
        return entry, parse_judge_response(content, len(candidates))

    raw_results = await asyncio.gather(*[judge_one(e) for e in topk_data])

    matched = []
    unmapped = []
    for entry, result in raw_results:
        candidates = entry["candidates"]
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

    return matched, unmapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge: filter top-k retrieval candidates to genuine matches"
    )
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    llm_client.check_provider()

    topk_data = json.loads(args.topk.read_text(encoding="utf-8"))
    if args.limit:
        topk_data = topk_data[: args.limit]
    print(f"Loaded {len(topk_data)} GDPR constraints")

    matched, unmapped = asyncio.run(run_judge_async(topk_data))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_pairs.json").write_text(json.dumps(matched, indent=2, ensure_ascii=False))
    (out_dir / "unmapped_gdpr.json").write_text(json.dumps(unmapped, indent=2, ensure_ascii=False))

    coverage = len(matched) / len(topk_data) if topk_data else 0.0
    print(f"\n{'─' * 50}")
    print(f"Matched:  {len(matched)}")
    print(f"Unmapped: {len(unmapped)}")
    print(f"Coverage: {coverage:.1%}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
