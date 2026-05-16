import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.classification.prompt import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    build_stage1_prompt,
    build_stage2_prompt,
)
from src.utils import llm_client

VALID_TYPES = {"none", "responsibility", "execution_style", "data", "negation", "severity"}

# Maximum number of distinct GDPR articles a single policy sentence can be credited for
# with the same deviation type. Prevents cross-article FP inflation when one modified
# sentence is retrieved against many topically-adjacent articles.
DEDUP_MAX_ARTICLES: int = 2


def parse_response(content: str) -> Optional[dict]:
    """Parse a classification response that may include 'none' as a valid type."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if data.get("deviation_type") not in VALID_TYPES:
        return None
    if not data.get("reasoning"):
        return None
    return data


def _parse_stage1(content: str) -> Optional[dict]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("has_deviation"), bool):
        return None
    return data


def _parse_stage2(content: str) -> Optional[dict]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if data.get("deviation_type") not in VALID_TYPES - {"none"}:
        return None
    if not data.get("reasoning"):
        return None
    return data


def filter_pairs(pairs: list) -> list:
    return [p for p in pairs if p.get("policy_id") != "pol_001"]


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Limit cross-article FP inflation from the same policy sentence.

    For each (policy_text, deviation_type) group, keep only the top-DEDUP_MAX_ARTICLES
    GDPR articles by cosine similarity. Non-deviation pairs are always preserved.

    A single modified policy sentence is evidence for at most DEDUP_MAX_ARTICLES
    article-level deviations of the same type. Cross-article flooding — where one
    sentence is retrieved against many topically-adjacent GDPR articles — produces
    many false positives. Deduplication credits only the most relevant articles.
    """
    keep_types = {"none", "parse_error"}
    kept = [p for p in pairs if p.get("deviation_type", "none") in keep_types]
    deviation_pairs = [p for p in pairs if p.get("deviation_type", "none") not in keep_types]

    group: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in deviation_pairs:
        key = (p.get("policy_text", "")[:150], p.get("deviation_type"))
        art = p.get("gdpr_article", -1)
        group[key][art].append(p)

    for (_, dtype), art_map in group.items():
        art_best: list[tuple[float, int, dict]] = []
        for art, art_pairs in art_map.items():
            best = max(art_pairs, key=lambda x: x.get("similarity", 0.0) or 0.0)
            art_best.append((best.get("similarity", 0.0) or 0.0, art, best))

        art_best.sort(key=lambda x: -x[0])
        for i, (_, art, best_pair) in enumerate(art_best):
            if i < DEDUP_MAX_ARTICLES:
                kept.append(best_pair)
            else:
                downgraded = {**best_pair, "deviation_type": "none",
                              "reasoning": f"[dedup] downgraded from {dtype} — "
                                           f"same sentence already credited to "
                                           f"{DEDUP_MAX_ARTICLES} higher-similarity articles"}
                kept.append(downgraded)

    return kept


async def _classify_one(sem: asyncio.Semaphore, pair: dict) -> dict:
    gdpr_text = pair["gdpr_text"]
    gdpr_article = pair.get("gdpr_article", 0)
    policy_text = pair["policy_text"]

    async with sem:
        # Stage 1: binary deviation gate
        stage1 = None
        last_err = "unknown error"
        for _ in range(2):
            try:
                raw = await llm_client.call_async(
                    STAGE1_SYSTEM_PROMPT,
                    build_stage1_prompt(gdpr_text, gdpr_article, policy_text),
                    json_mode=True,
                    timeout=300,
                )
                stage1 = _parse_stage1(raw)
                if stage1:
                    break
                last_err = f"invalid stage1 response: {raw[:200]}"
            except Exception as e:
                last_err = str(e)

        if stage1 is None:
            return {"deviation_type": "parse_error", "reasoning": f"Stage1 failed: {last_err}"}

        if not stage1["has_deviation"]:
            return {
                "deviation_type": "none",
                "reasoning": stage1.get("reasoning", "No deviation detected by stage 1 gate."),
            }

        # Stage 2: classify the confirmed deviation
        stage2 = None
        for _ in range(2):
            try:
                raw = await llm_client.call_async(
                    STAGE2_SYSTEM_PROMPT,
                    build_stage2_prompt(gdpr_text, gdpr_article, policy_text),
                    json_mode=True,
                    timeout=300,
                )
                stage2 = _parse_stage2(raw)
                if stage2:
                    break
                last_err = f"invalid stage2 response: {raw[:200]}"
            except Exception as e:
                last_err = str(e)

        if stage2 is None:
            return {"deviation_type": "parse_error", "reasoning": f"Stage2 failed: {last_err}"}

        return stage2


async def classify_pairs_async(pairs: list) -> list:
    sem = asyncio.Semaphore(llm_client.concurrency())
    return list(await asyncio.gather(*[_classify_one(sem, p) for p in pairs]))


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        prev = json.load(f)
    return {(p["gdpr_id"], p["policy_id"]): p for p in prev.get("pairs", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GDPR/policy matched pairs")
    parser.add_argument("--matched-pairs", required=True)
    parser.add_argument("--unmapped", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    llm_client.check_provider()

    with open(args.matched_pairs) as f:
        pairs = json.load(f)
    with open(args.unmapped) as f:
        unmapped = json.load(f)

    pairs = filter_pairs(pairs)
    if args.limit:
        pairs = pairs[: args.limit]

    output_path = Path(args.output)
    existing = load_existing(output_path)

    new_pairs = [p for p in pairs if (p["gdpr_id"], p["policy_id"]) not in existing]
    skip_pairs = [existing[(p["gdpr_id"], p["policy_id"])] for p in pairs if (p["gdpr_id"], p["policy_id"]) in existing]

    if skip_pairs:
        print(f"Skipping {len(skip_pairs)} already-classified pairs")

    print(f"Classifying {len(new_pairs)} pairs …")
    results_raw = asyncio.run(classify_pairs_async(new_pairs))

    new_classified = []
    for pair, result in zip(new_pairs, results_raw):
        new_classified.append({
            "gdpr_id": pair["gdpr_id"],
            "policy_id": pair["policy_id"],
            "gdpr_article": pair.get("gdpr_article"),
            "gdpr_text": pair["gdpr_text"],
            "policy_text": pair["policy_text"],
            "similarity": pair.get("similarity"),
            "deviation_type": result["deviation_type"],
            "reasoning": result.get("reasoning", ""),
        })

    classified_pairs = deduplicate_pairs(skip_pairs + new_classified)

    classified_unmapped = [
        {
            "gdpr_id": u["gdpr_id"],
            "gdpr_text": u["gdpr_text"],
            "deviation_type": "constraint_coverage",
            "reasoning": "No policy constraint matched above similarity threshold.",
        }
        for u in unmapped
    ]

    output = {
        "use_case": args.use_case,
        "model": os.environ.get("OPENAI_MODEL", "qwen3.5:9b"),
        "pairs": classified_pairs,
        "unmapped": classified_unmapped,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done: {len(classified_pairs)} pairs classified, {len(classified_unmapped)} unmapped auto-labeled.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
