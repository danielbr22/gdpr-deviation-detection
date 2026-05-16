#!/usr/bin/env python3
"""
Evaluate pipeline classification output against gold-standard deviation manifests.

Matching is article-level:
  - constraint_coverage : TP if any unmapped GDPR constraint belongs to an affected article
  - other types         : TP if any matched pair has an affected article AND the correct type

Precision is approximate: the gold standard covers 17 introduced deviations per use case.
FPs may include genuine policy issues not in the gold standard.

Two precision improvements are applied before computing predictions:

1. Stricter in-scope filter (MIN_ORIGINAL_PAIRS): GDPR articles are only considered
   in-scope if the original policy had at least this many substantive matches.
   Articles with only 1 spurious match are structurally irrelevant and excluded from
   constraint_coverage predictions. Requires Phase 0 output to exist.

2. Pair deduplication (DEDUP_MAX_ARTICLES_PER_SENTENCE): when the same policy sentence
   is classified as the same deviation type against many GDPR articles, only the top-K
   articles by cosine similarity are credited. One policy sentence is evidence for at most
   K article-level deviations of the same type.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
GOLD_DIR = ROOT / "gold_standard"
EVAL_DIR = DATA_DIR / "evaluation"

# Minimum number of matched pairs an article must have in the ORIGINAL policy to be
# considered "in-scope" for constraint_coverage evaluation. Articles with fewer matches
# are structurally irrelevant to the company policy and excluded to reduce FPs.
MIN_ORIGINAL_PAIRS: int = 2

# Maximum number of distinct GDPR articles a single policy sentence can be credited for
# with the same deviation type. Prevents cross-article FP inflation when one modified
# sentence is retrieved against many topically-adjacent GDPR articles.
DEDUP_MAX_ARTICLES_PER_SENTENCE: int = 2

DEVIATION_TYPES = [
    "constraint_coverage",
    "severity",
    "execution_style",
    "negation",
    "responsibility",
    "data",
]

USE_CASES: dict[str, dict[str, Path]] = {
    "hetzner": {
        "classified": DATA_DIR / "classification" / "hetzner_hybrid_classified.json",
        "manifest": GOLD_DIR / "deviation_manifest.json",
        "original_matched": DATA_DIR / "retrieval" / "hetzner_original" / "matched_pairs.json",
    },
    "zalando": {
        "classified": DATA_DIR / "classification" / "zalando_hybrid_classified.json",
        "manifest": GOLD_DIR / "zalando_deviation_manifest.json",
        "original_matched": DATA_DIR / "retrieval" / "zalando_original" / "matched_pairs.json",
    },
    "traderepublic": {
        "classified": DATA_DIR / "classification" / "traderepublic_hybrid_classified.json",
        "manifest": GOLD_DIR / "traderepublic_deviation_manifest.json",
        "original_matched": DATA_DIR / "retrieval" / "traderepublic_original" / "matched_pairs.json",
    },
}


def deduplicate_pairs(pairs: list[dict], max_articles: int) -> list[dict]:
    """Limit cross-article FP inflation: for each (policy_text, deviation_type) group,
    keep only the top-max_articles GDPR articles by cosine similarity.

    Rationale: a single modified policy sentence is evidence for at most a few GDPR
    articles' deviations. When embedding retrieval floods many topically-adjacent
    articles with the same sentence, only the highest-similarity articles are credited.
    Non-deviation pairs (none, parse_error) are always kept unchanged.
    """
    KEEP_TYPES = {"none", "parse_error"}
    kept = [p for p in pairs if p.get("deviation_type", "none") in KEEP_TYPES]
    deviation_pairs = [p for p in pairs if p.get("deviation_type", "none") not in KEEP_TYPES]

    # Group by (truncated policy text, deviation_type)
    group: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in deviation_pairs:
        key = (p.get("policy_text", "")[:150], p.get("deviation_type"))
        art = p.get("gdpr_article", -1)
        group[key][art].append(p)

    for (_, _), art_map in group.items():
        # For each article, take the pair with the highest similarity
        art_best: list[tuple[float, int, dict]] = []
        for art, art_pairs in art_map.items():
            best = max(art_pairs, key=lambda x: x.get("similarity", 0.0) or 0.0)
            art_best.append((best.get("similarity", 0.0) or 0.0, art, best))

        # Keep only top max_articles unique articles; mark the rest as none
        art_best.sort(key=lambda x: -x[0])
        for i, (_, art, best_pair) in enumerate(art_best):
            if i < max_articles:
                kept.append(best_pair)
            else:
                # Preserve pair but downgrade to none so it doesn't inflate predictions
                downgraded = {**best_pair, "deviation_type": "none",
                              "reasoning": f"[dedup] downgraded from {best_pair['deviation_type']} — "
                                           f"same sentence credited to {max_articles} higher-similarity articles"}
                kept.append(downgraded)

    return kept


def _parse_article(article_str: str) -> int:
    m = re.search(r"\d+", article_str)
    return int(m.group()) if m else -1


def load_gdpr_article_map() -> dict[str, int]:
    with open(DATA_DIR / "constraints" / "gdpr_constraints.json") as f:
        return {c["id"]: c["article"] for c in json.load(f)}


def evaluate_use_case(use_case: str, article_map: dict[str, int]) -> dict:
    cfg = USE_CASES[use_case]

    with open(cfg["classified"]) as f:
        classified = json.load(f)
    with open(cfg["manifest"]) as f:
        gold = json.load(f)

    pairs = classified["pairs"]
    unmapped = classified["unmapped"]

    # Load in-scope articles from original policy run (Phase 0 of run.sh).
    # Only articles with >= MIN_ORIGINAL_PAIRS substantive matches in the original policy
    # are considered in-scope. Articles with fewer matches are structurally irrelevant
    # to the company policy and excluded from constraint_coverage to reduce FPs.
    in_scope_articles: set[int] | None = None
    orig_path: Path = cfg["original_matched"]
    if orig_path.exists():
        with open(orig_path) as f:
            orig_matched = json.load(f)
        art_pair_counts: dict[int, int] = defaultdict(int)
        for p in orig_matched:
            art = p.get("gdpr_article")
            if art is not None:
                art_pair_counts[art] += 1
        in_scope_articles = {
            art for art, count in art_pair_counts.items() if count >= MIN_ORIGINAL_PAIRS
        }

    # Deduplication: limit cross-article FP inflation from the same policy sentence.
    # When one modified sentence is retrieved against many topically-adjacent GDPR articles,
    # only the DEDUP_MAX_ARTICLES_PER_SENTENCE highest-similarity articles are credited.
    pairs = deduplicate_pairs(pairs, DEDUP_MAX_ARTICLES_PER_SENTENCE)

    # article → set of deviation types predicted by pipeline (non-none, non-error)
    article_predictions: dict[int, set[str]] = defaultdict(set)
    for p in pairs:
        dtype = p.get("deviation_type", "none")
        if dtype not in ("none", "parse_error"):
            article_predictions[p["gdpr_article"]].add(dtype)

    # articles present in unmapped (constraint_coverage predictions), filtered to in-scope
    unmapped_articles: set[int] = set()
    for u in unmapped:
        art = article_map.get(u["gdpr_id"])
        if art is not None and (in_scope_articles is None or art in in_scope_articles):
            unmapped_articles.add(art)

    # --- evaluate each gold deviation ---
    gold_results: list[dict] = []
    for g in gold:
        articles = {_parse_article(a) for a in g["gdpr_articles_affected"]}
        dtype = g["deviation_type"]

        if dtype == "constraint_coverage":
            detected = bool(articles & unmapped_articles)
        else:
            detected = any(dtype in article_predictions.get(a, set()) for a in articles)

        gold_results.append(
            {
                "id": g["id"],
                "deviation_type": dtype,
                "articles": sorted(articles),
                "detected": detected,
            }
        )

    # --- per-type TP / FP / FN ---
    # gold_articles[type] = set of article numbers the gold standard names for that type
    gold_articles: dict[str, set[int]] = defaultdict(set)
    for g in gold:
        for a in g["gdpr_articles_affected"]:
            gold_articles[g["deviation_type"]].add(_parse_article(a))

    type_metrics: dict[str, dict] = {}
    for dtype in DEVIATION_TYPES:
        gold_set = gold_articles.get(dtype, set())
        tp = sum(1 for r in gold_results if r["deviation_type"] == dtype and r["detected"])
        fn = sum(1 for r in gold_results if r["deviation_type"] == dtype and not r["detected"])

        if dtype == "constraint_coverage":
            predicted_set = unmapped_articles
        else:
            predicted_set = {a for a, types in article_predictions.items() if dtype in types}

        fp = len(predicted_set - gold_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        type_metrics[dtype] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    total_tp = sum(r["detected"] for r in gold_results)
    total_fn = len(gold_results) - total_tp
    total_fp = sum(m["fp"] for m in type_metrics.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )

    return {
        "use_case": use_case,
        "gold_results": gold_results,
        "type_metrics": type_metrics,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_p, 3),
            "recall": round(overall_r, 3),
            "f1": round(overall_f1, 3),
        },
    }


def aggregate(results: list[dict]) -> dict:
    agg_types: dict[str, dict] = {}
    for dtype in DEVIATION_TYPES:
        tp = sum(r["type_metrics"].get(dtype, {}).get("tp", 0) for r in results)
        fp = sum(r["type_metrics"].get(dtype, {}).get("fp", 0) for r in results)
        fn = sum(r["type_metrics"].get(dtype, {}).get("fn", 0) for r in results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        agg_types[dtype] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    all_tp = sum(r["overall"]["tp"] for r in results)
    all_fp = sum(r["overall"]["fp"] for r in results)
    all_fn = sum(r["overall"]["fn"] for r in results)
    overall_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )
    return {
        "type_metrics": agg_types,
        "overall": {
            "tp": all_tp,
            "fp": all_fp,
            "fn": all_fn,
            "precision": round(overall_p, 3),
            "recall": round(overall_r, 3),
            "f1": round(overall_f1, 3),
        },
    }


def print_report(results: list[dict], agg: dict) -> None:
    W = 70

    print("\n" + "=" * W)
    print("DEVIATION DETECTION EVALUATION")
    print("=" * W)
    print(
        "\nNOTE: Precision is approximate — the gold standard covers 17 introduced deviations\n"
        "per use case (Art. 77 excluded — outside GDPR Art. 5–43 scope). constraint_coverage unmapped set is filtered\n"
        "to GDPR articles substantively covered by the original policy (Phase 0 scope detection).\n"
        "FPs may still include genuine policy issues not in the gold standard.\n"
    )

    for r in results:
        print(f"{'─' * W}")
        print(f"USE CASE: {r['use_case'].upper()}")
        print(f"{'─' * W}")
        print("Gold deviations:")
        for gr in r["gold_results"]:
            mark = "✓" if gr["detected"] else "✗"
            arts = ", ".join(f"Art. {a}" for a in gr["articles"])
            print(f"  {mark}  {gr['id']}  {gr['deviation_type']:<22}  ({arts})")

        print()
        print(f"  {'Type':<22} {'P':>7} {'R':>7} {'F1':>7}   TP / FP / FN")
        print(f"  {'─' * 22}  {'─' * 7}  {'─' * 7}  {'─' * 7}   {'─' * 14}")
        for dtype in DEVIATION_TYPES:
            m = r["type_metrics"][dtype]
            if m["tp"] + m["fp"] + m["fn"] == 0:
                continue
            print(
                f"  {dtype:<22}  {m['precision']:>7.3f}  {m['recall']:>7.3f}  {m['f1']:>7.3f}"
                f"   {m['tp']} / {m['fp']} / {m['fn']}"
            )
        o = r["overall"]
        print(
            f"\n  {'OVERALL':<22}  {o['precision']:>7.3f}  {o['recall']:>7.3f}  {o['f1']:>7.3f}"
            f"   {o['tp']} / {o['fp']} / {o['fn']}"
        )
        print()

    print(f"{'=' * W}")
    print("AGGREGATE (all use cases combined)")
    print(f"{'=' * W}")
    print(f"\n  {'Type':<22} {'P':>7} {'R':>7} {'F1':>7}   TP / FP / FN")
    print(f"  {'─' * 22}  {'─' * 7}  {'─' * 7}  {'─' * 7}   {'─' * 14}")
    for dtype in DEVIATION_TYPES:
        m = agg["type_metrics"][dtype]
        if m["tp"] + m["fp"] + m["fn"] == 0:
            continue
        print(
            f"  {dtype:<22}  {m['precision']:>7.3f}  {m['recall']:>7.3f}  {m['f1']:>7.3f}"
            f"   {m['tp']} / {m['fp']} / {m['fn']}"
        )
    o = agg["overall"]
    print(
        f"\n  {'OVERALL':<22}  {o['precision']:>7.3f}  {o['recall']:>7.3f}  {o['f1']:>7.3f}"
        f"   {o['tp']} / {o['fp']} / {o['fn']}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deviation detection pipeline")
    parser.add_argument(
        "--use-case",
        choices=[*USE_CASES, "all"],
        default="all",
        help="Which use case(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "results.json",
        help="Path for JSON output",
    )
    args = parser.parse_args()

    article_map = load_gdpr_article_map()
    cases = list(USE_CASES) if args.use_case == "all" else [args.use_case]
    results = [evaluate_use_case(uc, article_map) for uc in cases]
    agg = aggregate(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"results": results, "aggregate": agg}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results saved → {args.output}")

    print_report(results, agg)


if __name__ == "__main__":
    main()
