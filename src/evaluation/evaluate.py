#!/usr/bin/env python3
"""
Evaluate pipeline classification output against gold-standard deviation manifests.

Matching is article-level:
  - constraint_coverage : TP if any unmapped GDPR constraint belongs to an affected article
  - other types         : TP if any matched pair has an affected article AND the correct type

Precision is approximate: the gold standard contains 18 *introduced* deviations per
use case (3 per type × 6 types).  FPs include genuine policy deviations not in the gold standard.
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
    },
    "zalando": {
        "classified": DATA_DIR / "classification" / "zalando_hybrid_classified.json",
        "manifest": GOLD_DIR / "zalando_deviation_manifest.json",
    },
    "traderepublic": {
        "classified": DATA_DIR / "classification" / "traderepublic_hybrid_classified.json",
        "manifest": GOLD_DIR / "traderepublic_deviation_manifest.json",
    },
}


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

    # article → set of deviation types predicted by pipeline (non-none, non-error)
    article_predictions: dict[int, set[str]] = defaultdict(set)
    for p in pairs:
        dtype = p.get("deviation_type", "none")
        if dtype not in ("none", "parse_error"):
            article_predictions[p["gdpr_article"]].add(dtype)

    # articles present in unmapped (constraint_coverage predictions)
    unmapped_articles: set[int] = set()
    for u in unmapped:
        art = article_map.get(u["gdpr_id"])
        if art is not None:
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
        "\nNOTE: Precision is approximate — the gold standard covers 18 introduced\n"
        "deviations per use case (3 per type × 6 types). Pipeline FPs may include genuine policy issues.\n"
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
