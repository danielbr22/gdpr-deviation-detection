import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import requests

from src.classification.prompt import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    build_stage1_prompt,
    build_stage2_prompt,
)

_OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _OLLAMA_BASE + "/api/chat"
MODEL = "qwen3.5:9b"
VALID_TYPES = {"none", "responsibility", "execution_style", "data", "negation", "severity"}


def _call_ollama(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "/no_think\n" + user},
        ],
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


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


def check_ollama() -> None:
    try:
        requests.get(_OLLAMA_BASE + "/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR: Ollama unreachable at {_OLLAMA_BASE} — {e}")


def classify_pair(pair: dict) -> dict:
    gdpr_text = pair["gdpr_text"]
    gdpr_article = pair.get("gdpr_article", 0)
    policy_text = pair["policy_text"]

    # Stage 1: binary deviation gate
    last_err = "unknown error"
    stage1 = None
    for _ in range(2):
        try:
            raw = _call_ollama(
                STAGE1_SYSTEM_PROMPT,
                build_stage1_prompt(gdpr_text, gdpr_article, policy_text),
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
            raw = _call_ollama(
                STAGE2_SYSTEM_PROMPT,
                build_stage2_prompt(gdpr_text, gdpr_article, policy_text),
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


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        prev = json.load(f)
    return {(p["gdpr_id"], p["policy_id"]): p for p in prev.get("pairs", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GDPR/policy matched pairs via Ollama")
    parser.add_argument("--matched-pairs", required=True)
    parser.add_argument("--unmapped", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--limit", type=int, help="Only classify first N pairs (for testing)")
    args = parser.parse_args()

    check_ollama()

    with open(args.matched_pairs) as f:
        pairs = json.load(f)
    with open(args.unmapped) as f:
        unmapped = json.load(f)

    pairs = filter_pairs(pairs)
    if args.limit:
        pairs = pairs[: args.limit]

    output_path = Path(args.output)
    existing = load_existing(output_path)

    classified_pairs = []
    for i, pair in enumerate(pairs, 1):
        key = (pair["gdpr_id"], pair["policy_id"])
        if key in existing:
            print(f"pair {i}/{len(pairs)} — skipping (already done)")
            classified_pairs.append(existing[key])
            continue

        print(f"pair {i}/{len(pairs)} — {pair['gdpr_id']} × {pair['policy_id']}", flush=True)
        result = classify_pair(pair)
        classified_pairs.append(
            {
                "gdpr_id": pair["gdpr_id"],
                "policy_id": pair["policy_id"],
                "gdpr_article": pair.get("gdpr_article"),
                "gdpr_text": pair["gdpr_text"],
                "policy_text": pair["policy_text"],
                "similarity": pair.get("similarity"),
                "deviation_type": result["deviation_type"],
                "reasoning": result.get("reasoning", ""),
            }
        )

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
        "model": MODEL,
        "pairs": classified_pairs,
        "unmapped": classified_unmapped,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {len(classified_pairs)} pairs classified, {len(classified_unmapped)} unmapped auto-labeled.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
