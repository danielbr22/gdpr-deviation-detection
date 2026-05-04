import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

from src.classification.prompt import SYSTEM_PROMPT, build_user_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
VALID_TYPES = {"none", "responsibility", "execution_style", "data", "negation"}


def parse_response(content: str) -> Optional[dict]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if data.get("deviation_type") not in VALID_TYPES:
        return None
    if not data.get("reasoning"):
        return None
    return data


def filter_pairs(pairs: list) -> list:
    return [p for p in pairs if p.get("policy_id") != "pol_001"]


def check_ollama() -> None:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR: Ollama unreachable at localhost:11434 — {e}")


def call_ollama(gdpr_text: str, gdpr_article: int, policy_text: str) -> str:
    user_msg = "/no_think\n" + build_user_prompt(gdpr_text, gdpr_article, policy_text)
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def classify_pair(pair: dict) -> dict:
    last_err = "unknown error"
    for _ in range(2):
        try:
            content = call_ollama(
                pair["gdpr_text"], pair.get("gdpr_article", 0), pair["policy_text"]
            )
            result = parse_response(content)
            if result:
                return result
            last_err = f"invalid response: {content[:200]}"
        except Exception as e:
            last_err = str(e)
    return {"deviation_type": "parse_error", "reasoning": f"Failed after 2 attempts: {last_err}"}


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
