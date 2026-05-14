#!/usr/bin/env python3
"""
Hybrid policy extraction — Step 1: contextual LLM extraction.

For each sentence in the policy, provides ±WINDOW surrounding sentences as
context and asks Qwen3.5:9b (via Ollama) whether the sentence states a
data-protection obligation or right.

Output JSON format (same schema as policy_constraints.json):
  [{"id": "pol_001", "source": "policy", "section": "...", "text": "..."}]

Usage:
  python3 src/preprocessing/hybrid_extract.py \
      --policy data/policy/hetzner_policy_modified.txt \
      --output data/constraints/hetzner_hybrid_constraints.json [--verbose]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
import spacy

from src.preprocessing.hybrid_prompt import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt

_OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _OLLAMA_BASE + "/api/chat"
MODEL = "qwen3.5:9b"
WINDOW = 5

DEVIATION_BLOCK_RE = re.compile(r"\[DEVIATION.*?\]", re.DOTALL)
NUMBERED_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", re.MULTILINE)

_nlp_cache = None


def _get_nlp():
    global _nlp_cache
    if _nlp_cache is None:
        _nlp_cache = spacy.load("en_core_web_sm")
        _nlp_cache.max_length = 2_000_000
    return _nlp_cache


def _strip_annotations(text: str) -> str:
    return DEVIATION_BLOCK_RE.sub("", text)


def filter_sentence(sentence: str) -> bool:
    """Return False for sentences that are never data-protection obligations."""
    s = sentence.strip()
    if s.endswith("?"):
        return False
    if len(s.split()) < 5:
        return False
    if re.match(r"^[=\-]{5,}", s):
        return False
    return True


def get_context_window(
    sentences: list, idx: int, window: int = WINDOW
) -> tuple:
    before = sentences[max(0, idx - window): idx]
    after = sentences[idx + 1: idx + 1 + window]
    return before, after


def _parse_sections(text: str) -> list:
    positions = []
    for m in NUMBERED_SECTION_RE.finditer(text):
        positions.append((m.start(), m.group(0).strip()))
    return positions


def _get_section(char_pos: int, section_positions: list) -> str:
    section = ""
    for pos, name in section_positions:
        if pos <= char_pos:
            section = name
        else:
            break
    return section


def check_ollama() -> None:
    try:
        requests.get(_OLLAMA_BASE + "/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR: Ollama unreachable at {_OLLAMA_BASE} — {e}")


def call_ollama(sentence: str, context_before: list, context_after: list) -> bool:
    user_msg = "/no_think\n" + build_extraction_prompt(sentence, context_before, context_after)
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "options": {"temperature": 0},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["message"]["content"].strip().lower()
    return content.startswith("yes")


def extract_policy_obligations(policy_text: str, verbose: bool = False) -> list:
    clean = _strip_annotations(policy_text)
    clean = re.sub(r"^[=\-]{3,}\s*$", "", clean, flags=re.MULTILINE)

    section_positions = _parse_sections(clean)

    nlp = _get_nlp()
    doc = nlp(clean)

    sentences = []  # list of (text, start_char)
    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            sentences.append((s, sent.start_char))

    constraints = []
    seen = set()
    texts_only = [s for s, _ in sentences]

    for i, (sentence, start_char) in enumerate(sentences):
        if sentence in seen or not filter_sentence(sentence):
            continue

        before, after = get_context_window(texts_only, i)
        is_obligation = call_ollama(sentence, before, after)

        if verbose:
            mark = "✓" if is_obligation else "✗"
            print(f"  [{i + 1}/{len(sentences)}] {mark} {sentence[:80]}")

        if is_obligation:
            seen.add(sentence)
            section = _get_section(start_char, section_positions)
            constraints.append({
                "id": f"pol_{len(constraints) + 1:03d}",
                "source": "policy",
                "section": section,
                "text": sentence,
            })

    return constraints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid policy extraction: contextual LLM-based obligation detection"
    )
    parser.add_argument("--policy", type=Path, required=True,
                        help="Path to policy text file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON file for extracted constraints")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sentence classification results")
    args = parser.parse_args()

    check_ollama()

    policy_text = args.policy.read_text(encoding="utf-8")
    print(f"Extracting from {args.policy.name} …")
    print(f"Model: {MODEL}  |  Context window: ±{WINDOW} sentences")

    constraints = extract_policy_obligations(policy_text, verbose=args.verbose)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(constraints, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {len(constraints)} constraints saved to {args.output}")


if __name__ == "__main__":
    main()
