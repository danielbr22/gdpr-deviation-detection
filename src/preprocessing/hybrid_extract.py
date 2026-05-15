#!/usr/bin/env python3
"""
Hybrid policy extraction — Step 1: contextual LLM extraction.

For each sentence in the policy, provides ±WINDOW surrounding sentences as
context and asks the LLM whether the sentence states a data-protection
obligation or right.

Provider is selected via LLM_PROVIDER env var (ollama default, openai for
external). With concurrency > 1 (OpenAI) all sentences are classified in
parallel.

Output JSON format:
  [{"id": "pol_001", "source": "policy", "section": "...", "text": "..."}]

Usage:
  python3 src/preprocessing/hybrid_extract.py \
      --policy data/policy/hetzner_policy_modified.txt \
      --output data/constraints/hetzner_hybrid_constraints.json [--verbose]
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import spacy

from src.preprocessing.hybrid_prompt import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.utils import llm_client

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
    s = sentence.strip()
    if s.endswith("?"):
        return False
    if len(s.split()) < 5:
        return False
    if re.match(r"^[=\-]{5,}", s):
        return False
    return True


def get_context_window(sentences: list, idx: int, window: int = WINDOW) -> tuple:
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


async def extract_policy_obligations_async(policy_text: str, verbose: bool = False) -> list:
    clean = _strip_annotations(policy_text)
    clean = re.sub(r"^[=\-]{3,}\s*$", "", clean, flags=re.MULTILINE)

    section_positions = _parse_sections(clean)

    nlp = _get_nlp()
    doc = nlp(clean)

    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            sentences.append((s, sent.start_char))

    texts_only = [s for s, _ in sentences]

    # Deduplicate + filter before sending any requests
    seen_text: set = set()
    to_classify = []
    for i, (sentence, start_char) in enumerate(sentences):
        if sentence in seen_text or not filter_sentence(sentence):
            continue
        seen_text.add(sentence)
        before, after = get_context_window(texts_only, i)
        to_classify.append((i, sentence, start_char, before, after))

    sem = asyncio.Semaphore(llm_client.concurrency())

    async def classify_one(original_idx: int, sentence: str, start_char: int, before: list, after: list):
        async with sem:
            content = await llm_client.call_async(
                EXTRACTION_SYSTEM_PROMPT,
                build_extraction_prompt(sentence, before, after),
                timeout=60,
            )
        is_obligation = content.strip().lower().startswith("yes")
        if verbose:
            mark = "✓" if is_obligation else "✗"
            print(f"  [{original_idx + 1}/{len(sentences)}] {mark} {sentence[:80]}")
        return original_idx, sentence, start_char, is_obligation

    results = await asyncio.gather(*[classify_one(*item) for item in to_classify])

    # Sort by original sentence order before assigning IDs
    results = sorted(results, key=lambda x: x[0])

    constraints = []
    for _, sentence, start_char, is_obligation in results:
        if is_obligation:
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
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    llm_client.check_provider()

    policy_text = args.policy.read_text(encoding="utf-8")
    print(f"Extracting from {args.policy.name} …")
    print(f"  Context window: ±{WINDOW} sentences")

    constraints = asyncio.run(extract_policy_obligations_async(policy_text, verbose=args.verbose))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(constraints, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {len(constraints)} constraints saved to {args.output}")


if __name__ == "__main__":
    main()
