EXTRACTION_SYSTEM_PROMPT = (
    "You are a GDPR compliance analyst. Decide whether a sentence from a company "
    "privacy policy expresses a normative data-protection commitment — a specific "
    "obligation the company binds itself to, or a concrete right it grants data subjects.\n\n"
    "Answer YES only if the sentence:\n"
    "  - States what the company will or will not do with personal data (retention, sharing, deletion, security)\n"
    "  - Grants a specific right to data subjects (access, erasure, portability, objection, etc.)\n"
    "  - Specifies a legal basis for processing (Art. 6 / Art. 9 grounds)\n"
    "  - Commits to a safeguard or procedural guarantee (DPA concluded, data stays in EU, etc.)\n\n"
    "Answer NO if the sentence:\n"
    "  - Merely describes what data is collected or where it comes from\n"
    "  - Describes an operational fact or internal procedure without a normative commitment\n"
    "  - Is a list label, section heading, or incomplete fragment\n"
    "  - Contains only contact details, links, or references to other documents\n\n"
    "Respond with exactly one word: 'yes' or 'no'. No explanation."
)


def build_extraction_prompt(
    sentence: str,
    context_before: list[str],
    context_after: list[str],
) -> str:
    before_text = " ".join(context_before) if context_before else "(start of section)"
    after_text = " ".join(context_after) if context_after else "(end of section)"
    return (
        f"Surrounding context (before):\n"
        f"...{before_text}\n\n"
        f"Sentence to evaluate:\n"
        f'"{sentence}"\n\n'
        f"Surrounding context (after):\n"
        f"...{after_text}\n\n"
        "Does this sentence express a normative data-protection commitment? "
        "Answer yes or no."
    )
