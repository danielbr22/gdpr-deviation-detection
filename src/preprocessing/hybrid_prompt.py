EXTRACTION_SYSTEM_PROMPT = (
    "You are a GDPR compliance analyst. Your task is to decide whether a sentence "
    "from a company privacy policy states a data-protection obligation or right — "
    "a commitment the company makes, something it must do, or a right it grants to "
    "data subjects. Respond with exactly one word: 'yes' or 'no'. No explanation."
)


def build_extraction_prompt(
    sentence: str,
    context_before: list[str],
    context_after: list[str],
) -> str:
    before_text = " ".join(context_before) if context_before else "(start of section)"
    after_text = " ".join(context_after) if context_after else "(end of section)"
    return (
        f"Surrounding context:\n"
        f"...{before_text}\n\n"
        f"Sentence to evaluate:\n"
        f'"{sentence}"\n\n'
        f"...{after_text}\n\n"
        "Does this sentence state a data-protection obligation or right? "
        "Answer yes or no."
    )
