EXTRACTION_SYSTEM_PROMPT = (
    "You are a GDPR compliance analyst. Decide whether a sentence from a company "
    "privacy policy addresses a data-protection obligation or right — meaning it makes "
    "a substantive claim about how the company handles personal data that could be "
    "compared against a GDPR requirement.\n\n"
    "Respond 'yes' if the sentence addresses any of:\n"
    "  - Retention or deletion of personal data\n"
    "  - Sharing, transfer, or disclosure of data to third parties\n"
    "  - Legal basis for processing (Art. 6 / Art. 9 grounds, consent, legitimate interest)\n"
    "  - Data subject rights (access, rectification, erasure, portability, objection, etc.)\n"
    "  - Security measures or data protection safeguards\n"
    "  - Restrictions on processing purpose or data minimisation\n"
    "  - DPA agreements, international transfer safeguards\n"
    "  - WHO is responsible for handling data subject rights requests or complaints "
    "(e.g. naming a contact point, DPO designation, processor/controller role assignment)\n"
    "  - HOW data subjects must exercise their rights (channel, procedure, contact person)\n"
    "  - Role of a third-party service provider (processor vs. independent controller)\n\n"
    "Respond 'no' if the sentence:\n"
    "  - Only describes what categories of data are collected or where data comes from\n"
    "  - Is a list label, section heading, or sentence fragment\n"
    "  - Contains only a URL, email address, or postal address with no normative content\n"
    "  - Makes only a generic claim with no specific data-protection content "
    "(e.g. 'we comply with GDPR', 'we take privacy seriously')\n\n"
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
