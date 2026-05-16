JUDGE_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert. Given a GDPR constraint and a list of "
    "candidate policy passages, decide which candidate (if any) has a substantive "
    "connection to the GDPR constraint — meaning the policy passage DIRECTLY and "
    "SPECIFICALLY addresses the primary obligation stated in this constraint.\n\n"
    "STRICT MATCHING RULES — choose 0 (no match) if any of these apply:\n"
    "  1. The passage is more specifically about a DIFFERENT GDPR right or obligation "
    "     than the one in this constraint (e.g., the constraint is about the right to "
    "     restrict processing, but the passage is about the right to erasure).\n"
    "  2. The passage only addresses this topic indirectly or as a side effect.\n"
    "  3. The passage merely enumerates rights by article number without explaining "
    "     what the right entails or how to exercise it.\n"
    "  4. The passage is from a different processing context (e.g., a constraint about "
    "     automated decision-making matched to a passage about newsletter consent).\n\n"
    "Choose the candidate ONLY if its primary topic directly matches the constraint's "
    "primary obligation. When in doubt, choose 0.\n\n"
    "Respond with valid JSON only. No markdown."
)


def build_judge_prompt(
    gdpr_text: str,
    gdpr_article: int,
    candidates: list,
) -> str:
    candidate_lines = "\n".join(
        f'[{i + 1}] "{c["policy_text"]}"'
        for i, c in enumerate(candidates)
    )
    return (
        f"GDPR constraint (Art. {gdpr_article}):\n"
        f'"{gdpr_text}"\n\n'
        f"Candidate policy passages:\n"
        f"{candidate_lines}\n\n"
        "Which candidate (if any) has a substantive connection to the GDPR constraint?\n"
        "Reply with the candidate number (1, 2, ...) if one matches, or 0 if none do.\n"
        "Provide a one-sentence reasoning.\n"
        '{"match": <number>, "reasoning": "<one sentence>"}'
    )
