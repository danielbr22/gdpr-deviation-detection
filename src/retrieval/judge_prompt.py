JUDGE_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert. Given a GDPR constraint and a list of "
    "candidate policy passages, decide which candidate (if any) has a substantive "
    "connection to the GDPR constraint — meaning the policy passage directly "
    "addresses, implements, or is materially relevant to the GDPR requirement. "
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
