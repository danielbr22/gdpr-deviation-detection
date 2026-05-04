SYSTEM_PROMPT = (
    "You are a GDPR compliance expert. Your task is to classify the relationship "
    "between a GDPR regulatory constraint and a company policy sentence.\n\n"
    "The RCASR deviation taxonomy defines five deviation types:\n"
    "- none: The policy sentence adequately addresses the GDPR constraint. No deviation.\n"
    "- responsibility: Same task and data, but the wrong party is responsible (WHO).\n"
    "- execution_style: Same responsibility and data, but incorrect procedure (HOW).\n"
    "- data: Same responsibility and task, but incorrect data scope (WHAT data).\n"
    "- negation: The policy directly negates or contradicts the GDPR constraint.\n\n"
    "Respond with valid JSON only. No markdown, no explanation outside the JSON."
)


def build_user_prompt(gdpr_text: str, gdpr_article: int, policy_text: str) -> str:
    return (
        f"GDPR constraint (Art. {gdpr_article}):\n"
        f'"{gdpr_text}"\n\n'
        f"Company policy sentence:\n"
        f'"{policy_text}"\n\n'
        "Classify the deviation type and provide a one-sentence reasoning.\n"
        '{"deviation_type": "<type>", "reasoning": "<one sentence>"}'
    )
