STAGE1_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert performing a binary compliance check.\n\n"
    "You will be given a GDPR regulatory constraint and a company policy sentence that "
    "was retrieved as a potential match. Your task is to determine whether the policy "
    "sentence contains a genuine compliance deviation relative to the GDPR constraint.\n\n"
    "A genuine deviation means the policy differs from the GDPR requirement in a "
    "concrete, specific way — it says something different, restricts a right, assigns "
    "wrong responsibility, uses a wrong procedure, or contradicts the regulation.\n\n"
    "NOT a deviation: the policy sentence addresses the same topic adequately, even if "
    "phrased differently. Most retrieved pairs are compliant — default to 'no' unless "
    "you can point to a specific, concrete conflict.\n\n"
    "You MUST quote the exact text from both documents that creates the conflict before "
    "deciding. If you cannot quote a specific conflict, the answer is 'no'.\n\n"
    "Respond with valid JSON only. No markdown, no explanation outside the JSON."
)

STAGE2_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert classifying a confirmed compliance deviation.\n\n"
    "A previous check confirmed that the company policy deviates from the GDPR "
    "constraint. Your task is to classify the deviation type using the RCASR taxonomy:\n\n"
    "- responsibility: Same task and data, but the wrong party is responsible (WHO).\n"
    "- execution_style: Same responsibility and data, but incorrect procedure (HOW).\n"
    "- data: Same responsibility and task, but incorrect data scope (WHAT data).\n"
    "- negation: The policy directly negates or contradicts the GDPR constraint.\n"
    "- severity: Same responsibility, task and data, but the policy imposes a stricter "
    "standard than the GDPR requires (over-compliance, e.g. shorter deadlines, narrower "
    "retention periods, or additional procedural burdens beyond what the regulation mandates).\n\n"
    "Respond with valid JSON only. No markdown, no explanation outside the JSON."
)


def build_stage1_prompt(gdpr_text: str, gdpr_article: int, policy_text: str) -> str:
    return (
        f"GDPR constraint (Art. {gdpr_article}):\n"
        f'"{gdpr_text}"\n\n'
        f"Company policy sentence:\n"
        f'"{policy_text}"\n\n'
        "Step 1 — Quote the specific GDPR obligation:\n"
        "Step 2 — Quote the specific policy text that conflicts with it (if any):\n"
        "Step 3 — Is there a genuine compliance deviation?\n\n"
        '{"gdpr_quote": "<exact quote>", "policy_quote": "<exact quote or null>", '
        '"has_deviation": true/false, "reasoning": "<one sentence>"}'
    )


def build_stage2_prompt(gdpr_text: str, gdpr_article: int, policy_text: str) -> str:
    return (
        f"GDPR constraint (Art. {gdpr_article}):\n"
        f'"{gdpr_text}"\n\n'
        f"Company policy sentence:\n"
        f'"{policy_text}"\n\n'
        "This pair has a confirmed deviation. Classify the deviation type.\n"
        '{"deviation_type": "<type>", "reasoning": "<one sentence>"}'
    )
