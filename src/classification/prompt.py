_RCASR_TAXONOMY = """\
Each constraint can be decomposed into three components: (responsibility, task, data).
A deviation occurs when the company policy differs from the GDPR constraint in one of
the following ways:

- responsibility (RCASR7): Responsibility and data match, but the wrong party is named
  as responsible for the task (WHO deviates).
- execution_style (RCASR5): Responsibility and data match, but the procedure for
  carrying out the task deviates — e.g. different channel, extra steps, different
  timeline (HOW deviates).
- data (RCASR8): Responsibility and task match, but the scope of data covered deviates —
  e.g. "account data" instead of "personal data" (WHAT data deviates).
- negation (RCASR6): The policy directly contradicts or negates the GDPR constraint —
  e.g. "may not" where GDPR says "may", or a right is explicitly denied.
- severity (RCASR4): All three components match, but the policy imposes a stricter
  standard than the GDPR requires — e.g. shorter deadlines, narrower retention periods,
  or additional procedural burdens beyond what the regulation mandates (over-compliance).

constraint_coverage (RCASR3) — a GDPR constraint with no policy counterpart at all —
is handled separately upstream and does NOT appear as a classifier label.\
"""

STAGE1_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert performing a binary compliance check.\n\n"
    "You will be given a GDPR regulatory constraint and a company policy sentence "
    "retrieved as a potential match. Determine whether the policy sentence contains a "
    "genuine compliance deviation relative to the GDPR constraint.\n\n"
    + _RCASR_TAXONOMY
    + "\n\n"
    "A genuine deviation requires a concrete, specific mismatch in one of the five "
    "categories above. Superficial wording differences or reasonable paraphrases are "
    "NOT deviations. Most retrieved pairs are compliant — default to false unless you "
    "can identify a specific conflict category.\n\n"
    "You MUST quote the exact text from both documents that creates the conflict before "
    "deciding. If you cannot identify a specific conflicting quote, has_deviation is false.\n\n"
    "Respond with valid JSON only. No markdown, no explanation outside the JSON."
)

STAGE2_SYSTEM_PROMPT = (
    "You are a GDPR compliance expert classifying a confirmed compliance deviation.\n\n"
    "A previous check confirmed that the company policy deviates from the GDPR "
    "constraint. Classify the deviation type using the RCASR taxonomy:\n\n"
    + _RCASR_TAXONOMY
    + "\n\n"
    "Assign exactly one of: responsibility, execution_style, data, negation, severity.\n\n"
    "Respond with valid JSON only. No markdown, no explanation outside the JSON."
)


def build_stage1_prompt(gdpr_text: str, gdpr_article: int, policy_text: str) -> str:
    return (
        f"GDPR constraint (Art. {gdpr_article}):\n"
        f'"{gdpr_text}"\n\n'
        f"Company policy sentence:\n"
        f'"{policy_text}"\n\n'
        "Step 1 — Quote the specific GDPR obligation from the constraint above.\n"
        "Step 2 — Quote the specific policy text that conflicts with it (or null if none).\n"
        "Step 3 — Identify which deviation category applies (or null if none).\n"
        "Step 4 — Set has_deviation to true only if a specific conflict exists.\n\n"
        '{"gdpr_quote": "<exact quote>", "policy_quote": "<exact quote or null>", '
        '"deviation_category": "<category or null>", '
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
