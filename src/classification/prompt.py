_RCASR_TAXONOMY = """\
Each constraint can be decomposed into three components: (responsibility, task, data).
A deviation occurs when the company policy differs from the GDPR constraint in one of
the following ways:

- responsibility (RCASR7): Responsibility and data match, but the wrong party is named
  as responsible for the task (WHO deviates).
- execution_style (RCASR5): Responsibility and data match, but the procedure for
  carrying out the task deviates — e.g. different channel, extra steps, different
  timeline (HOW deviates). Key signal: the GDPR grants a right or obligation without
  specifying HOW it must be exercised, but the policy adds a specific procedural
  requirement (e.g. "written request by post", "signed form", "authenticated portal
  only", "request must be notarised"). The added procedure restricts or changes how
  the right can be exercised, even if the underlying right itself is not denied.
- data (RCASR8): Responsibility and task match, but the scope of data covered deviates —
  e.g. "account data" instead of "personal data", or the policy narrows or broadens
  the category of data subject to a right or obligation (WHAT data deviates).
- negation (RCASR6): The policy directly contradicts or negates the GDPR constraint —
  e.g. "may not" where GDPR says "may", a right is explicitly denied, or the policy
  states it is not obliged to do something the GDPR requires.
  IMPORTANT: If the key difference is that the policy applies a right or obligation to a
  NARROWER CATEGORY OF DATA (e.g., "marketing-related personal data" instead of "personal
  data", or "account information" instead of "personal data"), that is a DATA deviation,
  not negation — even if the policy also includes other limiting language. Only classify as
  negation if the right or obligation is outright denied or reversed, not merely scoped to
  different data.
- severity (RCASR4): All three components match, but the policy deviates in the
  strictness of the standard — either stricter than GDPR requires (over-compliance,
  e.g. shorter deadlines than GDPR mandates, narrower retention than necessary) OR
  more lenient (under-compliance, e.g. longer response time than GDPR allows).
  Key signal: compare specific numbers, periods, or thresholds. If the GDPR says
  "within one month" and the policy says "within 5 business days", that is severity.
  If the GDPR says "no longer than necessary" and the policy specifies a concrete
  short retention period, that may also be severity.

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
    "IMPORTANT EXCEPTIONS — these are NOT deviations:\n"
    "  - If the GDPR constraint itself contains exceptions (e.g. Art. 9(2) exceptions to "
    "    the processing prohibition) and the policy invokes one of those exceptions, the "
    "    policy is COMPLIANT, not deviating. Citing a permitted exception is not negation.\n"
    "  - If the policy confirms a data processing agreement (DPA) with a processor, that "
    "    is compliant with Art. 28 obligations, NOT a negation — even if the wording "
    "    differs from the GDPR's exact phrasing.\n"
    "  - If the policy text addresses a DIFFERENT data subject right or obligation than "
    "    the one named in the GDPR constraint, it cannot constitute a deviation relative "
    "    to that constraint. has_deviation must be false.\n"
    "  - SEVERITY requires a specific numeric or time threshold in the GDPR constraint "
    "    itself (e.g. 'within one month', 'no longer than X years'). If the GDPR article "
    "    only requires the controller to DISCLOSE a retention period without setting one "
    "    (e.g. Art. 13/14 transparency obligations), a policy that states a specific "
    "    retention period is COMPLIANT — it satisfies the disclosure requirement. "
    "    has_deviation must be false for severity in that case.\n"
    "  - NEGATION requires the policy to DENY or REVERSE a right or obligation. A policy "
    "    sentence that merely DISCLOSES data practices (e.g. 'data may be shared with "
    "    third parties', 'data is transferred outside the EEA') is a transparency statement "
    "    and is NOT a negation of the GDPR transparency obligation — it IS compliance with "
    "    it. has_deviation must be false for negation in that case.\n\n"
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
        "  Special check for DATA deviation: Does the GDPR say 'personal data' but the policy\n"
        "  applies the right/obligation only to a narrower category such as 'account information',\n"
        "  'marketing data', 'profile data', etc.? If so, that IS a data deviation.\n"
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
