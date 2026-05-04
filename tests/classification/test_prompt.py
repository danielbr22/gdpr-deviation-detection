import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.classification.prompt import SYSTEM_PROMPT, build_user_prompt


def test_user_prompt_includes_gdpr_text():
    p = build_user_prompt("data shall be lawful", 5, "we process data lawfully")
    assert "data shall be lawful" in p


def test_user_prompt_includes_policy_text():
    p = build_user_prompt("data shall be lawful", 5, "we process data lawfully")
    assert "we process data lawfully" in p


def test_user_prompt_includes_article():
    p = build_user_prompt("x", 17, "y")
    assert "17" in p


def test_user_prompt_includes_json_template():
    p = build_user_prompt("x", 5, "y")
    assert "deviation_type" in p
    assert "reasoning" in p


def test_system_prompt_includes_all_deviation_types():
    for t in ["none", "responsibility", "execution_style", "data", "negation"]:
        assert t in SYSTEM_PROMPT, f"Missing deviation type: {t}"


def test_system_prompt_instructs_json_only():
    assert "JSON" in SYSTEM_PROMPT
