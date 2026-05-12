import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.retrieval.judge_prompt import build_judge_prompt, JUDGE_SYSTEM_PROMPT


def test_prompt_contains_gdpr_text():
    prompt = build_judge_prompt(
        gdpr_text="Personal data shall be processed lawfully.",
        gdpr_article=5,
        candidates=[{"policy_id": "pol_001", "policy_text": "We process data on legal grounds."}],
    )
    assert "Personal data shall be processed lawfully." in prompt


def test_prompt_contains_candidate_texts():
    prompt = build_judge_prompt(
        gdpr_text="X",
        gdpr_article=5,
        candidates=[
            {"policy_id": "pol_001", "policy_text": "We process data on legal grounds."},
            {"policy_id": "pol_002", "policy_text": "We store your email address."},
        ],
    )
    assert "We process data on legal grounds." in prompt
    assert "We store your email address." in prompt


def test_prompt_numbers_candidates():
    prompt = build_judge_prompt(
        gdpr_text="X",
        gdpr_article=5,
        candidates=[
            {"policy_id": "pol_001", "policy_text": "A."},
            {"policy_id": "pol_002", "policy_text": "B."},
        ],
    )
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_system_prompt_is_nonempty_string():
    assert isinstance(JUDGE_SYSTEM_PROMPT, str)
    assert len(JUDGE_SYSTEM_PROMPT) > 50
