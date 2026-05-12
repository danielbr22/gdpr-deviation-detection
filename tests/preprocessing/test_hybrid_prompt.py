import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing.hybrid_prompt import build_extraction_prompt, EXTRACTION_SYSTEM_PROMPT


def test_prompt_contains_target_sentence():
    prompt = build_extraction_prompt(
        sentence="You have the right to access your data.",
        context_before=["We process your data."],
        context_after=["You can request deletion."],
    )
    assert "You have the right to access your data." in prompt


def test_prompt_contains_before_context():
    prompt = build_extraction_prompt(
        sentence="We will inform you.",
        context_before=["Previous sentence."],
        context_after=["Next sentence."],
    )
    assert "Previous sentence." in prompt


def test_prompt_contains_after_context():
    prompt = build_extraction_prompt(
        sentence="We will inform you.",
        context_before=["Previous sentence."],
        context_after=["Next sentence."],
    )
    assert "Next sentence." in prompt


def test_prompt_handles_empty_context():
    prompt = build_extraction_prompt(
        sentence="We process data.",
        context_before=[],
        context_after=[],
    )
    assert "We process data." in prompt
    assert isinstance(prompt, str)


def test_system_prompt_is_nonempty_string():
    assert isinstance(EXTRACTION_SYSTEM_PROMPT, str)
    assert len(EXTRACTION_SYSTEM_PROMPT) > 50


def test_prompt_empty_context_shows_fallback():
    prompt = build_extraction_prompt("We process data.", [], [])
    assert "(start of section)" in prompt
    assert "(end of section)" in prompt
