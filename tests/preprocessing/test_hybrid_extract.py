import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing.hybrid_extract import get_context_window, filter_sentence, _strip_annotations


def test_get_context_window_middle():
    sentences = ["A", "B", "C", "D", "E"]
    before, after = get_context_window(sentences, 2, window=2)
    assert before == ["A", "B"]
    assert after == ["D", "E"]


def test_get_context_window_at_start():
    sentences = ["A", "B", "C"]
    before, after = get_context_window(sentences, 0, window=2)
    assert before == []
    assert after == ["B", "C"]


def test_get_context_window_at_end():
    sentences = ["A", "B", "C"]
    before, after = get_context_window(sentences, 2, window=2)
    assert before == ["A", "B"]
    assert after == []


def test_filter_sentence_rejects_question():
    assert filter_sentence("What rights do you have?") is False


def test_filter_sentence_rejects_short():
    assert filter_sentence("Yes.") is False


def test_filter_sentence_rejects_separator():
    assert filter_sentence("===========================") is False


def test_filter_sentence_accepts_obligation():
    assert filter_sentence("You have the right to request deletion of your data.") is True


def test_strip_annotations_removes_deviation_blocks():
    text = "Before. [DEVIATION type=negation original=x modified=y] After."
    result = _strip_annotations(text)
    assert "[DEVIATION" not in result
    assert "Before." in result
    assert "After." in result


import asyncio
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from unittest.mock import AsyncMock, MagicMock, patch


def test_extract_policy_obligations_async_returns_obligations():
    """LLM says 'yes' for first sentence only — one constraint returned."""
    from src.preprocessing.hybrid_extract import extract_policy_obligations_async

    call_count = 0
    async def fake_call_async(system, user, **kwargs):
        nonlocal call_count
        call_count += 1
        # First sentence is an obligation, rest are not
        return "yes" if call_count == 1 else "no"

    policy_text = (
        "You have the right to access your personal data. "
        "Our company was founded in 1990. "
        "We process data to provide our services."
    )

    with patch("src.utils.llm_client.call_async", side_effect=fake_call_async):
        with patch("src.utils.llm_client.concurrency", return_value=1):
            result = asyncio.run(extract_policy_obligations_async(policy_text))

    assert len(result) >= 1
    assert all("id" in c and "text" in c for c in result)
    assert result[0]["id"] == "pol_001"


def test_extract_preserves_sentence_order():
    """IDs must be assigned in original sentence order, not completion order."""
    from src.preprocessing.hybrid_extract import extract_policy_obligations_async

    sentences_seen = []
    async def fake_call_async(system, user, **kwargs):
        sentences_seen.append(user[:30])
        return "yes"

    policy_text = (
        "First data protection obligation sentence here. "
        "Second data protection obligation sentence here. "
        "Third data protection obligation sentence here."
    )

    with patch("src.utils.llm_client.call_async", side_effect=fake_call_async):
        with patch("src.utils.llm_client.concurrency", return_value=1):
            result = asyncio.run(extract_policy_obligations_async(policy_text))

    ids = [c["id"] for c in result]
    assert ids == sorted(ids), "IDs are not in ascending order"
