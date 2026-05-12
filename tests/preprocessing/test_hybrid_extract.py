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
