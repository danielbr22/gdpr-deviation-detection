import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.retrieval.llm_judge import parse_judge_response


def test_parse_valid_match():
    raw = '{"match": 2, "reasoning": "Candidate 2 directly addresses Art. 5."}'
    result = parse_judge_response(raw, n_candidates=3)
    assert result is not None
    assert result["match"] == 2
    assert "Art. 5" in result["reasoning"]


def test_parse_no_match_zero():
    raw = '{"match": 0, "reasoning": "No candidate is relevant."}'
    result = parse_judge_response(raw, n_candidates=3)
    assert result is not None
    assert result["match"] == 0


def test_parse_invalid_json_returns_none():
    assert parse_judge_response("not json at all", n_candidates=3) is None


def test_parse_match_out_of_range_returns_none():
    raw = '{"match": 5, "reasoning": "x"}'
    result = parse_judge_response(raw, n_candidates=3)
    assert result is None


def test_parse_missing_reasoning_returns_none():
    raw = '{"match": 1}'
    result = parse_judge_response(raw, n_candidates=3)
    assert result is None


def test_parse_match_as_string_number():
    # LLMs sometimes return "1" instead of 1
    raw = '{"match": "1", "reasoning": "Direct match."}'
    result = parse_judge_response(raw, n_candidates=2)
    assert result is not None
    assert result["match"] == 1
