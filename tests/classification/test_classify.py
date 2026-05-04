import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.classification.classify import parse_response, filter_pairs


def test_parse_response_valid_none():
    raw = '{"deviation_type": "none", "reasoning": "Both match."}'
    result = parse_response(raw)
    assert result is not None
    assert result["deviation_type"] == "none"
    assert result["reasoning"] == "Both match."


def test_parse_response_valid_negation():
    raw = '{"deviation_type": "negation", "reasoning": "Policy contradicts GDPR."}'
    result = parse_response(raw)
    assert result["deviation_type"] == "negation"


def test_parse_response_invalid_json_returns_none():
    assert parse_response("not json at all") is None


def test_parse_response_invalid_type_returns_none():
    raw = '{"deviation_type": "unknown_type", "reasoning": "x"}'
    assert parse_response(raw) is None


def test_parse_response_missing_reasoning_returns_none():
    raw = '{"deviation_type": "none"}'
    assert parse_response(raw) is None


def test_filter_pairs_removes_pol001():
    pairs = [
        {"gdpr_id": "gdpr_001", "policy_id": "pol_001"},
        {"gdpr_id": "gdpr_002", "policy_id": "pol_004"},
    ]
    result = filter_pairs(pairs)
    assert len(result) == 1
    assert result[0]["policy_id"] == "pol_004"


def test_filter_pairs_keeps_all_when_no_pol001():
    pairs = [
        {"gdpr_id": "gdpr_001", "policy_id": "pol_002"},
        {"gdpr_id": "gdpr_002", "policy_id": "pol_003"},
    ]
    assert len(filter_pairs(pairs)) == 2
