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


import asyncio
from unittest.mock import patch


def test_classify_pairs_async_no_deviation():
    from src.classification.classify import classify_pairs_async

    pair = {
        "gdpr_id": "gdpr_001",
        "policy_id": "pol_002",
        "gdpr_article": 5,
        "gdpr_text": "Data shall be adequate.",
        "policy_text": "We collect all data.",
        "similarity": 0.7,
    }

    stage1_response = '{"has_deviation": false, "reasoning": "Compliant."}'

    async def fake_call(system, user, **kwargs):
        return stage1_response

    with patch("src.utils.llm_client.call_async", side_effect=fake_call):
        with patch("src.utils.llm_client.concurrency", return_value=1):
            results = asyncio.run(classify_pairs_async([pair]))

    assert len(results) == 1
    assert results[0]["deviation_type"] == "none"


def test_classify_pairs_async_deviation_detected():
    from src.classification.classify import classify_pairs_async

    pair = {
        "gdpr_id": "gdpr_001",
        "policy_id": "pol_002",
        "gdpr_article": 5,
        "gdpr_text": "Data shall be adequate.",
        "policy_text": "We collect all data without restriction.",
        "similarity": 0.7,
    }

    stage1_response = '{"has_deviation": true, "reasoning": "Deviation found."}'
    stage2_response = '{"deviation_type": "data", "reasoning": "Data minimisation violated."}'
    responses = iter([stage1_response, stage2_response])

    async def fake_call(system, user, **kwargs):
        return next(responses)

    with patch("src.utils.llm_client.call_async", side_effect=fake_call):
        with patch("src.utils.llm_client.concurrency", return_value=1):
            results = asyncio.run(classify_pairs_async([pair]))

    assert results[0]["deviation_type"] == "data"
    assert results[0]["reasoning"] == "Data minimisation violated."
