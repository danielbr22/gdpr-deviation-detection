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


import asyncio
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from unittest.mock import AsyncMock, patch


def test_judge_async_collects_matched_and_unmapped():
    from src.retrieval.llm_judge import run_judge_async

    topk_data = [
        {
            "gdpr_id": "gdpr_001",
            "gdpr_text": "Controller must inform data subject.",
            "gdpr_article": 13,
            "candidates": [
                {"policy_id": "pol_001", "policy_text": "We inform you.", "similarity": 0.9, "policy_section": "1"},
                {"policy_id": "pol_002", "policy_text": "We store data.", "similarity": 0.5, "policy_section": "2"},
            ],
        },
        {
            "gdpr_id": "gdpr_002",
            "gdpr_text": "Data must be adequate.",
            "gdpr_article": 5,
            "candidates": [
                {"policy_id": "pol_003", "policy_text": "Unrelated sentence.", "similarity": 0.2, "policy_section": "3"},
            ],
        },
    ]

    responses = [
        '{"match": 1, "reasoning": "Candidate 1 matches."}',  # gdpr_001 → matched
        '{"match": 0, "reasoning": "No match."}',              # gdpr_002 → unmapped
    ]
    response_iter = iter(responses)

    async def fake_call_async(system, user, **kwargs):
        return next(response_iter)

    with patch("src.utils.llm_client.call_async", side_effect=fake_call_async):
        with patch("src.utils.llm_client.concurrency", return_value=1):
            matched, unmapped = asyncio.run(run_judge_async(topk_data))

    assert len(matched) == 1
    assert matched[0]["gdpr_id"] == "gdpr_001"
    assert matched[0]["policy_id"] == "pol_001"
    assert len(unmapped) == 1
    assert unmapped[0]["gdpr_id"] == "gdpr_002"
    assert unmapped[0]["deviation_type"] == "missing_coverage"
