"""
Integration tests — call Ollama with 3 explicit pairs.
Skipped automatically if Ollama is unreachable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import requests

from src.classification.classify import classify_pair, VALID_TYPES


def ollama_available():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(not ollama_available(), reason="Ollama not running")

# dev_003 — negation deviation: "cannot" vs GDPR right to erasure
PAIR_NEGATION = {
    "gdpr_id": "gdpr_test_negation",
    "policy_id": "pol_020",
    "gdpr_article": 17,
    "gdpr_text": (
        "The data subject shall have the right to obtain from the controller "
        "the erasure of personal data concerning him or her without undue delay."
    ),
    "policy_text": (
        "You cannot request the deletion of your data solely on the grounds "
        "that it is no longer required for its original purpose."
    ),
}

# dev_004 — responsibility deviation: "customer service" vs "data protection officer"
PAIR_RESPONSIBILITY = {
    "gdpr_id": "gdpr_test_responsibility",
    "policy_id": "pol_027",
    "gdpr_article": 37,
    "gdpr_text": "The controller and the processor shall designate a data protection officer.",
    "policy_text": (
        "If you would like to make use of any of these rights, "
        "please contact our customer service team directly."
    ),
}

# Clean pair — should ideally be none
PAIR_NONE = {
    "gdpr_id": "gdpr_test_none",
    "policy_id": "pol_019",
    "gdpr_article": 16,
    "gdpr_text": (
        "The data subject shall have the right to obtain from the controller "
        "without undue delay the rectification of inaccurate personal data concerning him or her."
    ),
    "policy_text": (
        "If the data we have stored about you is incorrect or incomplete, "
        "you can request its correction or completion."
    ),
}


@skip_if_no_ollama
def test_negation_pair_returns_valid_type(capsys):
    result = classify_pair(PAIR_NEGATION)
    assert result["deviation_type"] in VALID_TYPES | {"parse_error"}
    assert isinstance(result["reasoning"], str)
    print(f"\nnegation → {result['deviation_type']}: {result['reasoning']}")


@skip_if_no_ollama
def test_responsibility_pair_returns_valid_type(capsys):
    result = classify_pair(PAIR_RESPONSIBILITY)
    assert result["deviation_type"] in VALID_TYPES | {"parse_error"}
    assert isinstance(result["reasoning"], str)
    print(f"\nresponsibility → {result['deviation_type']}: {result['reasoning']}")


@skip_if_no_ollama
def test_clean_pair_returns_valid_type(capsys):
    result = classify_pair(PAIR_NONE)
    assert result["deviation_type"] in VALID_TYPES | {"parse_error"}
    assert isinstance(result["reasoning"], str)
    print(f"\nclean pair → {result['deviation_type']}: {result['reasoning']}")
