import json
import os
import pytest
import httpx
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

from healthcare_rag.tools.openfda_search import fetch_openfda, openfda_search

load_dotenv()

# These tests are designed to pass WITHOUT hitting the live FDA API
#   (using httpx mocking), plus a small set of live smoke tests that
#   are skipped in CI unless OPENFDA_LIVE_TESTS=1 is set.
# Run with: uv run pytest tests/test_agent_openfda.py -v

# Shared fixtures
LABEL_RESPONSE = {
    "meta": {"results": {"total": 42}},
    "results": [
        {
            "openfda": {
                "brand_name": ["Eliquis"],
                "generic_name": ["apixaban"],
            },
            "indications_and_usage": [
                "Eliquis is indicated to reduce the risk of stroke and systemic "
                "embolism in patients with nonvalvular atrial fibrillation."
            ],
            "warnings": [
                "PREMATURE DISCONTINUATION OF ELIQUIS INCREASES THE RISK OF "
                "ISCHEMIC EVENTS."
            ],
            "dosage_and_administration": ["2.5 mg or 5 mg twice daily."],
            "contraindications": ["Active pathological bleeding."],
        }
    ],
}

EVENT_RESPONSE = {
    "meta": {"results": {"total": 155}},
    "results": [
        {
            "receiptdate": "20230601",
            "serious": "1",
            "patient": {
                "reaction": [
                    {"reactionmeddrapt": "HAEMORRHAGE", "reactionoutcome": "5"},  # fatal
                    {"reactionmeddrapt": "DIZZINESS", "reactionoutcome": "1"},    # recovered
                ],
            },
        }
    ],
}

ENFORCEMENT_RESPONSE = {
    "meta": {"results": {"total": 3}},
    "results": [
        {
            "classification": "Class II",
            "status": "Ongoing",
            "recall_initiation_date": "20240101",
            "product_description": "Apixaban 5mg tablets, lot ABC123",
            "reason_for_recall": "Labeling error — incorrect dosage on package insert.",
        }
    ],
}

def _mock_response(body: dict, status: int = 200) -> MagicMock:
    # Build a minimal httpx.Response mock.
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()  # no-op for 200
    return mock

# Unit tests (no live network)
class TestFetchOpenFDA:
    def test_label_happy_path(self):
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(LABEL_RESPONSE)
            )
            result = fetch_openfda("Eliquis", "label")

        assert "Eliquis" in result
        assert "apixaban" in result.lower() or "apixaban" in result
        assert "indications" in result.lower() or "Indications" in result

    def test_event_happy_path(self):
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(EVENT_RESPONSE)
            )
            result = fetch_openfda("Eliquis", "event")

        assert "HAEMORRHAGE" in result or "haemorrhage" in result.lower()
        assert "fatal" in result.lower()

    def test_enforcement_happy_path(self):
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(ENFORCEMENT_RESPONSE)
            )
            result = fetch_openfda("apixaban", "enforcement")

        assert "Class II" in result
        assert "Labeling error" in result

    def test_bad_query_type_returns_string(self):
        # Invalid query_type must return an error string, not raise.
        result = fetch_openfda("Eliquis", "banana")
        assert "Invalid query_type" in result
        assert "banana" in result

    def test_404_no_results_returns_helpful_string(self):
        # When the drug isn't found, return a helpful message without raising.
        not_found = MagicMock(spec=httpx.Response)
        not_found.status_code = 404
        not_found.json.return_value = {}
        not_found.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=not_found
        )

        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = not_found
            result = fetch_openfda("xyznotadrug99", "label")

        assert "No openFDA" in result or "not found" in result.lower()
        # Crucially, ensure that it did not raise
        assert isinstance(result, str)

    def test_timeout_returns_string(self):
        # Network timeout must never crash the agent (return a string).
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = (
                httpx.TimeoutException("timed out")
            )
            result = fetch_openfda("Eliquis", "label")

        assert "timed out" in result.lower() or "timeout" in result.lower()
        assert isinstance(result, str)

    def test_empty_results_returns_helpful_string(self):
        # Results list present but empty (should explain gracefully).
        empty = {"meta": {"results": {"total": 0}}, "results": []}
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(empty)
            )
            result = fetch_openfda("xyznotadrug99", "label")

        assert "No openFDA" in result
        assert isinstance(result, str)

    def test_fallback_broad_search_triggered_on_empty_brand_search(self):
        # If the brand-name scoped search returns 0 results, we should try
        #   a broader (unscoped) search. Verify that get() is called twice.

        empty   = {"meta": {"results": {"total": 0}}, "results": []}
        success = LABEL_RESPONSE

        call_count = 0

        def side_effect(url, params):
            nonlocal call_count
            call_count += 1
            resp = _mock_response(empty if call_count == 1 else success)
            return resp

        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = side_effect
            result = fetch_openfda("eliquis", "label")

        assert call_count == 2, "Expected brand-search then fallback broad search"
        assert "Eliquis" in result or "apixaban" in result.lower()

class TestOpenFDALangChainTool:
    def test_tool_name(self):
        assert openfda_search.name == "openfda_search"

    def test_tool_description_contains_positive_examples(self):
        desc = openfda_search.description
        assert "label" in desc.lower()
        assert "adverse" in desc.lower() or "event" in desc.lower()
        assert "recall" in desc.lower() or "enforcement" in desc.lower()

    def test_tool_description_contains_negative_examples(self):
        # NOTE: Agent relies on 'Do NOT use' to avoid misrouting.
        desc = openfda_search.description
        assert "do not" in desc.lower() or "don't" in desc.lower()

    def test_tool_invoke_returns_string(self):
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(LABEL_RESPONSE)
            )
            result = openfda_search.invoke({"drug_name": "Eliquis"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_default_query_type_is_label(self):
        # Omitting query_type should default to 'label' without error.
        with patch("healthcare_rag.tools.openfda_search.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                _mock_response(LABEL_RESPONSE)
            )
            result = openfda_search.invoke({"drug_name": "Eliquis"})

        assert isinstance(result, str)

# Live smoke tests. Opt-in only (set OPENFDA_LIVE_TESTS=1)
LIVE = pytest.mark.skipif(
    os.environ.get("OPENFDA_LIVE_TESTS") != "1",
    reason="Live FDA API tests skipped; set OPENFDA_LIVE_TESTS=1 to enable",
)

@LIVE
def test_live_eliquis_label():
    result = fetch_openfda("Eliquis", "label")
    assert "apixaban" in result.lower() or "Eliquis" in result
    assert "indications" in result.lower()

@LIVE
def test_live_eliquis_event():
    result = fetch_openfda("Eliquis", "event")
    # Should contain at least one reaction term
    assert "reaction" in result.lower() or "Report" in result

@LIVE
def test_live_nonsense_drug_returns_string():
    """Nonsense drug name must return a string, never crash."""
    result = fetch_openfda("xyznotadrug_abc999", "label")
    assert isinstance(result, str)
    assert len(result) > 0

@LIVE
def test_live_enforcement_lipitor():
    result = fetch_openfda("Lipitor", "enforcement")
    assert isinstance(result, str)