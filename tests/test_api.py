"""
FastAPI Integration & Endpoint Unit Tests using TestClient and Mock InferenceClient.
"""

import sys
from pathlib import Path
import json
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app
from src.scoring.inference_client import InferenceClient

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_backend" in data
    assert "catalog_facets_loaded" in data
    assert data["catalog_facets_loaded"] > 0


def test_api_evaluate_valid_request():
    payload = {"conversation": "I am taking a wild risk going skydiving!"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "metadata" in data
    assert data["metadata"]["retrieved_count"] > 0
    assert len(data["results"]) > 0


def test_api_evaluate_empty_conversation():
    payload = {"conversation": "   "}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422  # Pydantic validation error for whitespace
    data = response.json()
    assert "error" in data


def test_api_evaluate_missing_payload():
    response = client.post("/evaluate", json={})
    assert response.status_code == 422


def test_api_get_facet_by_id_valid():
    response = client.get("/facets/FACET_001")
    assert response.status_code == 200
    data = response.json()
    assert data["facet_id"] == "FACET_001"


def test_api_get_facet_by_id_invalid():
    response = client.get("/facets/FACET_INVALID_9999")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_api_retrieve_endpoint():
    payload = {"conversation": "I have been feeling dizzy.", "top_k": 5}
    response = client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "retrieved_candidates" in data
    assert data["total_retrieved"] == 5


def test_inference_client_health_check_offline():
    client_obj = InferenceClient(inference_url="http://localhost:9999", timeout=1)
    is_ok, msg = client_obj.health_check()
    assert is_ok is False
    assert "unreachable" in msg.lower() or "failed" in msg.lower()


@patch("urllib.request.urlopen")
def test_inference_client_generate_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": "[\n  {\"facet_id\": \"FACET_001\", \"status\": \"scored\", \"score\": 4}\n]"}}]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client_obj = InferenceClient(inference_url="http://localhost:8000")
    res = client_obj.generate("system prompt", "user prompt")
    assert "FACET_001" in res


def test_api_cors_preflight():
    response = client.options(
        "/evaluate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        }
    )
    assert response.status_code in [200, 204]
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
