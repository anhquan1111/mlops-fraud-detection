"""Verify actual HTTP outcomes, bounded labels and error-path gauge cleanup."""

import pytest
from fastapi.testclient import TestClient

from src import api
from src.metrics import IN_FLIGHT, REGISTRY


def sample(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.mark.parametrize("outcome", ["ok", "invalid", "unavailable", "exception"])
def test_metrics_follow_http_outcome(monkeypatch, outcome):
    expected = {"ok": 200, "invalid": 422, "unavailable": 503, "exception": 500}[outcome]
    labels = {"method": "POST", "route": "/predict", "status_class": f"{expected // 100}xx"}
    before = sample("fraud_http_requests_total", labels)
    latency_before = sample(
        "fraud_http_request_duration_seconds_count",
        {
            "method": "POST",
            "route": "/predict",
        },
    )
    with TestClient(api.app) as client:
        payload = dict(api._EXAMPLE_TRANSACTION)
        if outcome == "invalid":
            payload["Amount"] = -1
        elif outcome == "unavailable":
            monkeypatch.setattr(api, "_model", None)
            assert client.get("/ready").status_code == 503
        elif outcome == "exception":

            def explode(*args, **kwargs):
                raise RuntimeError("internal-model-path-do-not-expose")

            monkeypatch.setattr(api, "_predict_one", explode)
        response = client.post("/predict", json=payload)
        assert response.status_code == expected
        assert "internal-model-path-do-not-expose" not in response.text
        assert client.get("/health").status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "fraud_http_request_duration_seconds_bucket" in metrics.text
    assert sample("fraud_http_requests_total", labels) == before + 1
    assert (
        sample(
            "fraud_http_request_duration_seconds_count",
            {
                "method": "POST",
                "route": "/predict",
            },
        )
        == latency_before + 1
    )
    assert IN_FLIGHT._value.get() == 0


def test_unknown_paths_share_one_label_and_scrapes_are_excluded():
    labels = {"method": "GET", "route": "unmatched", "status_class": "4xx"}
    before = sample("fraud_http_requests_total", labels)
    with TestClient(api.app) as client:
        client.get("/unknown/transaction-a")
        client.get("/unknown/transaction-b")
        text = client.get("/metrics").text
    assert sample("fraud_http_requests_total", labels) == before + 2
    assert "transaction-a" not in text and "transaction-b" not in text
    assert 'route="/metrics"' not in text
