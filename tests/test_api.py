"""Tests for the FastAPI fraud detection endpoints.

Uses FastAPI's TestClient (synchronous) which wraps httpx.
Model is loaded from a temporary synthetic artifact provided by conftest.py.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_TRANSACTION = {
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan (loads model on startup)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------


class TestInfoEndpoints:
    def test_root_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_expected_keys(self, client: TestClient):
        data = client.get("/").json()
        assert "name" in data
        assert "version" in data
        assert "model" in data
        assert "endpoints" in data
        assert "threshold" in data

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client: TestClient):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, client: TestClient):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_valid_transaction(self, client: TestClient):
        response = client.post("/predict", json=EXAMPLE_TRANSACTION)
        assert response.status_code == 200

    def test_predict_response_schema(self, client: TestClient):
        data = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data
        assert "model_name" in data

    def test_predict_probability_in_range(self, client: TestClient):
        data = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_is_fraud_matches_threshold(self, client: TestClient):
        data = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        expected = data["fraud_probability"] >= data["threshold"]
        assert data["is_fraud"] == expected

    def test_predict_stub_response_is_repeatable(self, client: TestClient):
        """This is an API contract check, not evidence of real fraud-model quality."""
        data = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        repeated = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
        assert data == repeated

    def test_predict_negative_amount_rejected(self, client: TestClient):
        bad_tx = {**EXAMPLE_TRANSACTION, "Amount": -1.0}
        response = client.post("/predict", json=bad_tx)
        assert response.status_code == 422

    def test_predict_missing_field_rejected(self, client: TestClient):
        bad_tx = {k: v for k, v in EXAMPLE_TRANSACTION.items() if k != "V1"}
        response = client.post("/predict", json=bad_tx)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Batch predict endpoint
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    def test_batch_single_transaction(self, client: TestClient):
        response = client.post("/predict/batch", json=[EXAMPLE_TRANSACTION])
        assert response.status_code == 200

    def test_batch_response_schema(self, client: TestClient):
        data = client.post("/predict/batch", json=[EXAMPLE_TRANSACTION]).json()
        assert "predictions" in data
        assert "count" in data
        assert "fraud_count" in data
        assert data["count"] == 1

    def test_batch_multiple_transactions(self, client: TestClient):
        payload = [EXAMPLE_TRANSACTION] * 5
        data = client.post("/predict/batch", json=payload).json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_empty_list_rejected(self, client: TestClient):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 422

    def test_batch_over_limit_rejected(self, client: TestClient):
        payload = [EXAMPLE_TRANSACTION] * 101
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_fraud_count_consistent(self, client: TestClient):
        payload = [EXAMPLE_TRANSACTION] * 3
        data = client.post("/predict/batch", json=payload).json()
        actual_fraud = sum(1 for p in data["predictions"] if p["is_fraud"])
        assert actual_fraud == data["fraud_count"]


class TestMonitoringEndpoints:
    """Tests for the Evidently monitoring endpoints."""

    def test_monitoring_latest_returns_status(self, client: TestClient, monkeypatch, tmp_path):
        monkeypatch.setattr("src.api.REPORTS_DIR", tmp_path)
        response = client.get("/monitoring/latest")
        assert response.status_code == 404
        assert response.json()["status"] == "NO_REPORT"

    def test_monitoring_report_returns_html(self, client: TestClient, monkeypatch, tmp_path):
        monkeypatch.setattr("src.api.REPORTS_DIR", tmp_path)
        (tmp_path / "drift_summary.json").write_text('{"status": "SUCCESS"}')
        (tmp_path / "drift.html").write_text("<html>latest-test-report</html>")
        response = client.get("/monitoring/report")
        assert response.status_code == 200
        assert "latest-test-report" in response.text
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.parametrize("latest_status", ["QUALITY_FAILURE", "RUNNING", "ERROR"])
    def test_failed_or_running_job_hides_old_html(
        self, client, monkeypatch, tmp_path, latest_status
    ):
        import json

        monkeypatch.setattr("src.api.REPORTS_DIR", tmp_path)
        (tmp_path / "drift.html").write_text("<html>STALE-REPORT</html>")
        (tmp_path / "drift_summary.json").write_text(json.dumps({"status": latest_status}))
        assert client.get("/monitoring/latest").json()["status"] == latest_status
        response = client.get("/monitoring/report")
        assert response.status_code == 404
        assert "STALE-REPORT" not in response.text

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_input_rejected(self, client, value):
        response = client.post("/predict", json={**EXAMPLE_TRANSACTION, "V1": value})
        assert response.status_code == 422
