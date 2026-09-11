"""Regression cases found while reviewing the Day 3-4 integration."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from boto3.exceptions import S3UploadFailedError
from fastapi.testclient import TestClient

from src import api, monitor
from src.config import FEATURE_COLS
from src.quality import quality_gate
from src.storage import upload_to_s3


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch"])
@pytest.mark.parametrize("amount", [0.0, 1.0, 50.0])
def test_valid_raw_amount_reaches_model(monkeypatch, endpoint, amount):
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1]])
    monkeypatch.setattr(api, "_model", model)
    payload = {col: 0.1 for col in FEATURE_COLS} | {"Amount": amount}
    with TestClient(api.app) as client:
        # Lifespan may load a test artifact; inject after startup.
        monkeypatch.setattr(api, "_model", model)
        response = client.post(endpoint, json=[payload] if endpoint.endswith("batch") else payload)
    assert response.status_code == 200, response.text
    frame = model.predict_proba.call_args.args[0]
    assert frame["Amount"].iloc[0] == pytest.approx((amount - api._AMOUNT_MEAN) / api._AMOUNT_STD)


def test_quality_http_error_is_not_wrapped_as_500(monkeypatch):
    monkeypatch.setattr(api, "quality_gate", lambda *args, **kwargs: ["invalid_input"])
    with TestClient(api.app) as client:
        response = client.post("/predict", json=api._EXAMPLE_TRANSACTION)
    assert response.status_code == 422


def test_missing_source_never_creates_fake_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "RAW_DATA_PATH", tmp_path / "absent.csv")
    monkeypatch.setattr(monitor, "REFERENCE_BASELINE_PATH", tmp_path / "baseline.parquet")
    monkeypatch.setattr(monitor, "DATA_PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        monitor.get_or_create_reference_baseline()
    assert not (tmp_path / "baseline.parquet").exists()


def test_duplicate_feature_columns_are_quality_failure():
    frame = pd.DataFrame([[1.0] * 30], columns=FEATURE_COLS + ["Amount"])
    assert "duplicate_column:Amount" in quality_gate(frame, raise_on_error=False)


def test_managed_s3_upload_failure_returns_false(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}")
    client = MagicMock()
    client.upload_file.side_effect = S3UploadFailedError("Access denied")
    assert upload_to_s3(path, "report.json", client=client) is False
