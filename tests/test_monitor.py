"""Unit tests for Evidently monitoring and Matured Cohort analysis (src/monitor.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.features import load_data, preprocess, split_data
from src.monitor import (
    get_or_create_reference_baseline,
    matured_cohort_audit,
    run_drift_analysis,
)


@pytest.fixture
def synthetic_baseline() -> pd.DataFrame:
    """Generate a clean synthetic baseline of 500 rows."""
    rng = np.random.default_rng(42)
    data = {f"V{i}": rng.normal(0, 1, 500) for i in range(1, 29)}
    data["Amount"] = rng.lognormal(4.0, 1.0, 500)
    return pd.DataFrame(data)


class TestDriftAnalysis:
    """Test suite for run_drift_analysis."""

    def test_stable_data_reports_no_drift(self, synthetic_baseline: pd.DataFrame, tmp_path) -> None:
        # Current identical to reference distribution
        current = synthetic_baseline.copy()
        result = run_drift_analysis(
            current_df=current,
            reference_df=synthetic_baseline,
            output_dir=tmp_path,
        )
        assert result["status"] == "SUCCESS"
        assert result["drift"]["dataset_drift"] is False
        assert result["drift"]["drift_count"] == 0

    def test_drifted_data_detected(self, synthetic_baseline: pd.DataFrame, tmp_path) -> None:
        # Shift Amount and multiple V features
        current = synthetic_baseline.copy()
        current["Amount"] = current["Amount"] * 10
        for i in range(1, 15):
            current[f"V{i}"] += 5.0

        result = run_drift_analysis(
            current_df=current,
            reference_df=synthetic_baseline,
            output_dir=tmp_path,
        )
        assert result["status"] == "SUCCESS"
        assert result["drift"]["dataset_drift"] is True
        assert result["drift"]["columns"]["Amount"]["drift_detected"] is True

    def test_quality_failure_aborts_drift(self, synthetic_baseline: pd.DataFrame, tmp_path) -> None:
        # Introduce null into Amount
        current = synthetic_baseline.copy()
        current.loc[0, "Amount"] = np.nan

        result = run_drift_analysis(
            current_df=current,
            reference_df=synthetic_baseline,
            output_dir=tmp_path,
        )
        assert result["status"] == "QUALITY_FAILURE"
        assert result["drift"] is None
        assert "drift_not_run" in result["action"]

    def test_invalid_reference_never_runs_evidently(self, synthetic_baseline, tmp_path):
        reference = synthetic_baseline.drop(columns=["V1"])
        with patch("src.monitor.Report") as report:
            result = run_drift_analysis(synthetic_baseline, reference, tmp_path)
        assert result["status"] == "QUALITY_FAILURE"
        assert result["quality_source"] == "reference"
        assert result["drift"] is None
        report.assert_not_called()

    def test_no_drift_does_not_claim_model_health(self, synthetic_baseline, tmp_path):
        result = run_drift_analysis(synthetic_baseline, synthetic_baseline, tmp_path)
        assert "performance_not_assessed" in result["action"]

    def test_engine_error_invalidates_latest_report(self, synthetic_baseline, tmp_path):
        import json

        with patch("src.monitor.Report", side_effect=RuntimeError("engine failed")):
            with pytest.raises(RuntimeError, match="engine failed"):
                run_drift_analysis(synthetic_baseline, synthetic_baseline, tmp_path)
        assert json.loads((tmp_path / "drift_summary.json").read_text())["status"] == "ERROR"


def test_baseline_uses_training_rows_and_verifiable_raw_amount(synthetic_baseline, tmp_path):
    import json

    raw = synthetic_baseline.copy()
    raw["V1"] = np.arange(len(raw), dtype=float)  # Trace row membership across the split.
    raw["Time"] = np.arange(len(raw))
    raw[TARGET_COL] = [0] * 450 + [1] * 50
    source, destination = tmp_path / "raw.csv", tmp_path / "baseline.parquet"
    raw.to_csv(source, index=False)
    train, val, test = split_data(*preprocess(load_data(source)))[:3]
    baseline = get_or_create_reference_baseline(
        sample_size=100,
        raw_path=source,
        baseline_path=destination,
    )
    assert set(baseline["V1"]).issubset(train["V1"])
    assert set(baseline["V1"]).isdisjoint(val["V1"])
    assert set(baseline["V1"]).isdisjoint(test["V1"])
    assert (baseline["Amount"] >= 0).all()
    metadata = json.loads(destination.with_suffix(".metadata.json").read_text())
    assert metadata["amount_scale"] == "raw"
    cached = get_or_create_reference_baseline(raw_path=source, baseline_path=destination)
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), cached)
    baseline.assign(Amount=0).to_parquet(destination, index=False)
    with pytest.raises(ValueError, match="mismatch"):
        get_or_create_reference_baseline(raw_path=source, baseline_path=destination)


def test_future_labels_do_not_complete_matured_cohort():
    scored = pd.DataFrame(
        {
            "transaction_id": ["001", "002", "003"],
            "event_time": ["2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z", "2026-09-09T00:00:00Z"],
            "prediction": [1, 0, 0],
        }
    )
    labels = pd.DataFrame(
        {
            "transaction_id": ["001", "002", "003"],
            "label_available_at": [
                "2026-09-08T00:00:00Z",
                "2026-09-11T00:00:00Z",
                "2026-09-10T00:00:00Z",
            ],
            "Class": [1, 0, 0],
        }
    )
    result = matured_cohort_audit(scored, labels, pd.Timestamp("2026-09-10T00:00:00Z"))
    assert result["matured_cohort_size"] == 2
    assert result["observed_labels_count"] == 1
    assert result["metrics"] is None
    result = matured_cohort_audit(scored, labels, pd.Timestamp("2026-09-11T00:00:00Z"))
    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["tn"] == 1


@pytest.mark.parametrize("bad_prediction", [0.9, 2, np.nan])
def test_audit_rejects_invalid_prediction_labels(bad_prediction):
    scored = pd.DataFrame(
        {
            "transaction_id": ["001"],
            "event_time": ["2026-09-01"],
            "prediction": [bad_prediction],
        }
    )
    labels = pd.DataFrame(
        {
            "transaction_id": ["001"],
            "label_available_at": ["2026-09-08"],
            "Class": [1],
        }
    )
    with pytest.raises(ValueError, match="binary labels"):
        matured_cohort_audit(scored, labels, pd.Timestamp("2026-09-10", tz="UTC"))


class TestMaturedCohortAudit:
    """Test suite for matured_cohort_audit."""

    def test_complete_cohort_calculates_metrics(self) -> None:
        as_of = pd.Timestamp("2026-09-10T00:00:00Z")
        n = 100

        scored_df = pd.DataFrame(
            {
                "transaction_id": [f"tx_{i}" for i in range(n)],
                "event_time": [pd.Timestamp("2026-09-01T00:00:00Z")] * n,
                "prediction": [1 if i < 10 else 0 for i in range(n)],
                "fraud_probability": [0.9 if i < 10 else 0.1 for i in range(n)],
            }
        )

        labels_df = pd.DataFrame(
            {
                "transaction_id": [f"tx_{i}" for i in range(n)],
                "label_available_at": [pd.Timestamp("2026-09-08T00:00:00Z")] * n,
                TARGET_COL: [1 if i < 10 else 0 for i in range(n)],
            }
        )

        result = matured_cohort_audit(scored_df, labels_df, as_of=as_of, label_delay_days=7)
        assert result["status"] == "complete_matured_cohort"
        assert result["label_coverage"] == 1.0
        assert result["metrics"]["accuracy"] == 1.0
        assert result["metrics"]["tp"] == 10

    def test_partial_cohort_returns_insufficient_labels(self) -> None:
        as_of = pd.Timestamp("2026-09-10T00:00:00Z")
        n = 100

        scored_df = pd.DataFrame(
            {
                "transaction_id": [f"tx_{i}" for i in range(n)],
                "event_time": [pd.Timestamp("2026-09-01T00:00:00Z")] * n,
                "prediction": [0] * n,
            }
        )

        # Missing 1 label
        labels_df = pd.DataFrame(
            {
                "transaction_id": [f"tx_{i}" for i in range(n - 1)],
                "label_available_at": [pd.Timestamp("2026-09-08T00:00:00Z")] * (n - 1),
                TARGET_COL: [0] * (n - 1),
            }
        )

        result = matured_cohort_audit(scored_df, labels_df, as_of=as_of, label_delay_days=7)
        assert result["status"] == "insufficient_labels"
        assert result["metrics"] is None
        assert result["label_coverage"] < 1.0
