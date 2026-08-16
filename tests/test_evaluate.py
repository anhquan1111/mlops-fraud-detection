"""Unit tests for src/evaluate.py.

Uses mock/toy models — does NOT require a real trained model or dataset.
"""

from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLS, MIN_PRECISION, MIN_RECALL
from src.evaluate import evaluate_model, print_report

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

N = 200
N_FRAUD = 20  # 10% fraud — easy to create perfect predictions


def _make_toy_data(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Create a tiny synthetic feature matrix and binary labels."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((N, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = pd.Series(np.zeros(N, dtype=int))
    fraud_idx = rng.choice(N, size=N_FRAUD, replace=False)
    y.iloc[fraud_idx] = 1
    return X, y


class _PerfectModel:
    """Toy model that always predicts correctly."""

    def __init__(self, y_true: pd.Series) -> None:
        self._y = y_true.values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Returns fraud_prob=1.0 for fraud, 0.0 for legit
        proba = self._y.astype(float)
        return np.column_stack([1 - proba, proba])


class _AllNegativeModel:
    """Toy model that always predicts 'not fraud' (prob=0)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.ones(n), np.zeros(n)])


class _HighThresholdModel:
    """Model returning fixed probability 0.4 (below default threshold of 0.5)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


@pytest.fixture()
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    return _make_toy_data()


# ---------------------------------------------------------------------------
# evaluate_model tests
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_returns_all_required_keys(self, toy_data) -> None:
        """evaluate_model must return all expected metric keys."""
        X, y = toy_data
        model = _PerfectModel(y)
        metrics = evaluate_model(model, X, y)
        expected_keys = {"precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_perfect_model_scores_one(self, toy_data) -> None:
        """A perfect model should get recall=1.0, precision=1.0, f1=1.0."""
        X, y = toy_data
        model = _PerfectModel(y)
        metrics = evaluate_model(model, X, y)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_all_negative_model_has_zero_recall(self, toy_data) -> None:
        """A model that always predicts 'not fraud' must have recall=0.0."""
        X, y = toy_data
        model = _AllNegativeModel()
        metrics = evaluate_model(model, X, y)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_all_negative_model_has_zero_precision(self, toy_data) -> None:
        """A model predicting no positives must have precision=0.0."""
        X, y = toy_data
        model = _AllNegativeModel()
        metrics = evaluate_model(model, X, y)
        assert metrics["precision"] == pytest.approx(0.0)

    def test_threshold_effect_on_predictions(self, toy_data) -> None:
        """Higher threshold → model predicts fewer positives → lower or equal recall."""
        X, y = toy_data
        model = _PerfectModel(y)
        # At threshold=0.5, perfect model gets recall=1.0
        low_thresh = evaluate_model(model, X, y, threshold=0.5)
        # At threshold=1.0, no sample has prob >= 1.0 → no positives predicted
        high_thresh = evaluate_model(model, X, y, threshold=1.0)
        assert high_thresh["recall"] <= low_thresh["recall"]

    def test_metrics_are_in_unit_interval(self, toy_data) -> None:
        """All metrics (except threshold) must be in [0, 1]."""
        X, y = toy_data
        model = _PerfectModel(y)
        metrics = evaluate_model(model, X, y)
        for key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    def test_threshold_recorded_in_output(self, toy_data) -> None:
        """The threshold used must be recorded in the output dict."""
        X, y = toy_data
        model = _PerfectModel(y)
        custom_thresh = 0.3
        metrics = evaluate_model(model, X, y, threshold=custom_thresh)
        assert metrics["threshold"] == pytest.approx(custom_thresh)

    def test_values_are_rounded(self, toy_data) -> None:
        """Metrics should be rounded to 4 decimal places."""
        X, y = toy_data
        model = _PerfectModel(y)
        metrics = evaluate_model(model, X, y)
        for key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            val = metrics[key]
            assert val == round(val, 4), f"{key}={val} is not rounded to 4 decimal places"

    def test_with_real_sklearn_model(self, toy_data) -> None:
        """evaluate_model should work with a real sklearn-compatible model."""
        X, y = toy_data
        # Use a simple LR; we just need it to produce valid probabilities
        model = LogisticRegression(max_iter=200, random_state=0)
        model.fit(X, y)
        metrics = evaluate_model(model, X, y)
        # Sanity: all metric values are in range
        for key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# print_report tests
# ---------------------------------------------------------------------------


class TestPrintReport:
    def _capture_print(self, metrics: dict, model_name: str = "TestModel") -> str:
        """Capture stdout from print_report."""
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            print_report(metrics, model_name=model_name)
        finally:
            sys.stdout = old_stdout
        return buffer.getvalue()

    def test_does_not_raise_on_passing_model(self) -> None:
        """print_report must not raise when model meets thresholds."""
        metrics = {
            "precision": MIN_PRECISION + 0.1,
            "recall": MIN_RECALL + 0.05,
            "f1": 0.85,
            "pr_auc": 0.88,
            "roc_auc": 0.97,
            "threshold": 0.5,
        }
        # Should not raise
        print_report(metrics, model_name="GoodModel")

    def test_does_not_raise_on_failing_model(self) -> None:
        """print_report must not raise when model fails thresholds (prints warning)."""
        metrics = {
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.1,
            "pr_auc": 0.1,
            "roc_auc": 0.5,
            "threshold": 0.5,
        }
        # Should not raise — just print a warning
        print_report(metrics, model_name="BadModel")

    def test_pass_message_in_output(self) -> None:
        """Passing model should print '[PASS]' in output."""
        metrics = {
            "precision": 0.75,
            "recall": 0.85,
            "f1": 0.80,
            "pr_auc": 0.87,
            "roc_auc": 0.97,
            "threshold": 0.5,
        }
        output = self._capture_print(metrics)
        assert "PASS" in output

    def test_warn_message_in_output(self) -> None:
        """Failing model should print '[WARN]' in output."""
        metrics = {
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.1,
            "pr_auc": 0.1,
            "roc_auc": 0.5,
            "threshold": 0.5,
        }
        output = self._capture_print(metrics)
        assert "WARN" in output

    def test_model_name_appears_in_output(self) -> None:
        """The model name passed should appear in the output."""
        metrics = {
            "precision": 0.75,
            "recall": 0.85,
            "f1": 0.80,
            "pr_auc": 0.87,
            "roc_auc": 0.97,
            "threshold": 0.5,
        }
        output = self._capture_print(metrics, model_name="LightGBM-Large")
        assert "LightGBM-Large" in output
