"""Unit tests for src/validate.py — the validation gate.

All MLflow interactions are mocked — no real MLflow server required.
Tests cover: first deployment, promotion, rejection (PR-AUC regression,
recall failure, precision failure).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import MIN_PRECISION, MIN_RECALL
from src.validate import ValidationStatus, run_validation_gate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_METRICS = {
    "pr_auc": 0.90,
    "recall": 0.85,
    "precision": 0.75,
    "f1": 0.80,
    "roc_auc": 0.97,
}

_PROD_METRICS = {
    "pr_auc": 0.85,
    "recall": 0.82,
    "precision": 0.70,
    "f1": 0.75,
    "roc_auc": 0.96,
}


def _make_mock_client(
    prod_metrics: dict | None = None,
    raw_metric_keys: bool = False,
) -> MagicMock:
    """Build a mock MlflowClient.

    Args:
        prod_metrics: Metrics for the production model, or None to simulate
                      no existing production model.
        raw_metric_keys: Store prod_metrics under bare keys instead of `val_*`,
                      simulating a run from before the leak-free protocol.
    """
    client = MagicMock()

    if prod_metrics is None:
        # No production model → get_model_version_by_alias raises MlflowException
        from mlflow.exceptions import MlflowException

        client.get_model_version_by_alias.side_effect = MlflowException("Not found")
    else:
        # Return a fake model version with a run_id
        mock_version = MagicMock()
        mock_version.run_id = "fake-run-id"
        client.get_model_version_by_alias.return_value = mock_version

        # Return a fake run with the given metrics. A real leak-free run stores
        # them under `val_*`; the gate reads that family so that promotion is
        # never decided on test-set numbers. Callers of run_validation_gate()
        # still pass plain keys — the prefix is a storage detail of the run.
        mock_run = MagicMock()
        mock_run.data.metrics = (
            prod_metrics if raw_metric_keys else {f"val_{k}": v for k, v in prod_metrics.items()}
        )
        client.get_run.return_value = mock_run

    return client


# ---------------------------------------------------------------------------
# Tests: First deployment
# ---------------------------------------------------------------------------


class TestFirstDeployment:
    def test_status_is_first_deployment(self) -> None:
        """When no production model exists, status should be FIRST_DEPLOYMENT."""
        client = _make_mock_client(prod_metrics=None)

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_register.return_value = mock_version

            result = run_validation_gate(
                run_id="new-run-id",
                new_metrics=_GOOD_METRICS,
                client=client,
                promote_on_pass=True,
            )

        assert result.status == ValidationStatus.FIRST_DEPLOYMENT

    def test_first_deployment_promotes_model(self) -> None:
        """First deployment should trigger model registration."""
        client = _make_mock_client(prod_metrics=None)

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_register.return_value = mock_version

            result = run_validation_gate(
                run_id="new-run-id",
                new_metrics=_GOOD_METRICS,
                client=client,
                promote_on_pass=True,
            )

        mock_register.assert_called_once()
        assert result.promoted_version == "1"

    def test_first_deployment_dry_run_does_not_register(self) -> None:
        """With promote_on_pass=False, no registration should occur."""
        client = _make_mock_client(prod_metrics=None)

        with patch("src.validate.mlflow.register_model") as mock_register:
            result = run_validation_gate(
                run_id="new-run-id",
                new_metrics=_GOOD_METRICS,
                client=client,
                promote_on_pass=False,
            )

        mock_register.assert_not_called()
        assert result.promoted_version is None


# ---------------------------------------------------------------------------
# Tests: Promotion (better model)
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_better_model_is_promoted(self) -> None:
        """New model with higher PR-AUC should be PROMOTED."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS)

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_version = MagicMock()
            mock_version.version = "2"
            mock_register.return_value = mock_version

            result = run_validation_gate(
                run_id="better-run-id",
                new_metrics=_GOOD_METRICS,  # pr_auc=0.90 > prod pr_auc=0.85
                client=client,
                promote_on_pass=True,
            )

        assert result.status == ValidationStatus.PROMOTED

    def test_promoted_result_has_both_metrics(self) -> None:
        """PROMOTED result should contain both new and production metrics."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS)

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_version = MagicMock()
            mock_version.version = "2"
            mock_register.return_value = mock_version

            result = run_validation_gate(
                run_id="better-run-id",
                new_metrics=_GOOD_METRICS,
                client=client,
            )

        assert result.prod_metrics is not None
        assert result.new_metrics["pr_auc"] == pytest.approx(0.90)
        assert result.prod_metrics["pr_auc"] == pytest.approx(0.85)

    def test_equal_pr_auc_is_promoted(self) -> None:
        """Model with same PR-AUC as production should be PROMOTED (>= not >)."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS)
        # Same PR-AUC as prod
        equal_metrics = {**_GOOD_METRICS, "pr_auc": _PROD_METRICS["pr_auc"]}

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_version = MagicMock()
            mock_version.version = "3"
            mock_register.return_value = mock_version

            result = run_validation_gate(
                run_id="equal-run-id",
                new_metrics=equal_metrics,
                client=client,
            )

        assert result.status == ValidationStatus.PROMOTED


# ---------------------------------------------------------------------------
# Tests: Rejection
# ---------------------------------------------------------------------------


class TestRejection:
    def test_pr_auc_regression_is_rejected(self) -> None:
        """New model with lower PR-AUC than production should be REJECTED."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS)
        worse_metrics = {**_GOOD_METRICS, "pr_auc": 0.70}  # below prod 0.85

        with patch("src.validate.mlflow.register_model") as mock_register:
            result = run_validation_gate(
                run_id="worse-run-id",
                new_metrics=worse_metrics,
                client=client,
                promote_on_pass=True,
            )

        assert result.status == ValidationStatus.REJECTED
        mock_register.assert_not_called()

    def test_recall_below_minimum_is_rejected(self) -> None:
        """Model with Recall < MIN_RECALL must be REJECTED regardless of PR-AUC."""
        client = _make_mock_client(prod_metrics=None)  # Even with no prod model
        low_recall = {**_GOOD_METRICS, "recall": MIN_RECALL - 0.01}

        with patch("src.validate.mlflow.register_model") as mock_register:
            result = run_validation_gate(
                run_id="low-recall-run-id",
                new_metrics=low_recall,
                client=client,
            )

        assert result.status == ValidationStatus.REJECTED
        mock_register.assert_not_called()

    def test_precision_below_minimum_is_rejected(self) -> None:
        """Model with Precision < MIN_PRECISION must be REJECTED."""
        client = _make_mock_client(prod_metrics=None)
        low_precision = {**_GOOD_METRICS, "precision": MIN_PRECISION - 0.01}

        with patch("src.validate.mlflow.register_model") as mock_register:
            result = run_validation_gate(
                run_id="low-precision-run-id",
                new_metrics=low_precision,
                client=client,
            )

        assert result.status == ValidationStatus.REJECTED
        mock_register.assert_not_called()

    def test_rejected_result_has_reason(self) -> None:
        """REJECTED result must include a human-readable reason."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS)
        worse_metrics = {**_GOOD_METRICS, "pr_auc": 0.50}

        with patch("src.validate.mlflow.register_model"):
            result = run_validation_gate(
                run_id="bad-run",
                new_metrics=worse_metrics,
                client=client,
            )

        assert result.status == ValidationStatus.REJECTED
        assert len(result.reason) > 0

    def test_both_threshold_failures_reported(self) -> None:
        """When both recall and precision fail, reason should mention both."""
        client = _make_mock_client(prod_metrics=None)
        bad_metrics = {
            **_GOOD_METRICS,
            "recall": MIN_RECALL - 0.10,
            "precision": MIN_PRECISION - 0.10,
        }

        with patch("src.validate.mlflow.register_model"):
            result = run_validation_gate(
                run_id="double-fail",
                new_metrics=bad_metrics,
                client=client,
            )

        assert result.status == ValidationStatus.REJECTED
        reason_lower = result.reason.lower()
        assert "recall" in reason_lower
        assert "precision" in reason_lower


# ---------------------------------------------------------------------------
# Evaluation-protocol guard
# ---------------------------------------------------------------------------


class TestProtocolChange:
    """A production model from before the leak-free rewrite is not comparable.

    Its bare `pr_auc` was measured on a test set that had already driven early
    stopping, so it sits on a different (inflated) scale from `val_pr_auc`.
    Comparing the two would either block every good candidate or wave through a
    bad one, depending on which way the inflation ran. The gate refuses the
    comparison and resets the baseline instead.
    """

    def test_legacy_production_run_is_treated_as_first_deployment(self) -> None:
        client = _make_mock_client(prod_metrics=_PROD_METRICS, raw_metric_keys=True)

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_register.return_value = MagicMock(version="1")
            result = run_validation_gate(
                run_id="new-run-id",
                new_metrics=_GOOD_METRICS,
                client=client,
                promote_on_pass=True,
            )

        assert result.status == ValidationStatus.FIRST_DEPLOYMENT
        assert result.prod_metrics is None

    def test_legacy_run_does_not_block_a_lower_scoring_candidate(self) -> None:
        """A candidate below the legacy number must not be rejected on it."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS, raw_metric_keys=True)
        below_legacy = {**_GOOD_METRICS, "pr_auc": 0.60}  # under prod's 0.85

        with patch("src.validate.mlflow.register_model") as mock_register:
            mock_register.return_value = MagicMock(version="1")
            result = run_validation_gate(
                run_id="new-run-id",
                new_metrics=below_legacy,
                client=client,
                promote_on_pass=True,
            )

        assert result.status == ValidationStatus.FIRST_DEPLOYMENT

    def test_minimum_thresholds_still_apply_under_protocol_change(self) -> None:
        """Resetting the baseline must not disable the hard recall/precision floor."""
        client = _make_mock_client(prod_metrics=_PROD_METRICS, raw_metric_keys=True)
        weak = {**_GOOD_METRICS, "recall": MIN_RECALL - 0.2, "precision": MIN_PRECISION - 0.2}

        result = run_validation_gate(
            run_id="new-run-id",
            new_metrics=weak,
            client=client,
            promote_on_pass=False,
        )

        assert result.status == ValidationStatus.REJECTED
