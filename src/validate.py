"""Validation gate — compare new model vs production before promoting.

Workflow:
    1. Load current production model metrics from MLflow Registry.
    2. Compare new model metrics vs production.
    3. Promote if new model is strictly better (or first deployment).

Decision logic:
    - Model MUST meet minimum thresholds: Recall >= 0.80, Precision >= 0.50
    - Model MUST have PR-AUC >= current production PR-AUC (no regression)
    - If no production model exists → FIRST_DEPLOYMENT (auto-promote)

Usage (standalone):
    uv run python src/validate.py --run-id <mlflow_run_id>

Usage (in train.py):
    from src.validate import validate_and_promote
    result = validate_and_promote(client, run_id, new_metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MIN_PRECISION,
    MIN_RECALL,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ValidationStatus(StrEnum):
    """Outcome of the validation gate."""

    PROMOTED = "PROMOTED"  # new model is better → promoted to production
    REJECTED = "REJECTED"  # new model did not pass the gate
    FIRST_DEPLOYMENT = "FIRST_DEPLOYMENT"  # no production model exists → auto-promote


@dataclass
class ValidationResult:
    """Full result returned by the validation gate.

    Attributes:
        status: PROMOTED | REJECTED | FIRST_DEPLOYMENT
        new_metrics: Metrics dict for the candidate model.
        prod_metrics: Metrics dict for the current production model (None if first deploy).
        reason: Human-readable explanation of the decision.
        promoted_version: MLflow model version string if promoted, else None.
    """

    status: ValidationStatus
    new_metrics: dict[str, float]
    prod_metrics: dict[str, float] | None
    reason: str
    promoted_version: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_production_metrics(client: MlflowClient, model_name: str) -> dict[str, float] | None:
    """Fetch metrics of the model currently tagged with alias 'production'.

    Args:
        client: MLflow tracking client.
        model_name: Registered model name in MLflow Registry.

    Returns:
        Dict with pr_auc, recall, precision, f1, roc_auc or None if not found.
    """
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        logger.info("No model with alias 'production' found in registry.")
        return None

    run_id = prod_version.run_id
    if run_id is None:
        logger.warning("Production model version has no associated run_id.")
        return None

    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
    except Exception as exc:
        logger.warning(f"Could not load production run metrics: {exc}")
        return None

    if "pr_auc" not in metrics:
        logger.warning("Production run has no 'pr_auc' metric logged.")
        return None

    return {
        "pr_auc": metrics.get("pr_auc", 0.0),
        "recall": metrics.get("recall", 0.0),
        "precision": metrics.get("precision", 0.0),
        "f1": metrics.get("f1", 0.0),
        "roc_auc": metrics.get("roc_auc", 0.0),
    }


def _check_minimum_thresholds(metrics: dict[str, float]) -> list[str]:
    """Check if metrics meet the project minimum thresholds.

    Args:
        metrics: Dict with at minimum 'recall' and 'precision' keys.

    Returns:
        List of failure reasons (empty list = all thresholds met).
    """
    failures = []
    if metrics.get("recall", 0.0) < MIN_RECALL:
        failures.append(f"Recall {metrics['recall']:.4f} < {MIN_RECALL} (minimum threshold)")
    if metrics.get("precision", 0.0) < MIN_PRECISION:
        failures.append(
            f"Precision {metrics['precision']:.4f} < {MIN_PRECISION} (minimum threshold)"
        )
    return failures


def _promote_model(client: MlflowClient, run_id: str, model_name: str) -> str:
    """Register and promote a model run to production in MLflow Registry.

    Args:
        client: MLflow tracking client.
        run_id: MLflow run ID of the model to promote.
        model_name: Registered model name.

    Returns:
        Model version string that was promoted.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from run {run_id} → {model_name}")
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = model_version.version

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version,
    )
    logger.info(f"✅ Promoted {model_name} v{version} → alias 'production'")
    return str(version)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_validation_gate(
    run_id: str,
    new_metrics: dict[str, float],
    client: MlflowClient | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    promote_on_pass: bool = True,
) -> ValidationResult:
    """Main validation gate: compare new model vs production and optionally promote.

    Args:
        run_id: MLflow run ID of the new candidate model.
        new_metrics: Metrics dict for the candidate (must have pr_auc, recall, precision).
        client: MLflow client (created automatically if None).
        model_name: Registered model name in MLflow Registry.
        promote_on_pass: If True and model passes, register and promote automatically.

    Returns:
        ValidationResult with status, metrics comparison, and reason.
    """
    if client is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

    logger.info("=" * 60)
    logger.info("  VALIDATION GATE")
    logger.info("=" * 60)
    logger.info(
        f"  Candidate: PR-AUC={new_metrics.get('pr_auc', 0):.4f}  "
        f"Recall={new_metrics.get('recall', 0):.4f}  "
        f"Precision={new_metrics.get('precision', 0):.4f}"
    )

    # ------------------------------------------------------------------
    # 1. Check minimum thresholds (hard gate — always enforced)
    # ------------------------------------------------------------------
    threshold_failures = _check_minimum_thresholds(new_metrics)
    if threshold_failures:
        reason = "Failed minimum thresholds: " + "; ".join(threshold_failures)
        logger.warning(f"❌ REJECTED — {reason}")
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            new_metrics=new_metrics,
            prod_metrics=None,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 2. Compare vs production
    # ------------------------------------------------------------------
    prod_metrics = _get_production_metrics(client, model_name)

    if prod_metrics is None:
        # First deployment — no production model exists
        reason = "No production model found — first deployment, auto-promoting."
        logger.info(f"🚀 FIRST_DEPLOYMENT — {reason}")

        promoted_version = None
        if promote_on_pass:
            promoted_version = _promote_model(client, run_id, model_name)

        return ValidationResult(
            status=ValidationStatus.FIRST_DEPLOYMENT,
            new_metrics=new_metrics,
            prod_metrics=None,
            reason=reason,
            promoted_version=promoted_version,
        )

    logger.info(
        f"  Production: PR-AUC={prod_metrics['pr_auc']:.4f}  "
        f"Recall={prod_metrics['recall']:.4f}  "
        f"Precision={prod_metrics['precision']:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. PR-AUC must be >= production (no regression allowed)
    # ------------------------------------------------------------------
    new_pr_auc = new_metrics.get("pr_auc", 0.0)
    prod_pr_auc = prod_metrics.get("pr_auc", 0.0)

    if new_pr_auc < prod_pr_auc:
        reason = (
            f"PR-AUC regression: new={new_pr_auc:.4f} < production={prod_pr_auc:.4f}. "
            "Model must be at least as good as current production."
        )
        logger.warning(f"❌ REJECTED — {reason}")
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            new_metrics=new_metrics,
            prod_metrics=prod_metrics,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 4. All checks passed → promote
    # ------------------------------------------------------------------
    delta = new_pr_auc - prod_pr_auc
    reason = (
        f"New model passes all gates. "
        f"PR-AUC improvement: +{delta:.4f} ({prod_pr_auc:.4f} → {new_pr_auc:.4f}). "
        f"Recall={new_metrics['recall']:.4f} >= {MIN_RECALL}, "
        f"Precision={new_metrics['precision']:.4f} >= {MIN_PRECISION}."
    )
    logger.info(f"✅ PROMOTED — {reason}")

    promoted_version = None
    if promote_on_pass:
        promoted_version = _promote_model(client, run_id, model_name)

    return ValidationResult(
        status=ValidationStatus.PROMOTED,
        new_metrics=new_metrics,
        prod_metrics=prod_metrics,
        reason=reason,
        promoted_version=promoted_version,
    )


def print_validation_report(result: ValidationResult) -> None:
    """Pretty-print the validation gate result.

    Args:
        result: ValidationResult from run_validation_gate().
    """
    status_icon = {
        ValidationStatus.PROMOTED: "✅",
        ValidationStatus.REJECTED: "❌",
        ValidationStatus.FIRST_DEPLOYMENT: "🚀",
    }.get(result.status, "?")

    print(f"\n{'=' * 60}")
    print("  VALIDATION GATE REPORT")
    print(f"{'=' * 60}")
    print(f"  Status  : {status_icon} {result.status.value}")
    print(f"  Reason  : {result.reason}")
    print()
    print(f"  {'Metric':<18} {'New':>10} {'Production':>12}")
    print(f"  {'-' * 44}")

    metrics_to_show = ["pr_auc", "recall", "precision", "f1", "roc_auc"]
    for m in metrics_to_show:
        new_val = result.new_metrics.get(m, float("nan"))
        prod_val = result.prod_metrics.get(m, float("nan")) if result.prod_metrics else float("nan")
        prod_str = f"{prod_val:>12.4f}" if result.prod_metrics else f"{'N/A':>12}"
        print(f"  {m:<18} {new_val:>10.4f} {prod_str}")

    if result.promoted_version:
        print()
        print(f"  Promoted version: {result.promoted_version}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Validation gate: compare new model vs production."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID of the candidate model.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check metrics without actually promoting the model.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _client = MlflowClient()

    # Load metrics from MLflow run
    try:
        _run = _client.get_run(args.run_id)
        _metrics = {
            k: v
            for k, v in _run.data.metrics.items()
            if k in ("pr_auc", "recall", "precision", "f1", "roc_auc")
        }
    except Exception as e:
        print(f"ERROR: Could not load run '{args.run_id}': {e}")
        sys.exit(1)

    if "pr_auc" not in _metrics:
        print(f"ERROR: Run '{args.run_id}' has no 'pr_auc' metric. Run eval first.")
        sys.exit(1)

    _result = run_validation_gate(
        run_id=args.run_id,
        new_metrics=_metrics,
        client=_client,
        promote_on_pass=not args.dry_run,
    )
    print_validation_report(_result)

    if _result.status == ValidationStatus.REJECTED:
        sys.exit(1)
