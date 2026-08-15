"""Training pipeline for the fraud detection baseline model.

Usage:
    uv run python src/train.py

Pipeline:
    1. Load & preprocess data (src/features.py)
    2. Train Logistic Regression baseline (class_weight='balanced')
    3. Evaluate on held-out test set (src/evaluate.py)
    4. Log params, metrics, and model artifact to MLflow
    5. Print evaluation summary
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from src.config import (
    DECISION_THRESHOLD,
    LR_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
)
from src.evaluate import evaluate_model, print_report, save_pr_curve
from src.features import load_data, preprocess, split_data

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_baseline(data_path: Path = RAW_DATA_PATH) -> dict[str, float]:
    """Train Logistic Regression baseline and log to MLflow.

    Args:
        data_path: Path to creditcard.csv.

    Returns:
        Dictionary of evaluation metrics.
    """
    # ------------------------------------------------------------------
    # 1. Load & preprocess data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("=" * 60)

    df = load_data(data_path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # 2. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # 3. Train + evaluate + log inside a single MLflow run
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Training Logistic Regression baseline")
    logger.info("=" * 60)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Tags — for easy filtering in MLflow UI
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("purpose", "baseline")
        mlflow.set_tag("session", "2")

        # Log hyperparameters
        mlflow.log_params(LR_PARAMS)
        mlflow.log_param("threshold", DECISION_THRESHOLD)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("stratified_split", True)
        mlflow.log_param("time_feature_dropped", True)
        mlflow.log_param("amount_scaled", True)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_fraud_train", int(y_train.sum()))
        mlflow.log_param("n_fraud_test", int(y_test.sum()))

        # Train model
        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_train, y_train)
        logger.info("Training complete.")

        # Evaluate
        logger.info("=" * 60)
        logger.info("STEP 3: Evaluating on test set")
        logger.info("=" * 60)

        metrics = evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD)
        print_report(metrics, model_name="Logistic Regression (Baseline)")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save PR curve and log as artifact
        pr_curve_path = save_pr_curve(
            model, X_test, y_test, model_name="Logistic Regression Baseline"
        )
        if pr_curve_path and Path(pr_curve_path).exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,  # Do NOT auto-register baseline; manual promotion only
        )

        logger.info("=" * 60)
        logger.info(f"✅ MLflow run logged: {run_id}")
        logger.info("   View at: http://localhost:5000  (run: uv run mlflow ui)")
        logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for training pipeline."""
    logger.info("🚀 Starting fraud detection baseline training pipeline")
    metrics = train_baseline()

    # Exit with non-zero if thresholds not met (useful for CI)
    from src.config import MIN_PRECISION, MIN_RECALL

    if metrics["recall"] < MIN_RECALL or metrics["precision"] < MIN_PRECISION:
        logger.warning("⚠️  Model does not meet minimum thresholds — check metrics above.")
        sys.exit(1)

    logger.info("✅ Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
