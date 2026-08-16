"""Training pipeline for the fraud detection models.

Usage:
    uv run python src/train.py

Pipeline:
    1. Load & preprocess data (src/features.py)
    2. Train Logistic Regression baseline  → MLflow run (session 2 reference)
    3. Train XGBoost grid (3 configs)      → MLflow runs
    4. Train LightGBM grid (3 configs)     → MLflow runs
    5. Print summary table of all runs

Total: 1 baseline + 3 XGBoost + 3 LightGBM = 7 runs.
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    DECISION_THRESHOLD,
    LIGHTGBM_BASE_PARAMS,
    LIGHTGBM_GRID,
    LR_PARAMS,
    MIN_PRECISION,
    MIN_RECALL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    XGBOOST_BASE_PARAMS,
    XGBOOST_GRID,
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
# Helper: compute scale_pos_weight for XGBoost
# ---------------------------------------------------------------------------


def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute scale_pos_weight = n_negative / n_positive for XGBoost.

    Args:
        y_train: Training labels (0 = legit, 1 = fraud).

    Returns:
        Float ratio used to handle class imbalance in XGBoost.
    """
    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    ratio = n_negative / n_positive
    logger.info(f"scale_pos_weight = {n_negative:,} / {n_positive:,} = {ratio:.2f}")
    return ratio


# ---------------------------------------------------------------------------
# Baseline: Logistic Regression
# ---------------------------------------------------------------------------


def train_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Train Logistic Regression baseline and log to MLflow.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("Training: Logistic Regression (Baseline)")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="lr_baseline") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("purpose", "baseline")
        mlflow.set_tag("session", "4")

        mlflow.log_params(LR_PARAMS)
        mlflow.log_param("threshold", DECISION_THRESHOLD)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("stratified_split", True)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_fraud_train", int(y_train.sum()))
        mlflow.log_param("n_fraud_test", int(y_test.sum()))

        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD)
        print_report(metrics, model_name="Logistic Regression (Baseline)")
        mlflow.log_metrics(metrics)

        pr_curve_path = save_pr_curve(model, X_test, y_test, model_name="LR Baseline")
        if pr_curve_path and Path(pr_curve_path).exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        logger.info(f"✅ Baseline run logged: {run_id}")

    return metrics


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    grid_params: dict,
    scale_pos_weight: float,
) -> dict[str, float]:
    """Train one XGBoost run with given params and log to MLflow.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        grid_params: Experiment-specific params from XGBOOST_GRID (includes run_name).
        scale_pos_weight: Class imbalance ratio (n_neg / n_pos).

    Returns:
        Dictionary of evaluation metrics.
    """
    run_name = grid_params.pop("run_name", "xgboost")
    logger.info("=" * 60)
    logger.info(f"Training: XGBoost — {run_name}")
    logger.info("=" * 60)

    # Merge base + grid params, add scale_pos_weight
    params = {**XGBOOST_BASE_PARAMS, **grid_params, "scale_pos_weight": scale_pos_weight}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("purpose", "experiment")
        mlflow.set_tag("session", "4")

        mlflow.log_params(params)
        mlflow.log_param("threshold", DECISION_THRESHOLD)
        mlflow.log_param("imbalance_strategy", "scale_pos_weight")
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_fraud_train", int(y_train.sum()))
        mlflow.log_param("n_fraud_test", int(y_test.sum()))

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        logger.info(f"XGBoost training complete (best iteration: {model.best_iteration})")

        metrics = evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD)
        print_report(metrics, model_name=f"XGBoost ({run_name})")
        mlflow.log_metrics(metrics)
        best_iter = (
            model.best_iteration if model.best_iteration is not None
            else params["n_estimators"]
        )
        mlflow.log_metric("best_iteration", best_iter)

        pr_curve_path = save_pr_curve(model, X_test, y_test, model_name=f"XGBoost {run_name}")
        if pr_curve_path and Path(pr_curve_path).exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        logger.info(f"✅ XGBoost run logged: {run_id}")

    return metrics


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    grid_params: dict,
    scale_pos_weight: float,
) -> dict[str, float]:
    """Train one LightGBM run with given params and log to MLflow.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        grid_params: Experiment-specific params from LIGHTGBM_GRID (includes run_name).
        scale_pos_weight: Class imbalance ratio (n_neg / n_pos) — same as XGBoost.

    Returns:
        Dictionary of evaluation metrics.
    """
    run_name = grid_params.pop("run_name", "lightgbm")
    logger.info("=" * 60)
    logger.info(f"Training: LightGBM -- {run_name}")
    logger.info("=" * 60)

    # Merge base + grid params + scale_pos_weight
    params = {**LIGHTGBM_BASE_PARAMS, **grid_params, "scale_pos_weight": scale_pos_weight}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("purpose", "experiment")
        mlflow.set_tag("session", "4")

        mlflow.log_params(params)
        mlflow.log_param("threshold", DECISION_THRESHOLD)
        mlflow.log_param("imbalance_strategy", "scale_pos_weight")
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_fraud_train", int(y_train.sum()))
        mlflow.log_param("n_fraud_test", int(y_test.sum()))

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM training complete")

        metrics = evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD)
        print_report(metrics, model_name=f"LightGBM ({run_name})")
        mlflow.log_metrics(metrics)

        pr_curve_path = save_pr_curve(model, X_test, y_test, model_name=f"LightGBM {run_name}")
        if pr_curve_path and Path(pr_curve_path).exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        logger.info(f"[OK] LightGBM run logged: {run_id}")

    return metrics


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    """Print a comparison table of all runs.

    Args:
        results: List of dicts with keys: run_name, model_type, pr_auc, recall, precision, f1.
    """
    print("\n" + "=" * 75)
    print("  EXPERIMENT SUMMARY — All Runs (sorted by PR-AUC)")
    print("=" * 75)
    print(f"  {'Run':<22} {'Model':<20} {'PR-AUC':>7} {'Recall':>7} {'Prec':>7} {'F1':>7}")
    print(f"  {'-' * 70}")

    sorted_results = sorted(results, key=lambda x: x["pr_auc"], reverse=True)
    for i, r in enumerate(sorted_results):
        marker = " <- BEST" if i == 0 else ""
        recall_ok = "OK" if r["recall"] >= MIN_RECALL else "!!"
        prec_ok = "OK" if r["precision"] >= MIN_PRECISION else "!!"
        print(
            f"  {r['run_name']:<22} {r['model_type']:<20} "
            f"{r['pr_auc']:>7.4f} {r['recall']:>6.4f}[{recall_ok}] "
            f"{r['precision']:>6.4f}[{prec_ok}] {r['f1']:>7.4f}{marker}"
        )
    print("=" * 75)
    print("\n  Baseline reference: LR — PR-AUC=0.7156, Recall=0.9184, Precision=0.8571")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full experiment: baseline + XGBoost grid + LightGBM grid."""
    logger.info("🚀 Session 4: Multi-model fraud detection experiment")

    # ------------------------------------------------------------------
    # 1. Load & preprocess data (shared across all runs)
    # ------------------------------------------------------------------
    logger.info("Loading and preprocessing data ...")
    df = load_data(RAW_DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scale_pos_weight = _compute_scale_pos_weight(y_train)

    # ------------------------------------------------------------------
    # 2. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: list[dict] = []

    # ------------------------------------------------------------------
    # 3. Baseline: Logistic Regression
    # ------------------------------------------------------------------
    metrics = train_baseline(X_train, X_test, y_train, y_test)
    results.append(
        {
            "run_name": "lr_baseline",
            "model_type": "logistic_regression",
            **metrics,
        }
    )

    # ------------------------------------------------------------------
    # 4. XGBoost grid (3 runs)
    # ------------------------------------------------------------------
    logger.info("Starting XGBoost experiment grid (3 runs) ...")
    import copy

    for grid_params in copy.deepcopy(XGBOOST_GRID):
        run_name = grid_params.get("run_name", "xgboost")
        metrics = train_xgboost(X_train, X_test, y_train, y_test, grid_params, scale_pos_weight)
        results.append(
            {
                "run_name": run_name,
                "model_type": "xgboost",
                **metrics,
            }
        )

    # ------------------------------------------------------------------
    # 5. LightGBM grid (3 runs)
    # ------------------------------------------------------------------
    logger.info("Starting LightGBM experiment grid (3 runs) ...")
    for grid_params in copy.deepcopy(LIGHTGBM_GRID):
        run_name = grid_params.get("run_name", "lightgbm")
        metrics = train_lightgbm(X_train, X_test, y_train, y_test, grid_params, scale_pos_weight)
        results.append(
            {
                "run_name": run_name,
                "model_type": "lightgbm",
                **metrics,
            }
        )

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    _print_summary(results)

    # ------------------------------------------------------------------
    # 7. Exit check — at least one model must beat minimum thresholds
    # ------------------------------------------------------------------
    passing = [r for r in results if r["recall"] >= MIN_RECALL and r["precision"] >= MIN_PRECISION]
    if not passing:
        logger.error(
            "❌ No model met minimum thresholds "
            f"(Recall >= {MIN_RECALL}, Precision >= {MIN_PRECISION})"
        )
        sys.exit(1)

    best = max(passing, key=lambda x: x["pr_auc"])
    logger.info(
        f"[OK] Best qualifying model: {best['run_name']} "
        f"(PR-AUC={best['pr_auc']:.4f}, Recall={best['recall']:.4f}, "
        f"Precision={best['precision']:.4f})"
    )
    logger.info("[>>] Run: uv run python scripts/select_best_model.py  to register best model.")


if __name__ == "__main__":
    main()
