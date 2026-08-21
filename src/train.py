"""Training pipeline for the fraud detection models.

Usage:
    uv run python src/train.py

Pipeline:
    1. Load & preprocess data (src/features.py)
    2. Stratified 64/16/20 train/val/test split — test carved out FIRST
    3. Fit the Amount scaler on TRAIN ONLY, apply to val and test
    4. Train Logistic Regression baseline  -> MLflow run
    5. Train XGBoost grid (3 configs)      -> MLflow runs
    6. Train LightGBM grid (3 configs)     -> MLflow runs
    7. Print summary table of all runs

Total: 1 baseline + 3 XGBoost + 3 LightGBM = 7 runs.

Evaluation protocol (see docs/leakage_fix.md):
    - Early stopping uses the VALIDATION set. Using test here would let the test
      set choose the number of boosting rounds.
    - Champion selection uses `val_pr_auc` (scripts/select_best_model.py).
    - The TEST set is scored once per run purely for the final report and is
      never consulted for any decision.
    Both metric families are logged: `val_*` and `test_*`.
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
    TEST_SIZE,
    VAL_SIZE,
    XGBOOST_BASE_PARAMS,
    XGBOOST_GRID,
)
from src.evaluate import evaluate_model, print_report, save_pr_curve
from src.features import (
    apply_amount_scaler,
    fit_amount_scaler,
    load_data,
    preprocess,
    split_data,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
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


def _log_split_params(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Record the split shape on the active MLflow run, for reproducibility."""
    mlflow.log_param("threshold", DECISION_THRESHOLD)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("val_size", VAL_SIZE)
    mlflow.log_param("stratified_split", True)
    mlflow.log_param("split_protocol", "train64/val16/test20")
    mlflow.log_param("n_train_samples", len(X_train))
    mlflow.log_param("n_val_samples", len(X_val))
    mlflow.log_param("n_test_samples", len(X_test))
    mlflow.log_param("n_fraud_train", int(y_train.sum()))
    mlflow.log_param("n_fraud_val", int(y_val.sum()))
    mlflow.log_param("n_fraud_test", int(y_test.sum()))


def _evaluate_and_log(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    display_name: str,
) -> dict[str, float]:
    """Score on val and test, log both metric families, return the merged dict.

    `val_*` drives early stopping, the validation gate and champion selection.
    `test_*` is reported only — nothing in the pipeline branches on it.

    Args:
        model: Fitted classifier.
        X_val / y_val: Validation split.
        X_test / y_test: Held-out test split.
        display_name: Label used in the printed report.

    Returns:
        Dict with `val_*` and `test_*` keys (plus threshold).
    """
    val_metrics = evaluate_model(model, X_val, y_val, threshold=DECISION_THRESHOLD)
    test_metrics = evaluate_model(model, X_test, y_test, threshold=DECISION_THRESHOLD)

    print_report(val_metrics, model_name=f"{display_name} [VAL - used for selection]")
    print_report(test_metrics, model_name=f"{display_name} [TEST - report only]")

    merged: dict[str, float] = {"threshold": DECISION_THRESHOLD}
    for key, value in val_metrics.items():
        if key != "threshold":
            merged[f"val_{key}"] = value
    for key, value in test_metrics.items():
        if key != "threshold":
            merged[f"test_{key}"] = value

    mlflow.log_metrics(merged)
    return merged


# ---------------------------------------------------------------------------
# Baseline: Logistic Regression
# ---------------------------------------------------------------------------


def train_baseline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Train Logistic Regression baseline and log to MLflow.

    Args:
        X_train / X_val / X_test: Feature splits (Amount already scaled).
        y_train / y_val / y_test: Label splits.

    Returns:
        Dictionary of val_* and test_* evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("Training: Logistic Regression (Baseline)")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="lr_baseline") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("purpose", "baseline")
        mlflow.set_tag("protocol", "leakfree_v2")

        mlflow.log_params(LR_PARAMS)
        _log_split_params(X_train, X_val, X_test, y_train, y_val, y_test)

        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_train, y_train)

        metrics = _evaluate_and_log(
            model, X_val, y_val, X_test, y_test, "Logistic Regression (Baseline)"
        )

        pr_curve_path = save_pr_curve(model, X_test, y_test, model_name="LR Baseline")
        if pr_curve_path and Path(pr_curve_path).exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        logger.info(f"[OK] Baseline run logged: {run_id}")

    return metrics


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    grid_params: dict,
    scale_pos_weight: float,
) -> dict[str, float]:
    """Train one XGBoost run with given params and log to MLflow.

    Early stopping watches the VALIDATION split (eval_metric 'aucpr' is
    XGBoost's name for average precision / PR-AUC).

    Args:
        X_train / X_val / X_test: Feature splits.
        y_train / y_val / y_test: Label splits.
        grid_params: Experiment-specific params from XGBOOST_GRID (includes run_name).
        scale_pos_weight: Class imbalance ratio (n_neg / n_pos), computed on train.

    Returns:
        Dictionary of val_* and test_* evaluation metrics.
    """
    run_name = grid_params.pop("run_name", "xgboost")
    logger.info("=" * 60)
    logger.info(f"Training: XGBoost -- {run_name}")
    logger.info("=" * 60)

    # Merge base + grid params, add scale_pos_weight
    params = {**XGBOOST_BASE_PARAMS, **grid_params, "scale_pos_weight": scale_pos_weight}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("purpose", "experiment")
        mlflow.set_tag("protocol", "leakfree_v2")

        mlflow.log_params(params)
        mlflow.log_param("imbalance_strategy", "scale_pos_weight")
        mlflow.log_param("early_stopping_split", "val")
        _log_split_params(X_train, X_val, X_test, y_train, y_val, y_test)

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],  # VAL, never test
            verbose=False,
        )
        logger.info(f"XGBoost training complete (best iteration: {model.best_iteration})")

        metrics = _evaluate_and_log(model, X_val, y_val, X_test, y_test, f"XGBoost ({run_name})")
        best_iter = (
            model.best_iteration if model.best_iteration is not None else params["n_estimators"]
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

        logger.info(f"[OK] XGBoost run logged: {run_id}")

    return metrics


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    grid_params: dict,
) -> dict[str, float]:
    """Train one LightGBM run with given params and log to MLflow.

    Uses class_weight='balanced' (from LIGHTGBM_BASE_PARAMS) to handle imbalance.
    scale_pos_weight is NOT used for LGBM — at ratio ~577 it destabilises splits.
    Early stopping watches the VALIDATION split.

    Args:
        X_train / X_val / X_test: Feature splits.
        y_train / y_val / y_test: Label splits.
        grid_params: Experiment-specific params from LIGHTGBM_GRID (includes run_name).

    Returns:
        Dictionary of val_* and test_* evaluation metrics.
    """
    import lightgbm as lgb

    run_name = grid_params.pop("run_name", "lightgbm")
    logger.info("=" * 60)
    logger.info(f"Training: LightGBM -- {run_name}")
    logger.info("=" * 60)

    # Merge base + grid params (no scale_pos_weight — LGBM uses class_weight='balanced')
    params = {**LIGHTGBM_BASE_PARAMS, **grid_params}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("purpose", "experiment")
        mlflow.set_tag("protocol", "leakfree_v2")

        mlflow.log_params(params)
        mlflow.log_param("imbalance_strategy", "class_weight_balanced")
        mlflow.log_param("early_stopping_split", "val")
        _log_split_params(X_train, X_val, X_test, y_train, y_val, y_test)

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],  # VAL, never test
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),  # suppress per-iteration logs
            ],
        )
        best_iter = (
            model.best_iteration_ if model.best_iteration_ is not None else params["n_estimators"]
        )
        logger.info(f"LightGBM training complete (best_iteration={best_iter})")
        mlflow.log_metric("best_iteration", best_iter)

        metrics = _evaluate_and_log(model, X_val, y_val, X_test, y_test, f"LightGBM ({run_name})")

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
    """Print a comparison table of all runs, ranked by VALIDATION PR-AUC.

    The ranking column is deliberately `val_pr_auc`: ranking by test PR-AUC would
    reintroduce exactly the selection leak this pipeline was rewritten to remove.

    Args:
        results: List of dicts with run_name, model_type and val_*/test_* metrics.
    """
    print("\n" + "=" * 96)
    print("  EXPERIMENT SUMMARY -- All Runs (ranked by VAL PR-AUC; test shown for reference only)")
    print("=" * 96)
    header = (
        f"  {'Run':<20} {'Model':<20} {'VAL':>7} {'VAL':>7} {'VAL':>7} "
        f"{'TEST':>7} {'TEST':>7} {'TEST':>7}"
    )
    print(header)
    print(
        f"  {'':<20} {'':<20} {'PR-AUC':>7} {'Recall':>7} {'Prec':>7} "
        f"{'PR-AUC':>7} {'Recall':>7} {'Prec':>7}"
    )
    print(f"  {'-' * 92}")

    sorted_results = sorted(results, key=lambda x: x["val_pr_auc"], reverse=True)
    for i, r in enumerate(sorted_results):
        marker = "  <- BEST (by val)" if i == 0 else ""
        recall_ok = "OK" if r["val_recall"] >= MIN_RECALL else "!!"
        prec_ok = "OK" if r["val_precision"] >= MIN_PRECISION else "!!"
        print(
            f"  {r['run_name']:<20} {r['model_type']:<20} "
            f"{r['val_pr_auc']:>7.4f} {r['val_recall']:>5.4f}[{recall_ok}] "
            f"{r['val_precision']:>5.4f}[{prec_ok}] "
            f"{r['test_pr_auc']:>7.4f} {r['test_recall']:>7.4f} {r['test_precision']:>7.4f}"
            f"{marker}"
        )
    print("=" * 96)

    # Baseline reference is read from the run itself — never hard-coded, so it
    # stays correct whenever the split, the seed or the data changes.
    baseline = next((r for r in results if r["run_name"] == "lr_baseline"), None)
    if baseline is not None:
        print(
            f"\n  Baseline reference (LR, from this run): "
            f"val PR-AUC={baseline['val_pr_auc']:.4f}, "
            f"val Recall={baseline['val_recall']:.4f}, "
            f"val Precision={baseline['val_precision']:.4f}"
        )
        print(
            f"  Baseline on test (report only):          "
            f"test PR-AUC={baseline['test_pr_auc']:.4f}, "
            f"test Recall={baseline['test_recall']:.4f}, "
            f"test Precision={baseline['test_precision']:.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full experiment: baseline + XGBoost grid + LightGBM grid."""
    logger.info("[>>] Multi-model fraud detection experiment (leak-free protocol)")

    # ------------------------------------------------------------------
    # 1. Load, preprocess, split -- SPLIT BEFORE ANY FITTED TRANSFORM
    # ------------------------------------------------------------------
    logger.info("Loading and preprocessing data ...")
    df = load_data(RAW_DATA_PATH)
    X, y = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # 2. Fit the Amount scaler on TRAIN ONLY, then apply to every split
    # ------------------------------------------------------------------
    scaler = fit_amount_scaler(X_train)
    X_train = apply_amount_scaler(scaler, X_train)
    X_val = apply_amount_scaler(scaler, X_val)
    X_test = apply_amount_scaler(scaler, X_test)
    logger.info(
        "[!!] Copy these into src/config.py so the API scales identically: "
        f"AMOUNT_MEAN = {scaler.mean_[0]:.4f}, AMOUNT_STD = {scaler.scale_[0]:.4f}"
    )

    scale_pos_weight = _compute_scale_pos_weight(y_train)

    # ------------------------------------------------------------------
    # 3. MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: list[dict] = []
    splits = (X_train, X_val, X_test, y_train, y_val, y_test)

    # ------------------------------------------------------------------
    # 4. Baseline: Logistic Regression
    # ------------------------------------------------------------------
    metrics = train_baseline(*splits)
    results.append({"run_name": "lr_baseline", "model_type": "logistic_regression", **metrics})

    # ------------------------------------------------------------------
    # 5. XGBoost grid (3 runs)
    # ------------------------------------------------------------------
    logger.info("Starting XGBoost experiment grid (3 runs) ...")
    import copy

    for grid_params in copy.deepcopy(XGBOOST_GRID):
        run_name = grid_params.get("run_name", "xgboost")
        metrics = train_xgboost(*splits, grid_params, scale_pos_weight)
        results.append({"run_name": run_name, "model_type": "xgboost", **metrics})

    # ------------------------------------------------------------------
    # 6. LightGBM grid (3 runs)
    # ------------------------------------------------------------------
    logger.info("Starting LightGBM experiment grid (3 runs) ...")
    for grid_params in copy.deepcopy(LIGHTGBM_GRID):
        run_name = grid_params.get("run_name", "lightgbm")
        metrics = train_lightgbm(*splits, grid_params)
        results.append({"run_name": run_name, "model_type": "lightgbm", **metrics})

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    _print_summary(results)

    # ------------------------------------------------------------------
    # 8. Exit check -- at least one model must clear the thresholds ON VAL
    # ------------------------------------------------------------------
    passing = [
        r for r in results if r["val_recall"] >= MIN_RECALL and r["val_precision"] >= MIN_PRECISION
    ]
    if not passing:
        logger.error(
            "[!!] No model met minimum thresholds on val "
            f"(Recall >= {MIN_RECALL}, Precision >= {MIN_PRECISION})"
        )
        sys.exit(1)

    best = max(passing, key=lambda x: x["val_pr_auc"])
    logger.info(
        f"[OK] Best qualifying model (by val): {best['run_name']} "
        f"(val PR-AUC={best['val_pr_auc']:.4f}, val Recall={best['val_recall']:.4f}, "
        f"val Precision={best['val_precision']:.4f})"
    )
    logger.info("[>>] Run: uv run python scripts/select_best_model.py  to register best model.")


if __name__ == "__main__":
    main()
