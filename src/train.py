"""Pipeline huấn luyện mô hình Fraud Detection.

Quy trình chuẩn chống rò rỉ dữ liệu (Leak-free):
1. Nạp và tiền xử lý dữ liệu (src/features.py)
2. Chia 3 tập Stratified 64/16/20 (Train / Val / Test)
3. Fit Amount scaler DUY NHẤT trên tập Train
4. Huấn luyện Baseline (Logistic Regression)
5. Huấn luyện Grid Search XGBoost (3 configs) kèm Early Stopping trên Val
6. Huấn luyện Grid Search LightGBM (3 configs) kèm Early Stopping trên Val
7. Log metrics, biểu đồ PR-curve và artifact lên MLflow
8. Xếp hạng model dựa trên Validation PR-AUC
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
# Cấu hình Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hàm trợ giúp
# ---------------------------------------------------------------------------


def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Tính tỷ lệ mẫu âm / dương (scale_pos_weight) để cân bằng trọng số cho XGBoost."""
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
    """Ghi log kích thước và số ca gian lận từng tập (Train/Val/Test) lên MLflow."""
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
    """Đánh giá mô hình trên tập Val và Test, sau đó log toàn bộ metrics lên MLflow."""
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
# Huấn luyện Baseline: Logistic Regression
# ---------------------------------------------------------------------------


def train_baseline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Huấn luyện mô hình cơ sở Logistic Regression và log artifact lên MLflow."""
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
# Huấn luyện XGBoost
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
    """Huấn luyện XGBoost với Early Stopping trên tập Val và log artifact lên MLflow."""
    run_name = grid_params.pop("run_name", "xgboost")
    logger.info("=" * 60)
    logger.info(f"Training: XGBoost -- {run_name}")
    logger.info("=" * 60)

    # Gộp tham số cấu hình và bổ sung scale_pos_weight
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
            eval_set=[(X_val, y_val)],  # Giám sát trên tập Val, không dùng tập Test
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
# Huấn luyện LightGBM
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
    """Huấn luyện LightGBM với class_weight='balanced' và Early Stopping trên Val."""
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
            eval_set=[(X_val, y_val)],  # Giám sát trên tập Val, không dùng tập Test
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),  # Tắt log chi tiết từng vòng lặp
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
# Bảng tổng kết thực nghiệm
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    """In bảng tổng hợp so sánh tất cả các run, sắp xếp theo Validation PR-AUC."""
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
# Điểm khởi chạy chính
# ---------------------------------------------------------------------------


def main() -> None:
    """Chạy toàn bộ pipeline: Baseline LR -> Grid XGBoost -> Grid LightGBM."""
    logger.info("[>>] Multi-model fraud detection experiment (leak-free protocol)")

    # ------------------------------------------------------------------
    # 1. Nạp, tiền xử lý và chia dữ liệu (chia tập trước khi fit scaler)
    # ------------------------------------------------------------------
    logger.info("Loading and preprocessing data ...")
    df = load_data(RAW_DATA_PATH)
    X, y = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # 2. Fit Amount scaler DUY NHẤT trên tập Train, sau đó áp dụng cho cả 3 tập
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
    # 3. Thiết lập kết nối MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: list[dict] = []
    splits = (X_train, X_val, X_test, y_train, y_val, y_test)

    # ------------------------------------------------------------------
    # 4. Huấn luyện Baseline Logistic Regression
    # ------------------------------------------------------------------
    metrics = train_baseline(*splits)
    results.append({"run_name": "lr_baseline", "model_type": "logistic_regression", **metrics})

    # ------------------------------------------------------------------
    # 5. Huấn luyện Grid Search XGBoost (3 cấu hình)
    # ------------------------------------------------------------------
    logger.info("Starting XGBoost experiment grid (3 runs) ...")
    import copy

    for grid_params in copy.deepcopy(XGBOOST_GRID):
        run_name = grid_params.get("run_name", "xgboost")
        metrics = train_xgboost(*splits, grid_params, scale_pos_weight)
        results.append({"run_name": run_name, "model_type": "xgboost", **metrics})

    # ------------------------------------------------------------------
    # 6. Huấn luyện Grid Search LightGBM (3 cấu hình)
    # ------------------------------------------------------------------
    logger.info("Starting LightGBM experiment grid (3 runs) ...")
    for grid_params in copy.deepcopy(LIGHTGBM_GRID):
        run_name = grid_params.get("run_name", "lightgbm")
        metrics = train_lightgbm(*splits, grid_params)
        results.append({"run_name": run_name, "model_type": "lightgbm", **metrics})

    # ------------------------------------------------------------------
    # 7. In bảng tổng kết thực nghiệm
    # ------------------------------------------------------------------
    _print_summary(results)

    # ------------------------------------------------------------------
    # 8. Kiểm tra điều kiện tối thiểu trên tập Val để chọn Best Model
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
