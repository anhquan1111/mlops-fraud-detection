"""Giám sát trôi dạt dữ liệu (Evidently) và hiệu năng model theo nhóm thuần tập (Matured Cohort)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DATA_PROCESSED_DIR,
    FEATURE_COLS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from src.features import load_data, preprocess, split_data
from src.quality import quality_gate

logger = logging.getLogger(__name__)

REFERENCE_BASELINE_PATH = DATA_PROCESSED_DIR / "reference_baseline.parquet"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Ghi file JSON nguyên tử (Atomic Write) qua file tạm để tránh đọc file dở dang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Tính mã băm SHA-256 của file theo luồng stream để tránh tràn RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_or_create_reference_baseline(
    sample_size: int = 5000,
    random_state: int = 42,
    force_recreate: bool = False,
    raw_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
) -> pd.DataFrame:
    """Tải hoặc trích xuất bộ dữ liệu tham chiếu chuẩn (Reference Baseline) từ tập Train."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    source = Path(raw_path) if raw_path is not None else RAW_DATA_PATH
    destination = Path(baseline_path) if baseline_path is not None else REFERENCE_BASELINE_PATH
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.exists() and not force_recreate:
        if not metadata_path.exists():
            raise ValueError("Unverified legacy baseline: explicitly recreate from training data")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("origin") != "training_split_raw_features_v1" or metadata.get(
            "baseline_sha256"
        ) != _sha256(destination):
            raise ValueError("Baseline provenance/hash mismatch; investigate before recreating")
        baseline_df = pd.read_parquet(destination)
        quality_gate(baseline_df)
        return baseline_df

    # A missing real source must fail explicitly, never produce plausible-looking fake reports.
    raw_df = load_data(source)
    X, y = preprocess(raw_df)
    X_train = split_data(X, y)[0]  # Reuse the model's split, BEFORE Amount scaling.
    quality_gate(X_train)
    baseline_df = X_train.sample(n=min(sample_size, len(X_train)), random_state=random_state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(destination, index=False)
    _write_json(
        metadata_path,
        {
            "origin": "training_split_raw_features_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": _sha256(source),
            "baseline_sha256": _sha256(destination),
            "split_seed": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "sample_seed": random_state,
            "train_rows": len(X_train),
            "sample_rows": len(baseline_df),
            "features": FEATURE_COLS,
            "amount_scale": "raw",
        },
    )
    logger.info("Saved training reference baseline to %s", destination)
    return baseline_df


def create_feature_dataset(df: pd.DataFrame) -> Dataset:
    """Đóng gói DataFrame thành đối tượng Dataset của Evidently với định nghĩa cột số."""
    return Dataset.from_pandas(
        df[FEATURE_COLS].copy(),
        data_definition=DataDefinition(
            numerical_columns=FEATURE_COLS,
        ),
    )


def _run_drift_analysis(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    output_dir: Path | str = REPORTS_DIR,
    drift_share_threshold: float = 0.3,
) -> dict[str, Any]:
    """Thực thi kiểm tra Quality Gate và phân tích trôi dạt dữ liệu bằng Evidently."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. TRẠM GÁC QUALITY GATE: Chặn dữ liệu rác trước khi đo drift
    issues = quality_gate(current_df, raise_on_error=False)
    if issues:
        result = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": "QUALITY_FAILURE",
            "quality_issues": issues,
            "drift": None,
            "action": "investigate_data_pipeline; drift_not_run",
        }
        result["quality_source"] = "current"
        _write_json(out_path / "drift_summary.json", result)
        return result

    # 2. Chuẩn bị reference baseline
    ref_df = reference_df if reference_df is not None else get_or_create_reference_baseline()
    reference_issues = quality_gate(ref_df, raise_on_error=False)
    if reference_issues:
        result = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": "QUALITY_FAILURE",
            "quality_source": "reference",
            "quality_issues": reference_issues,
            "drift": None,
            "action": "investigate_reference; drift_not_run",
        }
        _write_json(out_path / "drift_summary.json", result)
        return result

    # 3. Cấu hình và chạy Evidently Report
    report = Report(
        [
            DataDriftPreset(
                columns=FEATURE_COLS,
                num_method="ks",
                num_threshold=0.01,
                drift_share=drift_share_threshold,
            )
        ],
        include_tests=True,
    )

    snapshot = report.run(
        current_data=create_feature_dataset(current_df),
        reference_data=create_feature_dataset(ref_df),
    )

    # Lưu báo cáo tương tác HTML và JSON
    snapshot.save_html(str(out_path / "drift.html"))
    snapshot.save_json(str(out_path / "drift.json"))

    # Bóc tách kết quả cốt lõi
    raw = snapshot.dict()
    aggregate = next(
        m["value"] for m in raw["metrics"] if m["config"]["type"].endswith(":DriftedColumnsCount")
    )

    columns: dict[str, dict[str, Any]] = {}
    for m in raw["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith(":ValueDrift"):
            test = next(t for t in raw["tests"] if t["metric_config"]["metric_id"] == m["id"])
            columns[cfg["column"]] = {
                "method": cfg["method"],
                "p_value": round(float(m["value"]), 6),
                "threshold": cfg["threshold"],
                "drift_detected": bool(test["status"] == "FAIL"),
            }

    drift_count = int(aggregate["count"])
    drift_share = float(aggregate["share"])
    is_dataset_drift = bool(drift_share >= drift_share_threshold)

    action = (
        "investigate_feature_drift; review_business_source"
        if is_dataset_drift
        else "no_dataset_feature_drift_detected; performance_not_assessed"
    )

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "SUCCESS",
        "quality_issues": [],
        "reference_rows": len(ref_df),
        "current_rows": len(current_df),
        "drift_share_threshold": drift_share_threshold,
        "amount_scale": "raw",
        "drift": {
            "dataset_drift": is_dataset_drift,
            "drift_count": drift_count,
            "drift_share": round(drift_share, 4),
            "columns": columns,
        },
        "action": action,
    }

    _write_json(out_path / "drift_summary.json", summary)
    return summary


def run_drift_analysis(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    output_dir: Path | str = REPORTS_DIR,
    drift_share_threshold: float = 0.3,
) -> dict[str, Any]:
    """Chạy phân tích trôi dạt dữ liệu và xuất bản kết quả báo cáo mới nhất."""
    summary_path = Path(output_dir) / "drift_summary.json"
    _write_json(
        summary_path,
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": "RUNNING",
            "drift": None,
        },
    )
    try:
        if not 0 < drift_share_threshold <= 1:
            raise ValueError("drift_share_threshold must be in (0, 1]")
        return _run_drift_analysis(current_df, reference_df, output_dir, drift_share_threshold)
    except Exception:
        _write_json(
            summary_path,
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "status": "ERROR",
                "drift": None,
                "action": "inspect_job_logs; drift_unavailable",
            },
        )
        raise


def matured_cohort_audit(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    as_of: pd.Timestamp,
    label_delay_days: int = 7,
) -> dict[str, Any]:
    """Đánh giá hiệu năng model trên các giao dịch đã đủ độ chín (Matured Cohort) có nhãn trễ."""
    if label_delay_days < 0:
        raise ValueError("label_delay_days must be non-negative")
    scored_df = scored_df.copy()
    labels_df = labels_df.copy()
    for name, frame, required, time_column in (
        ("scored", scored_df, ["transaction_id", "event_time", "prediction"], "event_time"),
        (
            "labels",
            labels_df,
            ["transaction_id", "label_available_at", TARGET_COL],
            "label_available_at",
        ),
    ):
        if frame.columns.duplicated().any() or not set(required).issubset(frame.columns):
            raise ValueError(f"Invalid {name} schema: required {required}")
        if frame["transaction_id"].isna().any() or frame["transaction_id"].duplicated().any():
            raise ValueError(f"{name}.transaction_id must be non-null and unique")
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="raise")
        if frame[time_column].isna().any():
            raise ValueError(f"{name}.{time_column} must not contain missing timestamps")
    # Preserve the predictions made at serving time. Never silently truncate 0.9 to label 0.
    if not scored_df["prediction"].isin([0, 1]).all():
        raise ValueError("prediction must contain binary labels 0/1")
    if not labels_df[TARGET_COL].dropna().isin([0, 1]).all():
        raise ValueError("Class must contain binary labels 0/1 or missing labels")
    if "fraud_probability" in scored_df:
        probabilities = scored_df["fraud_probability"]
        if probabilities.isna().any() or not probabilities.between(0, 1).all():
            raise ValueError("fraud_probability must be finite and in [0, 1]")
    as_of = pd.to_datetime(as_of, utc=True, errors="raise")
    if pd.isna(as_of):
        raise ValueError("as_of must be a valid timestamp")
    delay = pd.Timedelta(days=label_delay_days)
    cohort = scored_df.loc[scored_df["event_time"] <= as_of - delay].copy()
    known_labels = labels_df.loc[
        labels_df["label_available_at"] <= as_of,
        ["transaction_id", "label_available_at", TARGET_COL],
    ]
    cohort = cohort.drop(columns=[TARGET_COL, "label_available_at"], errors="ignore")

    joined = cohort.merge(known_labels, on="transaction_id", how="left", validate="one_to_one")

    n_matured = len(cohort)
    observed = joined[TARGET_COL].notna()
    n_observed = int(observed.sum())
    coverage = float(n_observed / n_matured) if n_matured > 0 else 0.0

    result: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "total_scored_predictions": len(scored_df),
        "matured_cohort_size": n_matured,
        "observed_labels_count": n_observed,
        "label_coverage": round(coverage, 4),
        "status": "insufficient_labels",
        "metrics": None,
    }

    # Chống Selection Bias: thiếu nhãn -> không tính điểm
    if n_matured == 0 or not observed.all():
        return result

    y_true = joined[TARGET_COL].astype(int).to_numpy()
    y_pred = joined["prediction"].astype(int).to_numpy()
    y_prob = joined["fraud_probability"].to_numpy() if "fraud_probability" in joined else None

    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())

    metrics: dict[str, Any] = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": round(float((tn + tp) / n_matured), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0.0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0.0)), 6),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    result["status"] = "complete_matured_cohort"
    result["metrics"] = metrics
    return result
