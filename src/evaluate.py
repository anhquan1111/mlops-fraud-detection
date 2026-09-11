"""Đánh giá hiệu năng mô hình Fraud Detection.

Thước đo chính: PR-AUC (Precision-Recall Area Under Curve).
Thước đo phụ: Recall, Precision, F1-score, ROC-AUC.
LƯU Ý: Không dùng Accuracy vì dữ liệu mất cân bằng nặng (0.17% fraud).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DECISION_THRESHOLD, FIGURES_DIR, MIN_PRECISION, MIN_RECALL

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    # Lấy xác suất dự đoán nhãn 1 (Fraud) và phân loại theo ngưỡng threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Đếm số lượng thực tế: TP (bắt đúng), FP (báo nhầm), FN (bỏ lọt), TN (đúng người tốt)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    return metrics


def print_report(metrics: dict[str, float], model_name: str = "Model") -> None:
    # So khớp với ngưỡng tối thiểu: Recall >= 0.80 và Precision >= 0.50
    recall_ok = metrics["recall"] >= MIN_RECALL
    precision_ok = metrics["precision"] >= MIN_PRECISION

    recall_icon = "[OK] " if recall_ok else "[!!]"
    precision_icon = "[OK] " if precision_ok else "[!!]"

    print(f"\n{'=' * 55}")
    print(f"  Evaluation Report -- {model_name}")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<20} {'Value':>8}   {'Status'}")
    print(f"  {'-' * 45}")
    print(f"  {'PR-AUC (primary)':<20} {metrics['pr_auc']:>8.4f}   (higher = better)")
    print(f"  {'Recall':<20} {metrics['recall']:>8.4f}   {recall_icon} (target >= {MIN_RECALL})")
    prec_line = (
        f"  {'Precision':<20} {metrics['precision']:>8.4f}"
        f"   {precision_icon} (target >= {MIN_PRECISION})"
    )
    print(prec_line)
    print(f"  {'F1-score':<20} {metrics['f1']:>8.4f}")
    print(f"  {'ROC-AUC (ref)':<20} {metrics['roc_auc']:>8.4f}   (reference only)")
    print(f"  {'Threshold':<20} {metrics['threshold']:>8.2f}")
    print(f"{'=' * 55}")

    if recall_ok and precision_ok:
        print("  [PASS] Model meets minimum thresholds!")
    else:
        fails = []
        if not recall_ok:
            fails.append(f"Recall {metrics['recall']:.4f} < {MIN_RECALL}")
        if not precision_ok:
            fails.append(f"Precision {metrics['precision']:.4f} < {MIN_PRECISION}")
        print(f"  [WARN] Below threshold: {', '.join(fails)}")
    print()


def save_pr_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    output_dir: Path = FIGURES_DIR,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping PR curve plot.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Đường cơ sở ngẫu nhiên = tỷ lệ ca gian lận trong tập test (~0.0017)
    baseline = float(np.mean(y_test))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"PR curve saved to {output_path}")
    return output_path
