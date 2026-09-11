"""Luồng xử lý dữ liệu chuẩn chống rò rỉ (Leak-free Pipeline):

1. load_data(path)                -> DataFrame (kiểm tra schema đủ 30 cột)
2. preprocess(df)                 -> X (Amount thô), y (bỏ cột Time)
3. split_data(X, y)               -> train / val / test (chia Stratified)
4. fit_amount_scaler(X_train)     -> StandardScaler CHỈ fit trên tập Train
5. apply_amount_scaler(scaler, X) -> transform lần lượt cho train / val / test
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    AMOUNT_SCALER_FEATURE,
    FEATURE_COLS,
    PCA_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)

# Schema gốc bắt buộc có đủ 30 cột: Time + 28 cột PCA + Amount + Class
_REQUIRED_COLS = ["Time"] + PCA_FEATURES + ["Amount", TARGET_COL]


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download creditcard.csv from Kaggle and place it in data/raw/."
        )

    logger.info(f"Loading data from {path} ...")
    df = pd.read_csv(path)

    # Thẩm định schema: Kiểm tra xem có bị thiếu cột nào không
    missing_cols = set(_REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    n_total = len(df)
    n_fraud = df[TARGET_COL].sum()
    fraud_pct = n_fraud / n_total * 100

    logger.info(
        f"Dataset loaded: {n_total:,} rows | "
        f"Fraud: {n_fraud:,} ({fraud_pct:.4f}%) | "
        f"Legit: {n_total - n_fraud:,} ({100 - fraud_pct:.4f}%)"
    )

    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Bỏ Time (số giây tương đối, không mang tính khái quát cho model)
    df = df.drop(columns=["Time"], errors="ignore")
    logger.info("Dropped 'Time' column.")

    # Tách X (features thô) và y (nhãn gian lận) — giữ nguyên Amount thô để chống data leakage
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Features: {X.shape[1]} columns | Samples: {X.shape[0]:,}")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Bước 1: Tách tập Test (20%) khóa lại, không bao giờ dùng cho early stopping hay chọn model
    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Bước 2: Từ phần còn lại (80%), tách tiếp Val (20% của 80% = 16% tổng) để chọn model
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_size,
        random_state=random_state,
        stratify=y_rest,
    )

    for name, y_split in (("Train", y_train), ("Val  ", y_val), ("Test ", y_test)):
        logger.info(
            f"{name}: {len(y_split):,} samples "
            f"({y_split.sum():,} fraud, {y_split.sum() / len(y_split) * 100:.4f}%)"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_amount_scaler(X_train: pd.DataFrame) -> StandardScaler:
    # CHỈ fit trên tập Train để lấy mean và std, ngăn rò rỉ phân phối tập Test vào model
    scaler = StandardScaler()
    scaler.fit(X_train[[AMOUNT_SCALER_FEATURE]])
    logger.info(
        f"Fitted Amount scaler on TRAIN only: mean={scaler.mean_[0]:.4f} std={scaler.scale_[0]:.4f}"
    )
    return scaler


def apply_amount_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    # Áp dụng scaler đã fit lên từng tập (trả về bản sao mới, không sửa in-place)
    X = X.copy()
    X[AMOUNT_SCALER_FEATURE] = scaler.transform(X[[AMOUNT_SCALER_FEATURE]])
    return X
