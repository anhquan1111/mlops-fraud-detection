"""Feature engineering and data loading for the fraud detection pipeline.

Pipeline:
    load_data(path) -> DataFrame
    preprocess(df)  -> X (DataFrame), y (Series)
    split_data(X, y) -> X_train, X_test, y_train, y_test
"""

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
)

logger = logging.getLogger(__name__)

# Expected columns in raw creditcard.csv
_REQUIRED_COLS = ["Time"] + PCA_FEATURES + ["Amount", TARGET_COL]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load raw CSV data and validate expected schema.

    Args:
        path: Path to creditcard.csv

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download creditcard.csv from Kaggle and place it in data/raw/."
        )

    logger.info(f"Loading data from {path} ...")
    df = pd.read_csv(path)

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
    """Preprocess raw data for modelling.

    Steps:
        1. Drop 'Time' column (seconds since first tx — not informative for baseline).
        2. Scale 'Amount' with StandardScaler (V1-V28 are already PCA-scaled).
        3. Separate features (X) and target (y).

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Tuple of (X, y) where X has scaled features and y is the binary target.
    """
    df = df.copy()

    # Drop Time — not informative for baseline (seconds since first transaction)
    df = df.drop(columns=["Time"], errors="ignore")
    logger.info("Dropped 'Time' column.")

    # Scale Amount — V1-V28 are already PCA-whitened, only Amount needs scaling
    scaler = StandardScaler()
    df[AMOUNT_SCALER_FEATURE] = scaler.fit_transform(df[[AMOUNT_SCALER_FEATURE]])
    logger.info("Scaled 'Amount' with StandardScaler.")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Features: {X.shape[1]} columns | Samples: {X.shape[0]:,}")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving fraud ratio.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction for test set (default 0.2).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keeps fraud ratio equal in train and test
    )

    logger.info(
        f"Train: {len(X_train):,} samples "
        f"({y_train.sum():,} fraud, {y_train.sum() / len(y_train) * 100:.4f}%)"
    )
    logger.info(
        f"Test:  {len(X_test):,} samples "
        f"({y_test.sum():,} fraud, {y_test.sum() / len(y_test) * 100:.4f}%)"
    )
    return X_train, X_test, y_train, y_test
