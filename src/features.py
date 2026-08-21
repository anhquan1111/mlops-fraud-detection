"""Feature engineering and data loading for the fraud detection pipeline.

Pipeline (leak-free order — the split happens BEFORE any statistic is fitted):

    load_data(path)                     -> DataFrame
    preprocess(df)                      -> X (raw Amount), y
    split_data(X, y)                    -> train / val / test
    fit_amount_scaler(X_train)          -> StandardScaler fitted on train only
    apply_amount_scaler(scaler, X)      -> scaled copy of X

`preprocess()` deliberately does NOT scale. Fitting a scaler on the full frame
and splitting afterwards leaks test-set statistics into training; the scaler is
therefore fitted from the training split alone and then applied to val and test.
See docs/leakage_fix.md.
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
    VAL_SIZE,
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
        2. Separate features (X) and target (y).

    `Amount` is returned RAW. Scaling is a fitted transformation and must not be
    applied before the train/test split — use fit_amount_scaler() on the training
    split, then apply_amount_scaler() on each split.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Tuple of (X, y) where X holds unscaled features and y is the binary target.
    """
    df = df.copy()

    # Drop Time — not informative for baseline (seconds since first transaction)
    df = df.drop(columns=["Time"], errors="ignore")
    logger.info("Dropped 'Time' column.")

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
    """Stratified three-way split preserving the fraud ratio in every split.

    The test set is carved out FIRST and never participates in early stopping or
    model selection. The validation set is then taken from what remains.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of the FULL dataset held out for final reporting.
        val_size: Fraction of the POST-TEST remainder used for validation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # 1. Hold out the test set first — it is touched exactly once, at reporting.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Carve the validation set out of the remainder (early stopping + selection).
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
    """Fit the `Amount` StandardScaler on the TRAINING SPLIT ONLY.

    V1–V28 are already PCA-whitened in the source dataset, so `Amount` is the
    only feature that needs scaling.

    Args:
        X_train: Training features, with a raw `Amount` column.

    Returns:
        A StandardScaler fitted on X_train[["Amount"]].
    """
    scaler = StandardScaler()
    scaler.fit(X_train[[AMOUNT_SCALER_FEATURE]])
    logger.info(
        f"Fitted Amount scaler on TRAIN only: mean={scaler.mean_[0]:.4f} std={scaler.scale_[0]:.4f}"
    )
    return scaler


def apply_amount_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted `Amount` scaler to a split, returning a scaled copy.

    Args:
        scaler: Scaler produced by fit_amount_scaler().
        X: Features with a raw `Amount` column.

    Returns:
        A new DataFrame with `Amount` scaled; the input is left untouched.
    """
    X = X.copy()
    X[AMOUNT_SCALER_FEATURE] = scaler.transform(X[[AMOUNT_SCALER_FEATURE]])
    return X
