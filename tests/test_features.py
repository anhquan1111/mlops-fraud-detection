"""Unit tests for src/features.py.

Uses synthetic data — does NOT require the real creditcard.csv dataset.
All tests are deterministic and fast (< 1s each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, PCA_FEATURES, TARGET_COL
from src.features import load_data, preprocess, split_data

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 500
N_FRAUD = 10  # ~2% fraud — small but enough to stratify


def _make_raw_df(n: int = N_SAMPLES, n_fraud: int = N_FRAUD, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic creditcard-shaped DataFrame.

    Matches the exact schema of creditcard.csv:
        Time, V1-V28, Amount, Class
    """
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"Time": rng.uniform(0, 172792, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    data["Amount"] = rng.uniform(0, 500, n)
    labels = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    labels[fraud_idx] = 1
    data[TARGET_COL] = labels
    return pd.DataFrame(data)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Synthetic raw DataFrame (no file I/O)."""
    return _make_raw_df()


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """Write synthetic data to a temp CSV and return the path."""
    df = _make_raw_df()
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_data tests
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError when path does not exist."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_data(tmp_path / "nonexistent.csv")

    def test_raises_value_error_on_missing_columns(self, tmp_path: Path) -> None:
        """ValueError when required columns are absent."""
        bad_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
        bad_path = tmp_path / "bad.csv"
        bad_df.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="Missing expected columns"):
            load_data(bad_path)

    def test_returns_dataframe(self, csv_file: Path) -> None:
        """Should return a pandas DataFrame."""
        df = load_data(csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_all_required_columns_present(self, csv_file: Path) -> None:
        """Loaded df must have Time, V1-V28, Amount, and Class."""
        df = load_data(csv_file)
        required = {"Time"} | set(PCA_FEATURES) | {"Amount", TARGET_COL}
        assert required.issubset(set(df.columns))

    def test_row_count_preserved(self, csv_file: Path) -> None:
        """Row count must match original data."""
        df = load_data(csv_file)
        assert len(df) == N_SAMPLES


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_drops_time_column(self, raw_df: pd.DataFrame) -> None:
        """'Time' column must be absent from X after preprocessing."""
        X, _ = preprocess(raw_df)
        assert "Time" not in X.columns

    def test_drops_class_column(self, raw_df: pd.DataFrame) -> None:
        """Target column 'Class' must be in y, not X."""
        X, y = preprocess(raw_df)
        assert TARGET_COL not in X.columns
        assert isinstance(y, pd.Series)

    def test_output_feature_count(self, raw_df: pd.DataFrame) -> None:
        """X must have exactly len(FEATURE_COLS) columns (V1-V28 + Amount = 29)."""
        X, _ = preprocess(raw_df)
        assert X.shape[1] == len(FEATURE_COLS)
        assert list(X.columns) == FEATURE_COLS

    def test_amount_is_scaled(self, raw_df: pd.DataFrame) -> None:
        """Amount should be standardized: mean ≈ 0, std ≈ 1 (approximately)."""
        X, _ = preprocess(raw_df)
        amount_mean = X["Amount"].mean()
        amount_std = X["Amount"].std()
        assert abs(amount_mean) < 0.1, (
            f"Amount mean after scaling: {amount_mean:.4f} (expected ≈ 0)"
        )
        assert abs(amount_std - 1.0) < 0.1, (
            f"Amount std after scaling: {amount_std:.4f} (expected ≈ 1)"
        )

    def test_pca_features_unchanged(self, raw_df: pd.DataFrame) -> None:
        """V1-V28 values must pass through unchanged."""
        X, _ = preprocess(raw_df)
        original_v1 = raw_df["V1"].values
        np.testing.assert_array_equal(X["V1"].values, original_v1)

    def test_target_values_are_binary(self, raw_df: pd.DataFrame) -> None:
        """y must contain only 0 and 1."""
        _, y = preprocess(raw_df)
        assert set(y.unique()).issubset({0, 1})

    def test_does_not_mutate_input(self, raw_df: pd.DataFrame) -> None:
        """preprocess() must not modify the input DataFrame in place."""
        original_amount_mean = raw_df["Amount"].mean()
        preprocess(raw_df)
        assert raw_df["Amount"].mean() == pytest.approx(original_amount_mean)


# ---------------------------------------------------------------------------
# split_data tests
# ---------------------------------------------------------------------------


class TestSplitData:
    @pytest.fixture()
    def preprocessed(self, raw_df: pd.DataFrame):
        return preprocess(raw_df)

    def test_default_split_ratio(self, preprocessed) -> None:
        """Default 80/20 split — test set should be ~20% of total."""
        X, y = preprocessed
        X_train, X_test, y_train, y_test = split_data(X, y)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        test_ratio = len(X_test) / total
        assert abs(test_ratio - 0.2) < 0.02  # ±2% tolerance

    def test_custom_split_ratio(self, preprocessed) -> None:
        """Custom test_size parameter should be respected."""
        X, y = preprocessed
        X_train, X_test, _, _ = split_data(X, y, test_size=0.3)
        test_ratio = len(X_test) / len(X)
        assert abs(test_ratio - 0.3) < 0.02

    def test_no_index_overlap(self, preprocessed) -> None:
        """Train and test sets must not share any row indices."""
        X, y = preprocessed
        X_train, X_test, _, _ = split_data(X, y)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_stratified_fraud_ratio(self, preprocessed) -> None:
        """Fraud ratio in train and test should be approximately equal."""
        X, y = preprocessed
        _, _, y_train, y_test = split_data(X, y)
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        # Tolerance: ±1% absolute
        assert abs(train_fraud_rate - test_fraud_rate) < 0.01

    def test_deterministic_with_same_seed(self, preprocessed) -> None:
        """Same random_state must produce identical splits."""
        X, y = preprocessed
        X_train_a, X_test_a, _, _ = split_data(X, y, random_state=99)
        X_train_b, X_test_b, _, _ = split_data(X, y, random_state=99)
        pd.testing.assert_frame_equal(X_train_a, X_train_b)
        pd.testing.assert_frame_equal(X_test_a, X_test_b)

    def test_returns_dataframes_and_series(self, preprocessed) -> None:
        """Return types: (DataFrame, DataFrame, Series, Series)."""
        X, y = preprocessed
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
