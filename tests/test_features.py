"""Unit tests for src/features.py.

Uses synthetic data — does NOT require the real creditcard.csv dataset.
All tests are deterministic and fast (< 1s each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    AMOUNT_MEAN,
    AMOUNT_STD,
    FEATURE_COLS,
    PCA_FEATURES,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.features import (
    apply_amount_scaler,
    fit_amount_scaler,
    load_data,
    preprocess,
    split_data,
)

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

    def test_amount_is_left_raw(self, raw_df: pd.DataFrame) -> None:
        """Amount must pass through UNSCALED — scaling happens after the split.

        Fitting the scaler inside preprocess() would compute mean/std over the
        whole dataset, including the test rows, and leak them into training.
        """
        X, _ = preprocess(raw_df)
        np.testing.assert_array_equal(X["Amount"].values, raw_df["Amount"].values)

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

    def test_returns_three_splits(self, preprocessed) -> None:
        """split_data() returns train/val/test features and labels."""
        X, y = preprocessed
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        for frame in (X_train, X_val, X_test):
            assert isinstance(frame, pd.DataFrame)
        for series in (y_train, y_val, y_test):
            assert isinstance(series, pd.Series)

    def test_splits_partition_the_dataset(self, preprocessed) -> None:
        """The three splits must cover every row exactly once."""
        X, y = preprocessed
        X_train, X_val, X_test, _, _, _ = split_data(X, y)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)

    def test_default_split_ratio_64_16_20(self, preprocessed) -> None:
        """Default proportions: 64% train, 16% val, 20% test (±2%)."""
        X, y = preprocessed
        X_train, X_val, X_test, _, _, _ = split_data(X, y)
        total = len(X)
        assert abs(len(X_test) / total - 0.20) < 0.02
        assert abs(len(X_val) / total - 0.16) < 0.02
        assert abs(len(X_train) / total - 0.64) < 0.02

    def test_custom_split_ratio(self, preprocessed) -> None:
        """test_size and val_size parameters must both be respected."""
        X, y = preprocessed
        X_train, X_val, X_test, _, _, _ = split_data(X, y, test_size=0.3, val_size=0.5)
        assert abs(len(X_test) / len(X) - 0.30) < 0.02
        # val_size is a fraction of the post-test remainder: 0.5 * 0.7 = 0.35
        assert abs(len(X_val) / len(X) - 0.35) < 0.02

    def test_no_index_overlap_between_any_pair(self, preprocessed) -> None:
        """No row may appear in more than one split — the core leak guarantee."""
        X, y = preprocessed
        X_train, X_val, X_test, _, _, _ = split_data(X, y)
        train_idx, val_idx, test_idx = (
            set(X_train.index),
            set(X_val.index),
            set(X_test.index),
        )
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_stratified_fraud_ratio(self, preprocessed) -> None:
        """Fraud ratio must be approximately equal across all three splits."""
        X, y = preprocessed
        _, _, _, y_train, y_val, y_test = split_data(X, y)
        rates = [y_train.mean(), y_val.mean(), y_test.mean()]
        assert max(rates) - min(rates) < 0.01

    def test_every_split_contains_fraud(self, preprocessed) -> None:
        """Stratification must keep at least one positive in each split."""
        X, y = preprocessed
        _, _, _, y_train, y_val, y_test = split_data(X, y)
        assert y_train.sum() > 0
        assert y_val.sum() > 0
        assert y_test.sum() > 0

    def test_deterministic_with_same_seed(self, preprocessed) -> None:
        """Same random_state must produce identical splits."""
        X, y = preprocessed
        a = split_data(X, y, random_state=99)
        b = split_data(X, y, random_state=99)
        for frame_a, frame_b in zip(a[:3], b[:3], strict=True):
            pd.testing.assert_frame_equal(frame_a, frame_b)

    def test_test_split_is_independent_of_val_size(self, preprocessed) -> None:
        """Changing val_size must not move a single row into or out of test.

        The test set is carved out first precisely so that tuning the
        train/val proportion can never disturb the held-out evaluation set.
        """
        X, y = preprocessed
        _, _, test_a, _, _, _ = split_data(X, y, val_size=0.2)
        _, _, test_b, _, _, _ = split_data(X, y, val_size=0.4)
        assert set(test_a.index) == set(test_b.index)


# ---------------------------------------------------------------------------
# Amount scaler tests — the leak-free replacement for in-preprocess scaling
# ---------------------------------------------------------------------------


class TestAmountScaler:
    @pytest.fixture()
    def splits(self, raw_df: pd.DataFrame):
        X, y = preprocess(raw_df)
        return split_data(X, y)

    def test_scaler_fitted_only_on_train(self, splits) -> None:
        """Scaler statistics must equal the TRAIN split's own mean/std."""
        X_train = splits[0]
        scaler = fit_amount_scaler(X_train)
        assert scaler.mean_[0] == pytest.approx(X_train["Amount"].mean())
        assert scaler.scale_[0] == pytest.approx(X_train["Amount"].std(ddof=0))

    def test_scaler_ignores_val_and_test(self, splits) -> None:
        """Refitting on train alone vs. on all data must differ — proving no leak."""
        X_train, X_val, X_test = splits[0], splits[1], splits[2]
        train_only = fit_amount_scaler(X_train)
        all_data = fit_amount_scaler(pd.concat([X_train, X_val, X_test]))
        # Statistics computed over different row sets must not coincide exactly.
        assert train_only.mean_[0] != all_data.mean_[0]

    def test_transformed_train_is_standardized(self, splits) -> None:
        """Applying the scaler to its own training split yields mean 0, std 1."""
        X_train = splits[0]
        scaler = fit_amount_scaler(X_train)
        scaled = apply_amount_scaler(scaler, X_train)
        assert scaled["Amount"].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled["Amount"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    def test_apply_does_not_mutate_input(self, splits) -> None:
        """apply_amount_scaler() must return a copy, never edit in place."""
        X_val = splits[1]
        original = X_val["Amount"].copy()
        scaler = fit_amount_scaler(splits[0])
        apply_amount_scaler(scaler, X_val)
        pd.testing.assert_series_equal(X_val["Amount"], original)

    def test_apply_leaves_pca_features_untouched(self, splits) -> None:
        """Only Amount is scaled; V1-V28 are already PCA-whitened."""
        X_val = splits[1]
        scaler = fit_amount_scaler(splits[0])
        scaled = apply_amount_scaler(scaler, X_val)
        for col in PCA_FEATURES:
            np.testing.assert_array_equal(scaled[col].values, X_val[col].values)

    def test_config_constants_match_real_training_split(self) -> None:
        """src/config.py AMOUNT_MEAN/STD must match the real train split.

        src/api.py scales incoming Amount with these constants because the
        deployed artifact is a bare estimator, not a pipeline. If they drift
        from the scaler the champion was trained with, every served prediction
        is silently skewed. Skipped when the Kaggle CSV is not present.
        """
        if not RAW_DATA_PATH.exists():
            pytest.skip("creditcard.csv not available")
        X, y = preprocess(load_data(RAW_DATA_PATH))
        scaler = fit_amount_scaler(split_data(X, y)[0])
        assert scaler.mean_[0] == pytest.approx(AMOUNT_MEAN, abs=5e-5)
        assert scaler.scale_[0] == pytest.approx(AMOUNT_STD, abs=5e-5)
