"""Unit tests for the Data Quality Gate (src/quality.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS
from src.quality import quality_gate, validate_transaction_contract


@pytest.fixture
def valid_sample_df() -> pd.DataFrame:
    """Fixture returning a single valid transaction row with all 29 features."""
    data = {f"V{i}": [0.1 * i] for i in range(1, 29)}
    data["Amount"] = [99.50]
    return pd.DataFrame(data)


class TestValidateTransactionContract:
    """Test suite for validate_transaction_contract."""

    def test_valid_dataframe_passes(self, valid_sample_df: pd.DataFrame) -> None:
        issues = validate_transaction_contract(valid_sample_df)
        assert issues == []

    def test_empty_dataframe_fails(self) -> None:
        empty_df = pd.DataFrame(columns=FEATURE_COLS)
        issues = validate_transaction_contract(empty_df)
        assert "empty_batch" in issues

    def test_missing_column_fails(self, valid_sample_df: pd.DataFrame) -> None:
        incomplete_df = valid_sample_df.drop(columns=["V1", "Amount"])
        issues = validate_transaction_contract(incomplete_df)
        assert "missing_column:Amount" in issues
        assert "missing_column:V1" in issues

    def test_null_value_fails(self, valid_sample_df: pd.DataFrame) -> None:
        df_with_null = valid_sample_df.copy()
        df_with_null.loc[0, "Amount"] = np.nan
        issues = validate_transaction_contract(df_with_null)
        assert any("null:Amount" in issue for issue in issues)

    def test_infinite_value_fails(self, valid_sample_df: pd.DataFrame) -> None:
        df_with_inf = valid_sample_df.copy()
        df_with_inf.loc[0, "V2"] = np.inf
        issues = validate_transaction_contract(df_with_inf)
        assert "infinite_value:V2" in issues

    def test_negative_amount_fails(self, valid_sample_df: pd.DataFrame) -> None:
        df_negative = valid_sample_df.copy()
        df_negative.loc[0, "Amount"] = -50.0
        issues = validate_transaction_contract(df_negative)
        assert "negative_amount" in issues

    def test_non_numeric_dtype_fails(self, valid_sample_df: pd.DataFrame) -> None:
        df_string = valid_sample_df.copy()
        df_string["V5"] = ["not_a_number"]
        issues = validate_transaction_contract(df_string)
        assert "non_numeric:V5" in issues


class TestQualityGate:
    """Test suite for quality_gate wrapper."""

    def test_gate_raises_on_error(self, valid_sample_df: pd.DataFrame) -> None:
        invalid_df = valid_sample_df.drop(columns=["Amount"])
        with pytest.raises(ValueError, match="Data Quality Gate failed"):
            quality_gate(invalid_df, raise_on_error=True)

    def test_gate_returns_issues_when_no_raise(self, valid_sample_df: pd.DataFrame) -> None:
        invalid_df = valid_sample_df.drop(columns=["Amount"])
        issues = quality_gate(invalid_df, raise_on_error=False)
        assert "missing_column:Amount" in issues

    def test_gate_passes_cleanly(self, valid_sample_df: pd.DataFrame) -> None:
        issues = quality_gate(valid_sample_df, raise_on_error=True)
        assert issues == []
