"""Cổng kiểm soát chất lượng dữ liệu đầu vào (Data Quality Gate).

Kiểm tra toàn vẹn schema, chặn giá trị null/inf và kiểm tra khoảng giá trị hợp lệ
trước khi đưa dữ liệu vào dự đoán hoặc phân tích drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from src.config import FEATURE_COLS


def validate_transaction_contract(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
) -> list[str]:
    """Kiểm tra dữ liệu giao dịch theo Data Contract (schema, null, inf, Amount >= 0)."""
    issues: list[str] = []
    required = required_features if required_features is not None else FEATURE_COLS

    if df.empty:
        issues.append("empty_batch")
        return issues

    # Duplicate names make df[col] a DataFrame, so reject the schema before value checks.
    duplicates = sorted(set(df.columns[df.columns.duplicated()]))
    if duplicates:
        return [f"duplicate_column:{col}" for col in duplicates]

    # 1. Missing columns
    missing_cols = sorted(set(required) - set(df.columns))
    issues.extend(f"missing_column:{col}" for col in missing_cols)

    # 2. Check present columns
    present_cols = sorted(set(required) & set(df.columns))
    for col in present_cols:
        series = df[col]

        # Check numeric type
        if not is_numeric_dtype(series) or is_bool_dtype(series) or is_complex_dtype(series):
            issues.append(f"non_numeric:{col}")
            continue

        # Check nulls
        null_count = series.isna().sum()
        if null_count > 0:
            null_rate = null_count / len(df)
            issues.append(f"null:{col}:{null_rate:.1%}")

        # Drop nulls for value range checks
        valid_vals = series.dropna()
        if not valid_vals.empty:
            if not np.isfinite(valid_vals).all():
                issues.append(f"infinite_value:{col}")

    # 3. Domain specific range check: Amount >= 0
    if "Amount" in present_cols and "non_numeric:Amount" not in issues:
        valid_amount = df["Amount"].dropna()
        if (valid_amount < 0).any():
            issues.append("negative_amount")

    return sorted(issues)


def quality_gate(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Cổng kiểm soát chất lượng dữ liệu đầu vào, ném lỗi ValueError nếu vi phạm."""
    issues = validate_transaction_contract(df, required_features=required_features)
    if issues and raise_on_error:
        raise ValueError(f"Data Quality Gate failed with {len(issues)} issue(s): {issues}")
    return issues
