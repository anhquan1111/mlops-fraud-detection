"""Hermetic API contract tests with a temporary synthetic model artifact."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLS, RANDOM_STATE


def _build_stub_model() -> LogisticRegression:
    """Fit a throwaway LogisticRegression on synthetic data with the real schema.

    Returns:
        A fitted classifier exposing predict_proba over len(FEATURE_COLS) inputs.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = 500
    # Fit on a named DataFrame so predictions on the API's DataFrame input do
    # not trip sklearn's "X does not have valid feature names" warning.
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    # Imbalanced labels so predict_proba spans a realistic low-probability range
    y = (rng.random(n) < 0.05).astype(int)
    y[0] = 1  # guarantee both classes are present
    y[1] = 0
    return LogisticRegression(class_weight="balanced", max_iter=200).fit(X, y)


@pytest.fixture(scope="session", autouse=True)
def isolated_api_model(tmp_path_factory):
    """Force API tests to use a temporary stub, never a user's model/registry.

    No artifacts are created at import time or in the project's models directory.
    The filename environment override is restored when the test session ends.
    """
    path = tmp_path_factory.mktemp("api-model") / "stub.pkl"
    joblib.dump(_build_stub_model(), path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_PATH", str(path))
        yield
