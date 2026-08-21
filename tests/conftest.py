"""Shared pytest setup.

Ensures the API tests always have a model file to load.

`models/` is gitignored (AGENTS.md forbids committing .pkl artifacts), so on a
fresh clone — and in CI — `models/baseline_lr.pkl` does not exist and the
FastAPI lifespan aborts with "Cannot start API without model". We generate a
tiny synthetic stand-in here so the API contract tests stay hermetic and never
depend on a locally trained artifact.

A real model already present on disk is never overwritten.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLS, PROJECT_ROOT, RANDOM_STATE

TEST_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_lr.pkl"


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


def _ensure_test_model(path: Path = TEST_MODEL_PATH) -> None:
    """Create a stub model file if none exists (no-op when a real model is present)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_build_stub_model(), path)


# Runs at conftest import — before tests/test_api.py imports src.api and starts
# the lifespan, which is the only point at which the file must already exist.
_ensure_test_model()
