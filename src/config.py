"""Central configuration for the MLOps Fraud Detection pipeline.

All paths, constants, and hyperparameter defaults are defined here.
Import from this module to avoid magic strings/numbers scattered across code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & data paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = DATA_RAW_DIR / "creditcard.csv"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

# SQLite backend — required by MLflow 3.x (file store deprecated)
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "fraud-detection"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TARGET_COL = "Class"

# V1–V28 are PCA-transformed features (anonymized)
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]

# Amount is the only non-PCA feature kept (Time is dropped for baseline)
NUMERIC_FEATURES: list[str] = PCA_FEATURES + ["Amount"]

# All features used for modelling
FEATURE_COLS: list[str] = NUMERIC_FEATURES  # Time is dropped in preprocessing

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

AMOUNT_SCALER_FEATURE = "Amount"  # only feature that needs scaling in baseline

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------
# ⚠️ DO NOT change without consulting AGENTS.md — this is a business decision.

DECISION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Logistic Regression baseline hyperparameters
# ---------------------------------------------------------------------------

LR_PARAMS: dict = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}

# ---------------------------------------------------------------------------
# Success thresholds (from AGENTS.md)
# ---------------------------------------------------------------------------

MIN_RECALL: float = 0.80
MIN_PRECISION: float = 0.50
