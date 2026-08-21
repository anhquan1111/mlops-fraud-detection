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

# StandardScaler statistics for `Amount`, fitted on the TRAINING SPLIT ONLY
# (64% of the data, RANDOM_STATE=42). src/api.py applies these at serve time
# because the deployed artifact is the bare estimator, not a full pipeline.
#
# ⚠️ These MUST be regenerated whenever the split or RANDOM_STATE changes:
#     uv run python src/train.py   (prints the fitted values)
# tests/test_features.py asserts they match the scaler fitted from the raw CSV
# when it is available, so drift here fails the test suite rather than silently
# skewing production scores.

AMOUNT_MEAN: float = 87.9702
AMOUNT_STD: float = 245.5762

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
# Three-way split, stratified at every step, so that the test set is touched
# exactly once — at final reporting.
#
#   test  = TEST_SIZE of the full dataset                  -> 20%
#   val   = VAL_SIZE of what remains after test is removed -> 0.2 * 0.8 = 16%
#   train = the rest                                       -> 64%
#
# Early stopping and champion selection both run on val. Using test for either
# leaks the test set into model selection and inflates the reported metrics —
# see docs/leakage_fix.md.

TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.2  # fraction of the post-test remainder, NOT of the full set
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
# XGBoost base hyperparameters
# ---------------------------------------------------------------------------
# scale_pos_weight = n_negative / n_positive — computed dynamically in train.py

XGBOOST_BASE_PARAMS: dict = {
    "eval_metric": "aucpr",  # optimize for PR-AUC
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",  # faster than exact
    "early_stopping_rounds": 20,  # stop if no improvement for 20 rounds
}

# Experiment grid: list of param dicts (merged with base)
XGBOOST_GRID: list[dict] = [
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "run_name": "xgb_default",
    },
    {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "run_name": "xgb_deep",
    },
    {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "run_name": "xgb_regularized",
    },
]

# ---------------------------------------------------------------------------
# LightGBM base hyperparameters
# ---------------------------------------------------------------------------
# ⚠️  Do NOT use scale_pos_weight with LightGBM at very high ratios (e.g. 577).
#    At such extreme ratios, LGBM's internal re-weighting destabilises leaf
#    splits and produces near-random PR-AUC (~0.09).  Use class_weight='balanced'
#    instead — it applies per-sample weights through sklearn's API and is
#    stable across all imbalance ratios.

LIGHTGBM_BASE_PARAMS: dict = {
    "class_weight": "balanced",  # imbalance fix — stable at any ratio
    "metric": "average_precision",  # PR-AUC equivalent in LGBM
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,  # suppress LGBM output
}

# Experiment grid: list of param dicts (merged with base)
LIGHTGBM_GRID: list[dict] = [
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "run_name": "lgbm_default",
    },
    {
        "n_estimators": 300,
        "max_depth": -1,  # -1 = no limit (LGBM default)
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "run_name": "lgbm_large",
    },
    {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,  # lower = less regularization, better for rare class
        "run_name": "lgbm_regularized",
    },
]

# ---------------------------------------------------------------------------
# MLflow Model Registry
# ---------------------------------------------------------------------------

REGISTERED_MODEL_NAME: str = "fraud-detection-model"

# ---------------------------------------------------------------------------
# Deployment artifact
# ---------------------------------------------------------------------------
# Filename of the exported champion pickle, both locally and on the Hugging Face
# Hub. scripts/export_model.py writes it, src/api.py downloads it — they MUST
# agree, so both import this constant instead of hard-coding a literal.
# The historical name is "baseline_lr.pkl"; the live HF repo still serves that
# filename, so changing it requires re-uploading before redeploying.

MODEL_ARTIFACT_FILENAME: str = "baseline_lr.pkl"
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / MODEL_ARTIFACT_FILENAME

# ---------------------------------------------------------------------------
# Success thresholds (from AGENTS.md)
# ---------------------------------------------------------------------------

MIN_RECALL: float = 0.80
MIN_PRECISION: float = 0.50
