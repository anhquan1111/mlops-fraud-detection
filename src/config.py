from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Đường dẫn gốc ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


# --- 1. Runtime Settings (Đọc từ .env / Environment Variables) ---
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MLFLOW_TRACKING_URI: str = Field(default="")
    FRAUD_REPORTS_DIR: str = Field(default="")
    DECISION_THRESHOLD: float = 0.5
    MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"
    AWS_S3_BUCKET: str = ""
    AWS_DEFAULT_REGION: str = "us-east-1"

    @field_validator("MLFLOW_TRACKING_URI", mode="before")
    @classmethod
    def _default_mlflow_uri(cls, v: Any) -> str:
        if v and str(v).strip():
            return str(v).strip()
        return f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

    @field_validator("FRAUD_REPORTS_DIR", mode="before")
    @classmethod
    def _default_reports_dir(cls, v: Any) -> str:
        if v and str(v).strip():
            return str(Path(str(v)).resolve())
        return str((PROJECT_ROOT / "reports").resolve())

    @field_validator("DECISION_THRESHOLD", mode="before")
    @classmethod
    def _validate_decision_threshold(cls, v: Any) -> float:
        if v is None:
            return 0.5
        if isinstance(v, str):
            trimmed = v.strip()
            if not trimmed:
                return 0.5
            try:
                val = float(trimmed)
            except ValueError as exc:
                raise ValueError(
                    f"DECISION_THRESHOLD={v!r} is not a number. "
                    "Set it to a value strictly between 0 and 1, e.g. 0.81."
                ) from exc
        else:
            try:
                val = float(v)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"DECISION_THRESHOLD={v!r} is not a number. "
                    "Set it to a value strictly between 0 and 1, e.g. 0.81."
                ) from exc

        if not 0.0 < val < 1.0:
            raise ValueError(
                f"DECISION_THRESHOLD={val} is out of range. It must be strictly between 0 and 1."
            )
        return val


# --- 2. Data & Feature Schema (Đọc từ configs/data.yaml) ---
class DataConfig(BaseModel):
    raw_data_path: str
    target_col: str = "Class"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    pca_features: list[str]
    amount_feature: str = "Amount"
    amount_mean: float = 87.9702
    amount_std: float = 245.5762

    @property
    def numeric_features(self) -> list[str]:
        return self.pca_features + [self.amount_feature]


# --- 3. Model Hyperparameters & Thresholds (Đọc từ configs/models.yaml) ---
class ModelConfig(BaseModel):
    registered_model_name: str = "fraud-detection-model"
    model_artifact_filename: str = "baseline_lr.pkl"
    min_recall: float = 0.80
    min_precision: float = 0.50
    logistic_regression: dict[str, Any]
    xgboost_base_params: dict[str, Any]
    xgboost_grid: list[dict[str, Any]]
    lightgbm_base_params: dict[str, Any]
    lightgbm_grid: list[dict[str, Any]]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Khởi tạo instances ---
settings = AppSettings()
data_config = DataConfig(**_load_yaml(CONFIGS_DIR / "data.yaml"))
model_config = ModelConfig(**_load_yaml(CONFIGS_DIR / "models.yaml"))

# --- Khai báo hằng số tương thích ngược cho toàn pipeline ---
REPORTS_DIR = Path(settings.FRAUD_REPORTS_DIR)
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = settings.MLFLOW_EXPERIMENT_NAME
DECISION_THRESHOLD = settings.DECISION_THRESHOLD

RAW_DATA_PATH = PROJECT_ROOT / data_config.raw_data_path
TARGET_COL = data_config.target_col
TEST_SIZE = data_config.test_size
VAL_SIZE = data_config.val_size
RANDOM_STATE = data_config.random_state
PCA_FEATURES = data_config.pca_features
NUMERIC_FEATURES = data_config.numeric_features
FEATURE_COLS = NUMERIC_FEATURES

AMOUNT_SCALER_FEATURE = data_config.amount_feature
AMOUNT_MEAN = data_config.amount_mean
AMOUNT_STD = data_config.amount_std

REGISTERED_MODEL_NAME = model_config.registered_model_name
MODEL_ARTIFACT_FILENAME = model_config.model_artifact_filename
LOCAL_MODEL_PATH = MODELS_DIR / MODEL_ARTIFACT_FILENAME
MIN_RECALL = model_config.min_recall
MIN_PRECISION = model_config.min_precision

LR_PARAMS = model_config.logistic_regression
XGBOOST_BASE_PARAMS = model_config.xgboost_base_params
XGBOOST_GRID = model_config.xgboost_grid
LIGHTGBM_BASE_PARAMS = model_config.lightgbm_base_params
LIGHTGBM_GRID = model_config.lightgbm_grid
