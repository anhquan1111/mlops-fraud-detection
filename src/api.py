"""FastAPI serving app for the fraud detection model.

Model loading strategy (in priority order):
    1. MODEL_PATH env var → load from local pickle file (dev / Docker with bundled model)
    2. HF_REPO_ID env var → download from Hugging Face Hub model repo (Render deploy)
    3. Fallback → load from MLflow Registry alias 'production' (local MLflow)

Endpoints:
    GET  /              → API info + model metadata
    GET  /health        → liveness probe (for Render / k8s)
    POST /predict       → predict single transaction
    POST /predict/batch → predict batch of transactions (max 100)

Usage (local dev):
    uv run uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import DECISION_THRESHOLD, FEATURE_COLS, MLFLOW_TRACKING_URI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTERED_MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "production"
HF_MODEL_FILENAME = "baseline_lr.pkl"  # filename on HF Hub (set by export_model.py)

# Approximate Amount scaler stats from the full Kaggle creditcard dataset
# (computed during EDA — used to scale incoming raw Amount values)
_AMOUNT_MEAN = 88.3496
_AMOUNT_STD = 250.1201

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: Any = None
_model_info: dict[str, str] = {}
_start_time = time.time()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_from_local(model_path: str) -> tuple[Any, dict]:
    """Load model from local pickle file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"✅ Model loaded from local file: {path}")
    return model, {"source": "local_file", "path": str(path)}


def _load_from_hf_hub(repo_id: str) -> tuple[Any, dict]:
    """Download model from Hugging Face Hub and load."""
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    logger.info(f"Downloading model from HF Hub: {repo_id} ...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=HF_MODEL_FILENAME,
        token=hf_token,
        cache_dir="/tmp/hf_cache",
    )
    model = joblib.load(local_path)
    logger.info(f"✅ Model downloaded from HF Hub: {repo_id}")
    return model, {"source": "huggingface_hub", "repo_id": repo_id}


def _load_from_mlflow() -> tuple[Any, dict]:
    """Load model from local MLflow Registry."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"Loading model from MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("✅ Model loaded from MLflow Registry.")
    return model, {"source": "mlflow_registry", "model_uri": model_uri}


def load_model() -> tuple[Any, dict]:
    """Load model using the priority strategy described in module docstring.

    Returns:
        Tuple of (model, info_dict).
    """
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return _load_from_local(model_path)

    hf_repo_id = os.environ.get("HF_REPO_ID")
    if hf_repo_id:
        return _load_from_hf_hub(hf_repo_id)

    return _load_from_mlflow()


# ---------------------------------------------------------------------------
# FastAPI lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model, _model_info

    logger.info("🚀 Starting Fraud Detection API ...")
    try:
        _model, _model_info = load_model()
        logger.info(f"Model ready. Source: {_model_info}")
    except Exception as exc:
        logger.error(f"❌ Failed to load model: {exc}")
        raise RuntimeError(f"Cannot start API without model: {exc}") from exc

    yield  # App runs here

    logger.info("🔻 Shutting down Fraud Detection API.")
    _model = None


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection powered by a LightGBM champion model "
        "(PR-AUC=0.8770, Recall=0.8571, Precision=0.8485). "
        "Input: 29 features (V1–V28 PCA-transformed + Amount scaled). "
        "Output: fraud probability and binary prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

_EXAMPLE_TRANSACTION = {
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}


class TransactionInput(BaseModel):
    """Single credit card transaction features.

    All V1-V28 are PCA-transformed (already scaled in original dataset).
    Amount should be raw (the API applies StandardScaler with dataset statistics internally).
    """

    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in USD (raw, >= 0)")

    model_config = {"json_schema_extra": {"example": _EXAMPLE_TRANSACTION}}


class PredictionResponse(BaseModel):
    """Fraud prediction result for a single transaction."""

    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0, 1]")
    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold")
    threshold: float = Field(..., description="Decision threshold used for prediction")
    model_name: str = Field(..., description="Model identifier")


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    count: int
    fraud_count: int


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    model_loaded: bool
    model_source: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preprocess_transaction(tx: TransactionInput) -> np.ndarray:
    """Convert a TransactionInput into a feature vector for prediction.

    Applies the same preprocessing as src/features.py:
    - V1-V28: pass-through (already PCA-scaled in dataset)
    - Amount: StandardScaler using dataset-level statistics

    Args:
        tx: Transaction input object.

    Returns:
        numpy array of shape (1, 29) ready for model.predict_proba().
    """
    features = []
    for col in FEATURE_COLS:
        val = getattr(tx, col)
        if col == "Amount":
            val = (val - _AMOUNT_MEAN) / _AMOUNT_STD
        features.append(val)
    return pd.DataFrame([features], columns=FEATURE_COLS)


def _predict_one(tx: TransactionInput, threshold: float = DECISION_THRESHOLD) -> PredictionResponse:
    """Run fraud prediction for a single transaction.

    Args:
        tx: Input transaction.
        threshold: Decision threshold.

    Returns:
        PredictionResponse with probability, label, and metadata.
    """
    X = _preprocess_transaction(tx)
    proba = float(_model.predict_proba(X)[0, 1])
    is_fraud = proba >= threshold
    return PredictionResponse(
        fraud_probability=round(proba, 6),
        is_fraud=is_fraud,
        threshold=threshold,
        model_name=f"{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="API info", tags=["Info"])
async def root() -> dict:
    """Return API metadata and model info."""
    return {
        "name": "Fraud Detection API",
        "version": app.version,
        "description": app.description,
        "model": _model_info,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
        },
        "threshold": DECISION_THRESHOLD,
        "feature_count": len(FEATURE_COLS),
    }


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["Info"])
async def health() -> HealthResponse:
    """Liveness probe — returns model status and uptime."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_source=_model_info.get("source", "unknown"),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict single transaction",
    tags=["Prediction"],
)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """Predict fraud probability for a single credit card transaction.

    Args:
        transaction: Transaction features (V1-V28 + Amount).

    Returns:
        Fraud probability, binary label, and threshold used.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again shortly.",
        )
    try:
        return _predict_one(transaction)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict batch of transactions",
    tags=["Prediction"],
)
async def predict_batch(transactions: list[TransactionInput]) -> BatchPredictionResponse:
    """Predict fraud for a batch of transactions (max 100).

    Args:
        transactions: List of transaction inputs.

    Returns:
        List of predictions with summary counts.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again shortly.",
        )
    if len(transactions) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="transactions list cannot be empty.",
        )
    if len(transactions) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Batch size exceeds limit: {len(transactions)} > 100.",
        )
    try:
        predictions = [_predict_one(tx) for tx in transactions]
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            fraud_count=fraud_count,
        )
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(exc)}",
        ) from exc
