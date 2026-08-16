"""Export trained model from MLflow Registry to local pickle + upload to Hugging Face Hub.

This script:
1. Loads the 'production' aliased model from MLflow Registry
2. Saves it as models/baseline_lr.pkl (gitignored)
3. Optionally uploads to Hugging Face Hub model repo for Render deployment

Usage:
    # Export only (local)
    uv run python scripts/export_model.py

    # Export + upload to HF Hub
    HF_TOKEN=hf_xxx HF_REPO_ID=your-username/fraud-detection-model \\
        uv run python scripts/export_model.py --upload
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "fraud-detection-model"  # matches src/config.py
MODEL_ALIAS = "production"
LOCAL_MODEL_PATH = Path("models/fraud_model.pkl")


def export_model(upload: bool = False) -> Path:
    """Load production model from MLflow Registry and save as pickle.

    Args:
        upload: If True, upload to Hugging Face Hub after exporting.

    Returns:
        Path to the saved pickle file.
    """
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get production alias version
    model_version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )
    logger.info(
        f"Found '{MODEL_ALIAS}' model: {REGISTERED_MODEL_NAME} v{model_version.version} "
        f"(run_id={model_version.run_id})"
    )

    # Load the sklearn model object
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"Loading model from {model_uri} ...")
    model = mlflow.sklearn.load_model(model_uri)

    # Save to pickle
    joblib.dump(model, LOCAL_MODEL_PATH)
    size_kb = LOCAL_MODEL_PATH.stat().st_size / 1024
    logger.info(f"✅ Model saved → {LOCAL_MODEL_PATH} ({size_kb:.1f} KB)")

    # Optional: upload to HF Hub
    if upload:
        _upload_to_hf_hub(LOCAL_MODEL_PATH, model_version)

    return LOCAL_MODEL_PATH


def _upload_to_hf_hub(model_path: Path, model_version) -> None:
    """Upload model pickle to Hugging Face Hub model repository.

    Args:
        model_path: Local path to the pickle file.
        model_version: MLflow model version object (for metadata).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: uv sync")
        return

    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")

    if not hf_repo_id:
        logger.error(
            "HF_REPO_ID environment variable not set. "
            "Example: HF_REPO_ID=your-username/fraud-detection-model"
        )
        return

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    logger.info(f"HF Hub repo ready: https://huggingface.co/{hf_repo_id}")

    # Upload model pickle
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="baseline_lr.pkl",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message=(
            f"Upload baseline LR model v{model_version.version} (run_id={model_version.run_id[:8]})"
        ),
    )
    logger.info(
        f"✅ Model uploaded to HF Hub: "
        f"https://huggingface.co/{hf_repo_id}/blob/main/baseline_lr.pkl"
    )

    # Upload model card
    card_content = f"""---
language: en
tags:
  - fraud-detection
  - lightgbm
  - scikit-learn
  - imbalanced-classification
license: mit
---

# Credit Card Fraud Detection — LightGBM Champion Model

## Model Description

LightGBM (`lgbm_large`) champion model for credit card fraud detection on the
[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Registered in MLflow Registry as `fraud-detection-model@production`.

## Performance (Test Set — 56,962 samples)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.8770 |
| Recall | 0.8571 |
| Precision | 0.8485 |
| F1 | 0.8528 |

## Usage

```python
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")
# features: V1-V28 (PCA), Amount (StandardScaler μ=88.35, σ=250.12)
X = np.array([[...]])  # shape (1, 29)
proba = model.predict_proba(X)[:, 1]  # fraud probability
```

## MLflow Tracking

- Run ID: `{model_version.run_id}`
- Model version: `{model_version.version}`
- Registered name: `{REGISTERED_MODEL_NAME}`
"""

    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    logger.info("Model card uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model from MLflow → local + HF Hub")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Hugging Face Hub (requires HF_TOKEN and HF_REPO_ID env vars)",
    )
    args = parser.parse_args()
    export_model(upload=args.upload)
