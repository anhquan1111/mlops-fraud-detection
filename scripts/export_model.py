"""Export trained model from MLflow Registry to local pickle + upload to Hugging Face Hub.

This script:
1. Loads the 'production' aliased model from MLflow Registry
2. Saves it locally as models/<MODEL_ARTIFACT_FILENAME> (gitignored)
3. Optionally uploads it to Hugging Face Hub under the SAME filename, which is
   what src/api.py downloads at startup on Render.

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

from src.config import (
    AMOUNT_MEAN,
    AMOUNT_STD,
    LOCAL_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACT_FILENAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "fraud-detection-model"  # matches src/config.py
MODEL_ALIAS = "production"


def _load_any_flavor(model_uri: str):
    """Load MLflow model using the correct flavor (sklearn / lightgbm / xgboost).

    Tries each flavor in order until one succeeds. This is needed because the
    registered production model may be LightGBM, XGBoost, or sklearn depending
    on which experiment run performed best.

    Args:
        model_uri: MLflow model URI, e.g. 'models:/fraud-detection-model@production'.

    Returns:
        Loaded model object supporting predict_proba().

    Raises:
        RuntimeError: If no supported flavor is found.
    """
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost

    loaders = [
        ("sklearn", mlflow.sklearn.load_model),
        ("lightgbm", mlflow.lightgbm.load_model),
        ("xgboost", mlflow.xgboost.load_model),
    ]
    last_exc: Exception | None = None
    for flavor_name, loader in loaders:
        try:
            model = loader(model_uri)
            logger.info(f"[OK] Loaded as '{flavor_name}' flavor.")
            return model
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Flavor '{flavor_name}' not available: {exc}")
            last_exc = exc
    raise RuntimeError(
        f"Cannot load model '{model_uri}' — no supported flavor found."
    ) from last_exc


def _load_run_metrics(run_id: str) -> dict[str, float]:
    """Read the metrics logged on an MLflow run.

    Args:
        run_id: MLflow run ID backing the registered model version.

    Returns:
        Metric name -> value, or an empty dict if the run cannot be read.
    """
    try:
        return dict(MlflowClient().get_run(run_id).data.metrics)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read metrics for run {run_id}: {exc}")
        return {}


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

    # Load the model — auto-detect flavor (sklearn / lightgbm / xgboost)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"Loading model from {model_uri} ...")
    model = _load_any_flavor(model_uri)

    # Save to pickle
    joblib.dump(model, LOCAL_MODEL_PATH)
    size_kb = LOCAL_MODEL_PATH.stat().st_size / 1024
    logger.info(f"[OK] Model saved -> {LOCAL_MODEL_PATH} ({size_kb:.1f} KB)")

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
        path_in_repo=MODEL_ARTIFACT_FILENAME,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message=(
            f"Upload champion model v{model_version.version} (run_id={model_version.run_id[:8]})"
        ),
    )
    logger.info(
        f"[OK] Model uploaded to HF Hub: "
        f"https://huggingface.co/{hf_repo_id}/blob/main/{MODEL_ARTIFACT_FILENAME}"
    )

    # Upload model card. Every number is read from the MLflow run so the card can
    # never drift away from the model it describes — the failure mode that put
    # fabricated metrics in this project's README for five days.
    metrics = _load_run_metrics(model_version.run_id)

    def _m(key: str, fmt: str = ".4f") -> str:
        value = metrics.get(key)
        return f"{value:{fmt}}" if value is not None else "n/a"

    card_content = f"""---
language: en
tags:
  - fraud-detection
  - lightgbm
  - scikit-learn
  - imbalanced-classification
license: mit
---

# Credit Card Fraud Detection — Champion Model

## Model Description

Champion model for credit card fraud detection on the
[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Registered in MLflow Registry as `{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}`.

Trained under a leak-free protocol: stratified 64/16/20 train/validation/test split, the
`Amount` scaler fitted on the training split only, early stopping watched on validation, and
the champion selected by validation PR-AUC. The test split influences no decision and is
scored once for reporting.

## Performance

Selected on **validation** ({_m("n_val_samples", ",.0f")} rows):

| Metric | Value |
|--------|-------|
| PR-AUC | {_m("val_pr_auc")} |
| Recall | {_m("val_recall")} |
| Precision | {_m("val_precision")} |
| F1 | {_m("val_f1")} |

Reported on the held-out **test** split ({_m("n_test_samples", ",.0f")} rows):

| Metric | Value |
|--------|-------|
| PR-AUC | {_m("test_pr_auc")} |
| Recall | {_m("test_recall")} |
| Precision | {_m("test_precision")} |
| F1 | {_m("test_f1")} |
| True positives | {_m("test_tp", ".0f")} |
| False positives | {_m("test_fp", ".0f")} |
| False negatives | {_m("test_fn", ".0f")} |

## Usage

```python
import joblib
import numpy as np

model = joblib.load("{MODEL_ARTIFACT_FILENAME}")
# features: V1-V28 (PCA, pass through), Amount scaled with the TRAINING-split
# StandardScaler: mu={AMOUNT_MEAN}, sigma={AMOUNT_STD}
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
