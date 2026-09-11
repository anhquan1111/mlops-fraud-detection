"""Xuất mô hình Production từ MLflow Registry ra file pickle cục bộ và tải lên Hugging Face Hub."""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

REGISTERED_MODEL_NAME = "fraud-detection-model"  # Khớp với khai báo trong src/config.py
MODEL_ALIAS = "production"


def _load_any_flavor(model_uri: str):
    """Nạp model từ MLflow theo đúng định dạng tương ứng (sklearn, lightgbm hoặc xgboost)."""
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
    """Đọc các metrics đã ghi nhận của run tương ứng từ MLflow."""
    try:
        return dict(MlflowClient().get_run(run_id).data.metrics)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read metrics for run {run_id}: {exc}")
        return {}


def export_model(upload: bool = False) -> Path:
    """Nạp mô hình Production từ MLflow Registry và lưu thành file pickle cục bộ."""
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Lấy thông tin model version mang alias 'production'
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

    # 3. Lưu mô hình thành file pickle cục bộ
    joblib.dump(model, LOCAL_MODEL_PATH)
    size_kb = LOCAL_MODEL_PATH.stat().st_size / 1024
    logger.info(f"[OK] Model saved -> {LOCAL_MODEL_PATH} ({size_kb:.1f} KB)")

    # 4. Tùy chọn tải lên Hugging Face Hub
    if upload:
        _upload_to_hf_hub(LOCAL_MODEL_PATH, model_version)

    return LOCAL_MODEL_PATH


def _upload_to_hf_hub(model_path: Path, model_version) -> None:
    """Tải file model pickle lên kho lưu trữ Hugging Face Hub."""
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

    # Tạo repo trên Hugging Face Hub nếu chưa tồn tại
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    logger.info(f"HF Hub repo ready: https://huggingface.co/{hf_repo_id}")

    # Tải file pickle lên Hugging Face Hub
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
    parser = argparse.ArgumentParser(description="Export model from MLflow -> local + HF Hub")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Tùy chọn tải lên Hugging Face Hub (yêu cầu biến HF_TOKEN và HF_REPO_ID)",
    )
    args = parser.parse_args()
    export_model(upload=args.upload)