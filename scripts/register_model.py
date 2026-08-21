"""DEPRECATED — session-2 one-shot script. Use scripts/select_best_model.py instead.

This registers a single hard-coded run under the name 'fraud-detection-baseline',
which is NOT the name the API serves ('fraud-detection-model', see src/config.py).
It predates the validation gate and performs no metric checks whatsoever.

Kept only as a record of the session-2 baseline registration. The supported path is:

    uv run python scripts/select_best_model.py

which ranks every run, enforces the validation gate (src/validate.py), and only
then moves the 'production' alias.

Usage (not recommended):
    uv run python scripts/register_model.py --i-know-this-is-deprecated
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "fraud-detection-baseline"
BEST_RUN_ID = "b989f5796e66402a99ee5a2965eb6732"  # PR-AUC=0.7156, Recall=0.9184


def register_model() -> None:
    """Register best MLflow run into Model Registry with alias 'production'."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_uri = f"runs:/{BEST_RUN_ID}/model"
    logger.info(f"Registering model from run {BEST_RUN_ID} ...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    logger.info(
        f"Registered: {REGISTERED_MODEL_NAME} v{model_version.version} (run_id={BEST_RUN_ID})"
    )

    # Set alias 'production'
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="production",
        version=model_version.version,
    )
    logger.info(f"✅ Alias 'production' → version {model_version.version}")

    # Add description
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=(
            "Logistic Regression baseline -- PR-AUC=0.7156, Recall=0.9184, Precision=0.0588 "
            "on creditcard.csv test set (session 2)."
        ),
    )
    logger.info("Model description updated.")


if __name__ == "__main__":
    import sys

    if "--i-know-this-is-deprecated" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    register_model()
