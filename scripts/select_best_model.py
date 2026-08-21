"""Auto-select best model from MLflow experiment and register to Model Registry.

Selection criteria:
    1. Filter: recall >= MIN_RECALL AND precision >= MIN_PRECISION
    2. Sort by: pr_auc descending
    3. Run the validation gate (src/validate.py) on the winner — it re-checks the
       minimum thresholds AND blocks any PR-AUC regression against the model
       currently holding the 'production' alias.
    4. Register winner as 'fraud-detection-model' with alias 'production'

The gate is the single source of truth for "may this model go to production".
This script never promotes a model that the gate rejected.

Usage:
    uv run python scripts/select_best_model.py
    uv run python scripts/select_best_model.py --dry-run   # evaluate, do not promote
"""

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MIN_PRECISION,
    MIN_RECALL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from src.validate import ValidationStatus, print_validation_report, run_validation_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_all_runs(client: MlflowClient, experiment_name: str) -> list[dict]:
    """Fetch all finished runs from the experiment.

    Args:
        client: MLflow client instance.
        experiment_name: Name of the MLflow experiment.

    Returns:
        List of dicts with run metadata and metrics.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.pr_auc DESC"],
    )

    results = []
    for run in runs:
        metrics = run.data.metrics
        tags = run.data.tags
        params = run.data.params

        # Skip runs that don't have pr_auc logged (e.g., incomplete runs)
        if "pr_auc" not in metrics:
            continue

        results.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "unnamed",
                "model_type": tags.get("model_type", "unknown"),
                "purpose": tags.get("purpose", "unknown"),
                "pr_auc": metrics.get("pr_auc", 0.0),
                "recall": metrics.get("recall", 0.0),
                "precision": metrics.get("precision", 0.0),
                "f1": metrics.get("f1", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.0),
                "n_estimators": params.get("n_estimators", "N/A"),
                "threshold": metrics.get("threshold", 0.5),
            }
        )

    return results


def print_comparison_table(results: list[dict], best_run_id: str | None = None) -> None:
    """Print a formatted comparison table of all runs.

    Args:
        results: List of run dicts from get_all_runs().
        best_run_id: run_id of the selected best model (highlighted).
    """
    print("\n" + "=" * 90)
    print("  ALL RUNS -- fraud-detection experiment (sorted by PR-AUC)")
    print("=" * 90)
    print(
        f"  {'Run Name':<24} {'Model Type':<22} {'PR-AUC':>7} "
        f"{'Recall':>7} {'Prec':>7} {'F1':>7}  {'Status'}"
    )
    print(f"  {'-' * 82}")

    for r in results:
        recall_ok = r["recall"] >= MIN_RECALL
        prec_ok = r["precision"] >= MIN_PRECISION
        qualifies = recall_ok and prec_ok

        status_parts = []
        if not recall_ok:
            status_parts.append(f"Recall<({r['recall']:.3f}<{MIN_RECALL})")
        if not prec_ok:
            status_parts.append(f"Prec<({r['precision']:.3f}<{MIN_PRECISION})")
        if qualifies:
            status_parts.append("[PASS]")
        if r["run_id"] == best_run_id:
            status_parts.append("<- BEST")

        status = " | ".join(status_parts) if status_parts else "-"

        print(
            f"  {r['run_name']:<24} {r['model_type']:<22} "
            f"{r['pr_auc']:>7.4f} {r['recall']:>7.4f} "
            f"{r['precision']:>7.4f} {r['f1']:>7.4f}  {status}"
        )

    print("=" * 90)
    print()


def register_best_model(client: MlflowClient, best_run: dict) -> tuple[str, str]:
    """Register the best run as a new model version in MLflow Registry.

    Steps:
        1. Register model from the run's artifact URI.
        2. Set alias 'production' on the new version.
        3. Remove 'production' alias from any previous version.

    Args:
        client: MLflow client instance.
        best_run: Dict with run metadata (from get_all_runs).

    Returns:
        Tuple of (model_name, version).
    """
    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Registering model from run '{best_run['run_name']}' (run_id={run_id}) ...")
    logger.info(f"  -> Model URI: {model_uri}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )
    version = model_version.version
    logger.info(f"Registered: {REGISTERED_MODEL_NAME} v{version}")

    # Update description
    description = (
        f"Model: {best_run['model_type']} | Run: {best_run['run_name']} | "
        f"PR-AUC={best_run['pr_auc']:.4f} | Recall={best_run['recall']:.4f} | "
        f"Precision={best_run['precision']:.4f} | F1={best_run['f1']:.4f} | "
        f"Session: 4"
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=description,
    )

    # Set alias 'production' on the new version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="production",
        version=version,
    )
    logger.info(f"[OK] Alias 'production' -> {REGISTERED_MODEL_NAME} v{version}")

    return REGISTERED_MODEL_NAME, version


def select_and_register(dry_run: bool = False) -> None:
    """Query runs, select the best, run the validation gate, then register.

    Args:
        dry_run: If True, run every check but do not touch the Model Registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ------------------------------------------------------------------
    # 1. Fetch all runs
    # ------------------------------------------------------------------
    logger.info(f"Querying MLflow experiment: '{MLFLOW_EXPERIMENT_NAME}' ...")
    all_runs = get_all_runs(client, MLFLOW_EXPERIMENT_NAME)

    if not all_runs:
        logger.error("No finished runs found in the experiment. Run train.py first.")
        sys.exit(1)

    logger.info(f"Found {len(all_runs)} finished run(s) with metrics.")

    # ------------------------------------------------------------------
    # 2. Filter runs that meet minimum thresholds
    # ------------------------------------------------------------------
    qualifying = [
        r for r in all_runs if r["recall"] >= MIN_RECALL and r["precision"] >= MIN_PRECISION
    ]

    if not qualifying:
        logger.error(
            f"[!!] No runs meet minimum thresholds "
            f"(Recall >= {MIN_RECALL}, Precision >= {MIN_PRECISION})."
        )
        print_comparison_table(all_runs)
        sys.exit(1)

    logger.info(f"{len(qualifying)} run(s) meet minimum thresholds.")

    # ------------------------------------------------------------------
    # 3. Select best by PR-AUC
    # ------------------------------------------------------------------
    best = max(qualifying, key=lambda r: r["pr_auc"])
    logger.info(
        f"Best model: '{best['run_name']}' | {best['model_type']} | "
        f"PR-AUC={best['pr_auc']:.4f} | Recall={best['recall']:.4f} | "
        f"Precision={best['precision']:.4f}"
    )

    # Print comparison table (before registering)
    print_comparison_table(all_runs, best_run_id=best["run_id"])

    # ------------------------------------------------------------------
    # 4. Validation gate — the authority on production promotion
    #    promote_on_pass=False: the gate decides, register_best_model() below
    #    performs the registration so the version description is filled in.
    # ------------------------------------------------------------------
    gate_metrics = {k: best[k] for k in ("pr_auc", "recall", "precision", "f1", "roc_auc")}
    gate_result = run_validation_gate(
        run_id=best["run_id"],
        new_metrics=gate_metrics,
        client=client,
        model_name=REGISTERED_MODEL_NAME,
        promote_on_pass=False,
    )
    print_validation_report(gate_result)

    if gate_result.status == ValidationStatus.REJECTED:
        logger.error(f"Validation gate REJECTED the best run — not promoting. {gate_result.reason}")
        sys.exit(1)

    if dry_run:
        logger.info("--dry-run: gate passed, skipping registration.")
        return

    # ------------------------------------------------------------------
    # 5. Register best model → MLflow Registry with alias 'production'
    # ------------------------------------------------------------------
    model_name, version = register_best_model(client, best)

    # ------------------------------------------------------------------
    # 6. Print final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  [OK] MODEL REGISTRATION COMPLETE")
    print("=" * 60)
    print(f"  Registry name : {model_name}")
    print(f"  Version       : {version}")
    print("  Alias         : production")
    print(f"  Model type    : {best['model_type']}")
    print(f"  Run name      : {best['run_name']}")
    print(f"  PR-AUC        : {best['pr_auc']:.4f}")
    print(f"  Recall        : {best['recall']:.4f}  (target >= {MIN_RECALL})")
    print(f"  Precision     : {best['precision']:.4f}  (target >= {MIN_PRECISION})")
    print(f"  F1            : {best['f1']:.4f}")
    print("=" * 60)
    print()
    print("  Load in API with:")
    print(f"    mlflow.pyfunc.load_model('models:/{model_name}@production')")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the best MLflow run, run the validation gate, and promote it."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run selection and the validation gate without touching the Model Registry.",
    )
    select_and_register(dry_run=parser.parse_args().dry_run)
