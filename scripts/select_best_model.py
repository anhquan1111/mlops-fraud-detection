"""Auto-select best model from MLflow experiment and register to Model Registry.

Selection criteria (all read VALIDATION metrics — the test split never
influences which model is chosen; see docs/leakage_fix.md):
    1. Filter: val_recall >= MIN_RECALL AND val_precision >= MIN_PRECISION
    2. Sort by: val_pr_auc descending
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
        order_by=["metrics.val_pr_auc DESC"],
    )

    results = []
    for run in runs:
        metrics = run.data.metrics
        tags = run.data.tags
        params = run.data.params

        # Only runs from the leak-free protocol carry val_* metrics. Older runs
        # scored on a test set that had already driven early stopping, so they
        # are not comparable and are excluded from selection entirely.
        if "val_pr_auc" not in metrics:
            continue

        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or "unnamed",
            "model_type": tags.get("model_type", "unknown"),
            "purpose": tags.get("purpose", "unknown"),
            "n_estimators": params.get("n_estimators", "N/A"),
            "threshold": metrics.get("threshold", 0.5),
        }
        for split in ("val", "test"):
            for key in ("pr_auc", "recall", "precision", "f1", "roc_auc", "tp", "fp", "fn"):
                row[f"{split}_{key}"] = metrics.get(f"{split}_{key}", 0.0)
        results.append(row)

    return results


def print_comparison_table(results: list[dict], best_run_id: str | None = None) -> None:
    """Print a formatted comparison table of all runs.

    Args:
        results: List of run dicts from get_all_runs().
        best_run_id: run_id of the selected best model (highlighted).
    """
    print("\n" + "=" * 104)
    print("  ALL RUNS -- fraud-detection experiment (ranked by VAL PR-AUC)")
    print("  Gate decisions use VAL only. TEST columns are reported, never selected on.")
    print("=" * 104)
    print(
        f"  {'Run Name':<20} {'Model Type':<20} "
        f"{'valPR':>6} {'valRec':>7} {'valPrec':>8} "
        f"{'tstPR':>6} {'tstRec':>7} {'tstPrec':>8}  {'Status'}"
    )
    print(f"  {'-' * 100}")

    for r in results:
        recall_ok = r["val_recall"] >= MIN_RECALL
        prec_ok = r["val_precision"] >= MIN_PRECISION
        qualifies = recall_ok and prec_ok

        status_parts = []
        if not recall_ok:
            status_parts.append(f"Recall<({r['val_recall']:.3f}<{MIN_RECALL})")
        if not prec_ok:
            status_parts.append(f"Prec<({r['val_precision']:.3f}<{MIN_PRECISION})")
        if qualifies:
            status_parts.append("[PASS]")
        if r["run_id"] == best_run_id:
            status_parts.append("<- BEST")

        status = " | ".join(status_parts) if status_parts else "-"

        print(
            f"  {r['run_name']:<20} {r['model_type']:<20} "
            f"{r['val_pr_auc']:>6.4f} {r['val_recall']:>7.4f} {r['val_precision']:>8.4f} "
            f"{r['test_pr_auc']:>6.4f} {r['test_recall']:>7.4f} {r['test_precision']:>8.4f}"
            f"  {status}"
        )

    print("=" * 104)
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
        f"Selected on VAL: PR-AUC={best_run['val_pr_auc']:.4f} "
        f"Recall={best_run['val_recall']:.4f} Precision={best_run['val_precision']:.4f} | "
        f"Held-out TEST: PR-AUC={best_run['test_pr_auc']:.4f} "
        f"Recall={best_run['test_recall']:.4f} Precision={best_run['test_precision']:.4f} "
        f"(TP={int(best_run['test_tp'])} FP={int(best_run['test_fp'])}) | "
        "protocol=leakfree_v2"
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
        r for r in all_runs if r["val_recall"] >= MIN_RECALL and r["val_precision"] >= MIN_PRECISION
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
    best = max(qualifying, key=lambda r: r["val_pr_auc"])
    logger.info(
        f"Best model (by val): '{best['run_name']}' | {best['model_type']} | "
        f"val PR-AUC={best['val_pr_auc']:.4f} | val Recall={best['val_recall']:.4f} | "
        f"val Precision={best['val_precision']:.4f}"
    )

    # Print comparison table (before registering)
    print_comparison_table(all_runs, best_run_id=best["run_id"])

    # ------------------------------------------------------------------
    # 4. Validation gate — the authority on production promotion
    #    promote_on_pass=False: the gate decides, register_best_model() below
    #    performs the registration so the version description is filled in.
    # ------------------------------------------------------------------
    gate_metrics = {k: best[f"val_{k}"] for k in ("pr_auc", "recall", "precision", "f1", "roc_auc")}
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
    print("  -- selected on VALIDATION --")
    print(f"  val PR-AUC    : {best['val_pr_auc']:.4f}")
    print(f"  val Recall    : {best['val_recall']:.4f}  (target >= {MIN_RECALL})")
    print(f"  val Precision : {best['val_precision']:.4f}  (target >= {MIN_PRECISION})")
    print(f"  val F1        : {best['val_f1']:.4f}")
    print("  -- held-out TEST (reported once, never selected on) --")
    print(f"  test PR-AUC   : {best['test_pr_auc']:.4f}")
    print(f"  test Recall   : {best['test_recall']:.4f}")
    print(f"  test Precision: {best['test_precision']:.4f}")
    print(f"  test F1       : {best['test_f1']:.4f}")
    print(
        f"  test TP/FP/FN : {int(best['test_tp'])} / "
        f"{int(best['test_fp'])} / {int(best['test_fn'])}"
    )
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
