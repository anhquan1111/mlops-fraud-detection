"""Select the decision threshold on VALIDATION, then verify it ONCE on test.

Why this script exists
----------------------
A threshold sweep is model selection. Running it on the test set and reading a
recommendation off the result is the same defect as early-stopping on test — it
just looks like analysis rather than training. This project made that mistake
twice: once in the pipeline (fixed, see docs/leakage_fix.md) and once in a
follow-up report that swept thresholds on test. This script exists so the
correct protocol is executable rather than merely described.

Protocol (the criterion is fixed here, in code, before any number is seen):

    1. Sweep candidate thresholds on the VALIDATION split only.
    2. Keep those meeting the project's recall floor (MIN_RECALL).
    3. Among them choose the highest validation precision.
       Tie-break: the lower threshold, which keeps more recall headroom.
    4. Score that single threshold on the test split exactly once and report.

Step 4 is a measurement, not a decision. If the test result is disappointing,
the correct response is to record it — not to return to step 1 with the test
number in mind, which would silently make test the selection set.

This script does NOT modify DECISION_THRESHOLD in src/config.py. Choosing an
operating point prices a missed fraud against an analyst's review time, and per
AGENTS.md that is a business decision, not one to take from a metric.

Usage:
    uv run python scripts/select_threshold.py
    uv run python scripts/select_threshold.py --model-uri runs:/<run_id>/model
"""

import argparse
import logging

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import (
    DECISION_THRESHOLD,
    MIN_PRECISION,
    MIN_RECALL,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    REGISTERED_MODEL_NAME,
)
from src.features import (
    apply_amount_scaler,
    fit_amount_scaler,
    load_data,
    preprocess,
    split_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def _score(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts and rates for one threshold.

    Args:
        y_true: True binary labels.
        proba: Predicted fraud probabilities.
        threshold: Decision threshold.

    Returns:
        Dict with recall, precision, tp, fp, fn.
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, (proba >= threshold).astype(int), labels=[0, 1]
    ).ravel()
    return {
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def select_on_validation(
    y_val: pd.Series, proba_val: np.ndarray
) -> tuple[float, dict[str, float], int]:
    """Choose a threshold using validation data only.

    Args:
        y_val: Validation labels.
        proba_val: Validation fraud probabilities.

    Returns:
        (chosen_threshold, its validation scores, number of eligible thresholds).

    Raises:
        RuntimeError: If no threshold on the grid reaches MIN_RECALL.
    """
    eligible = [(t, _score(y_val, proba_val, t)) for t in GRID]
    eligible = [(t, s) for t, s in eligible if s["recall"] >= MIN_RECALL]

    if not eligible:
        raise RuntimeError(
            f"No threshold on the grid reaches recall >= {MIN_RECALL} on validation. "
            "The model cannot satisfy the project's recall floor at any operating point."
        )

    best_precision = max(s["precision"] for _, s in eligible)
    chosen = min(t for t, s in eligible if s["precision"] == best_precision)
    return chosen, _score(y_val, proba_val, chosen), len(eligible)


def main() -> None:
    """Run the val-select / test-verify protocol and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=f"models:/{REGISTERED_MODEL_NAME}@production",
        help="MLflow model URI to analyse (default: the production champion).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    X, y = preprocess(load_data(RAW_DATA_PATH))
    X_train, X_val, X_test, _, y_val, y_test = split_data(X, y)
    scaler = fit_amount_scaler(X_train)
    X_val = apply_amount_scaler(scaler, X_val)
    X_test = apply_amount_scaler(scaler, X_test)

    model = mlflow.lightgbm.load_model(args.model_uri)
    proba_val = model.predict_proba(X_val)[:, 1]

    # ------------------------------------------------------------------
    # 1-3. Select on validation
    # ------------------------------------------------------------------
    chosen, val_scores, n_eligible = select_on_validation(y_val, proba_val)

    n_val_fraud = int(y_val.sum())
    print("\n" + "=" * 78)
    print("  THRESHOLD SELECTION -- VALIDATION ONLY")
    print("=" * 78)
    print(f"  Validation rows : {len(y_val):,} ({n_val_fraud} frauds)")
    print(f"  Criterion       : recall >= {MIN_RECALL}, then maximise precision")
    print(f"  Eligible grid   : {n_eligible} of {len(GRID)} thresholds")
    print(f"  Chosen          : {chosen:.2f}")
    print(
        f"  Val scores      : recall={val_scores['recall']:.4f} "
        f"precision={val_scores['precision']:.4f} "
        f"TP={val_scores['tp']} FP={val_scores['fp']} FN={val_scores['fn']}"
    )
    print(
        f"  Recall grain    : 1/{n_val_fraud} = {1 / n_val_fraud:.4f} "
        "-- recall can only take these discrete values on val"
    )

    # ------------------------------------------------------------------
    # 4. Verify once on test. Measurement only.
    # ------------------------------------------------------------------
    proba_test = model.predict_proba(X_test)[:, 1]
    default_scores = _score(y_test, proba_test, DECISION_THRESHOLD)
    chosen_scores = _score(y_test, proba_test, chosen)

    print("\n" + "=" * 78)
    print("  VERIFICATION -- TEST, SCORED ONCE")
    print("=" * 78)
    print(f"  Test rows       : {len(y_test):,} ({int(y_test.sum())} frauds)")
    print(f"  {'Threshold':<26}{'Recall':>9}{'Precision':>11}{'TP':>6}{'FP':>7}{'FN':>6}")
    print(f"  {'-' * 63}")
    for label, t, s in (
        (f"deployed default {DECISION_THRESHOLD:.2f}", DECISION_THRESHOLD, default_scores),
        (f"selected on val {chosen:.2f}", chosen, chosen_scores),
    ):
        floor = "" if s["precision"] >= MIN_PRECISION else f"  <- below {MIN_PRECISION} floor"
        print(
            f"  {label:<26}{s['recall']:>9.4f}{s['precision']:>11.4f}"
            f"{s['tp']:>6}{s['fp']:>7}{s['fn']:>6}{floor}"
        )

    d_tp = chosen_scores["tp"] - default_scores["tp"]
    d_fp = chosen_scores["fp"] - default_scores["fp"]
    print(
        f"\n  Moving {DECISION_THRESHOLD:.2f} -> {chosen:.2f} on test: {d_tp:+d} true positives, "
        f"{d_fp:+d} false positives."
    )
    print("\n  This is a measurement, not a recommendation to change the threshold.")
    print("  DECISION_THRESHOLD is a business decision -- see AGENTS.md.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
