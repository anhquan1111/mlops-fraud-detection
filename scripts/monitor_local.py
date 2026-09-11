"""Local batch monitoring commands. No AWS calls, retraining or model promotion.

Run from the repository root: uv run python -m scripts.monitor_local --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc dự án luôn có trong sys.path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, RAW_DATA_PATH, REPORTS_DIR
from src.monitor import (
    REFERENCE_BASELINE_PATH,
    _write_json,
    get_or_create_reference_baseline,
    matured_cohort_audit,
    run_drift_analysis,
)


def read_frame(path: Path) -> pd.DataFrame:
    """Read a local table, preserving transaction IDs such as '001' in CSV."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"transaction_id": "string"})
    raise ValueError("Expected a .csv or .parquet file")


def synthetic_demo(scenario: str, output_dir: Path) -> dict:
    """Explicit synthetic demonstration of quality/drift; not real model evaluation."""
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(rng.normal(size=(500, 29)), columns=FEATURE_COLS)
    reference["Amount"] = rng.lognormal(4, 1, len(reference))
    current = reference.copy()
    if scenario == "shift":
        current.loc[:, FEATURE_COLS[:15]] += 5
    elif scenario == "broken":
        current.loc[0, "Amount"] = np.nan
    summary = run_drift_analysis(current, reference, output_dir)
    summary["data_source"] = "synthetic_demo_only"
    _write_json(output_dir / "drift_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run one explicit batch job. Exit 2 means quality failure; 1 means job error."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="Freeze raw TRAIN features + provenance")
    baseline.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    baseline.add_argument("--output", type=Path, default=REFERENCE_BASELINE_PATH)
    baseline.add_argument("--sample-size", type=int, default=5000)
    baseline.add_argument(
        "--force", action="store_true", help="Explicitly replace existing baseline"
    )
    drift = commands.add_parser("drift", help="Compare raw feature windows, excluding labels/IDs")
    drift.add_argument("--current", type=Path, required=True)
    drift.add_argument("--reference", type=Path, help="Explicit reference; verify its provenance")
    drift.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    demo = commands.add_parser("demo", help="Synthetic examples, never a production fallback")
    demo.add_argument("--scenario", choices=["stable", "shift", "broken"], default="stable")
    demo.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "demo")
    cohort = commands.add_parser("cohort", help="Audit saved predictions with delayed labels")
    cohort.add_argument("--scored", type=Path, required=True)
    cohort.add_argument("--labels", type=Path, required=True)
    cohort.add_argument("--as-of", required=True, help="UTC timestamp, e.g. 2026-09-10T00:00:00Z")
    cohort.add_argument("--label-delay-days", type=int, default=7)
    cohort.add_argument("--output", type=Path, default=REPORTS_DIR / "cohort_summary.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "baseline":
            frame = get_or_create_reference_baseline(
                sample_size=args.sample_size,
                force_recreate=args.force,
                raw_path=args.raw,
                baseline_path=args.output,
            )
            result = {"status": "BASELINE_READY", "rows": len(frame), "path": str(args.output)}
        elif args.command == "drift":
            reference = read_frame(args.reference) if args.reference else None
            result = run_drift_analysis(read_frame(args.current), reference, args.output_dir)
        elif args.command == "demo":
            result = synthetic_demo(args.scenario, args.output_dir)
        else:
            result = matured_cohort_audit(
                read_frame(args.scored),
                read_frame(args.labels),
                args.as_of,
                args.label_delay_days,
            )
            _write_json(args.output, result)
    except (ValueError, OSError, KeyError) as exc:
        result = {"status": "ERROR", "error": str(exc), "drift": None}
        if args.command in {"drift", "demo"}:
            _write_json(args.output_dir / "drift_summary.json", result)
        elif args.command == "cohort":
            _write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "QUALITY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
