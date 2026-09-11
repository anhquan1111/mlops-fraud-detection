"""Exercise the local file-based workflow, including failure exit codes."""

import json

import pandas as pd

from scripts.monitor_local import main, read_frame


def test_csv_reader_preserves_leading_zero_ids(tmp_path):
    path = tmp_path / "scored.csv"
    path.write_text("transaction_id,prediction\n001,0\n002,1\n")
    assert read_frame(path)["transaction_id"].tolist() == ["001", "002"]


def test_explicit_broken_demo_returns_quality_exit_code(tmp_path, capsys):
    code = main(["demo", "--scenario", "broken", "--output-dir", str(tmp_path)])
    assert code == 2
    summary = json.loads((tmp_path / "drift_summary.json").read_text())
    assert summary["data_source"] == "synthetic_demo_only"
    assert summary["drift"] is None


def test_missing_current_file_marks_latest_as_error(tmp_path, capsys):
    (tmp_path / "drift_summary.json").write_text('{"status": "SUCCESS"}')
    code = main(
        ["drift", "--current", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path)]
    )
    assert code == 1
    assert json.loads((tmp_path / "drift_summary.json").read_text())["status"] == "ERROR"


def test_cohort_command_joins_files_and_writes_summary(tmp_path, capsys):
    scored = tmp_path / "scored.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "transaction_id": ["001", "002"],
            "event_time": ["2026-09-01"] * 2,
            "prediction": [1, 0],
            "fraud_probability": [0.9, 0.1],
        }
    ).to_csv(scored, index=False)
    pd.DataFrame(
        {
            "transaction_id": ["001", "002"],
            "label_available_at": ["2026-09-08"] * 2,
            "Class": [1, 0],
        }
    ).to_csv(labels, index=False)
    output = tmp_path / "cohort.json"
    assert (
        main(
            [
                "cohort",
                "--scored",
                str(scored),
                "--labels",
                str(labels),
                "--as-of",
                "2026-09-10T00:00:00Z",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["label_coverage"] == 1
    assert result["metrics"]["pr_auc"] == 1
