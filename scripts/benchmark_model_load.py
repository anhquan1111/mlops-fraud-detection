"""Measure how long the API takes to obtain its model at startup.

This is the cold-start cost on Render: the container holds no model, so
`src.api.load_model()` downloads the artifact from Hugging Face Hub and
unpickles it before the service can answer its first request.

Two numbers matter and they differ by orders of magnitude:

    cold  — empty cache: network round trip + transfer + joblib.load
    warm  — cache populated: hf_hub_download resolves locally, then joblib.load

Reporting only the warm number would be dishonest about startup latency;
reporting only the cold number ignores that a restarted container on the same
host may still have the cache. Both are printed.

Usage:
    uv run python scripts/benchmark_model_load.py                  # HF Hub (default)
    uv run python scripts/benchmark_model_load.py --repo-id user/repo
    uv run python scripts/benchmark_model_load.py --local models/baseline_lr.pkl

Network measurements depend on the connection and the Hub's response on the
day. Treat a single run as an order of magnitude, not a specification.
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import joblib

from src.config import MODEL_ARTIFACT_FILENAME

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _time_hf_load(repo_id: str, cache_dir: str) -> tuple[float, float, float]:
    """Download the artifact from HF Hub into `cache_dir` and unpickle it.

    Args:
        repo_id: Hugging Face Hub model repo, e.g. "user/fraud-detection-model".
        cache_dir: Cache directory to use. An empty one forces a real download.

    Returns:
        (download_seconds, unpickle_seconds, size_mb).
    """
    from huggingface_hub import hf_hub_download

    start = time.perf_counter()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=MODEL_ARTIFACT_FILENAME,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=cache_dir,
    )
    download = time.perf_counter() - start

    start = time.perf_counter()
    model = joblib.load(path)
    unpickle = time.perf_counter() - start

    assert hasattr(model, "predict_proba"), "loaded object is not a classifier"
    return download, unpickle, Path(path).stat().st_size / 1024 / 1024


def _time_local_load(path: str) -> tuple[float, float]:
    """Unpickle a local artifact.

    Args:
        path: Path to the .pkl file.

    Returns:
        (unpickle_seconds, size_mb).
    """
    start = time.perf_counter()
    model = joblib.load(path)
    unpickle = time.perf_counter() - start
    assert hasattr(model, "predict_proba"), "loaded object is not a classifier"
    return unpickle, Path(path).stat().st_size / 1024 / 1024


def main() -> None:
    """Run the cold/warm benchmark and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", "votrananhquan/fraud-detection-model"),
        help="Hugging Face Hub model repo to benchmark.",
    )
    parser.add_argument(
        "--local",
        metavar="PATH",
        help="Benchmark a local pickle instead of the Hub (no network).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  MODEL LOAD BENCHMARK")
    print("=" * 72)

    if args.local:
        unpickle, size_mb = _time_local_load(args.local)
        print(f"  Source        : local file {args.local}")
        print(f"  Artifact size : {size_mb:.2f} MB")
        print(f"  joblib.load   : {unpickle * 1000:.0f} ms")
        print("=" * 72 + "\n")
        return

    print(f"  Source        : huggingface.co/{args.repo_id}")
    print(f"  Filename      : {MODEL_ARTIFACT_FILENAME}")
    print()

    # Cold: a throwaway empty cache guarantees a real network fetch.
    cold_cache = tempfile.mkdtemp(prefix="hf_cold_")
    try:
        dl_cold, load_cold, size_mb = _time_hf_load(args.repo_id, cold_cache)
        # Warm: same cache, now populated — this is a container restart on a
        # host that already pulled the file.
        dl_warm, load_warm, _ = _time_hf_load(args.repo_id, cold_cache)
    finally:
        shutil.rmtree(cold_cache, ignore_errors=True)

    print(f"  Artifact size : {size_mb:.2f} MB")
    print()
    print(f"  {'Run':<28}{'download':>12}{'joblib.load':>14}{'total':>10}")
    print(f"  {'-' * 64}")
    for label, dl, ld in (
        ("cold (empty cache)", dl_cold, load_cold),
        ("warm (cache populated)", dl_warm, load_warm),
    ):
        print(f"  {label:<28}{dl * 1000:>10.0f} ms{ld * 1000:>12.0f} ms{(dl + ld):>9.2f} s")

    print()
    print("  Cold is the Render cold-start cost: a fresh container holds no cache.")
    print("  Network timings vary by connection and by the Hub's response; treat")
    print("  a single run as an order of magnitude, not a specification.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
