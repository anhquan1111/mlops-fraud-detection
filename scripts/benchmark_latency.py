"""Đo lường độ trễ (latency) của API dự đoán gian lận khi gửi request."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Load fraud_model.pkl (LightGBM champion) or baseline_lr.pkl
model_path = Path("models/fraud_model.pkl")
if not model_path.exists():
    model_path = Path("models/baseline_lr.pkl")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
sample_1 = pd.DataFrame([np.random.randn(29)], columns=feature_cols)
sample_10 = pd.DataFrame(np.random.randn(10, 29), columns=feature_cols)
sample_100 = pd.DataFrame(np.random.randn(100, 29), columns=feature_cols)

# 1. Warmup
for _ in range(50):
    model.predict_proba(sample_1)

# 2. Benchmark Single (300 iterations)
times_1 = []
for _ in range(300):
    t0 = time.perf_counter()
    model.predict_proba(sample_1)
    times_1.append((time.perf_counter() - t0) * 1000)

# 3. Benchmark Batch 10 (300 iterations)
times_10 = []
for _ in range(300):
    t0 = time.perf_counter()
    model.predict_proba(sample_10)
    times_10.append((time.perf_counter() - t0) * 1000)

# 4. Benchmark Batch 100 (300 iterations)
times_100 = []
for _ in range(300):
    t0 = time.perf_counter()
    model.predict_proba(sample_100)
    times_100.append((time.perf_counter() - t0) * 1000)

med_1 = np.median(times_1)
p95_1 = np.percentile(times_1, 95)
med_10 = np.median(times_10)
med_100 = np.median(times_100)

print("\n" + "=" * 65)
print("  INFERENCE LATENCY BENCHMARK RESULT (300 ITERATIONS)")
print("=" * 65)
print(f"1. Single tx (1 sample)  : Median = {med_1:.3f} ms | p95 = {p95_1:.3f} ms")
print(f"2. Batch 10 tx (10 rows) : Total  = {med_10:.3f} ms | Per tx = {med_10 / 10:.4f} ms/tx")
print(f"3. Batch 100 tx (100 rows): Total  = {med_100:.3f} ms | Per tx = {med_100 / 100:.4f} ms/tx")
print("=" * 65 + "\n")
