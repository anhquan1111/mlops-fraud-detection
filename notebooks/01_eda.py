"""Exploratory Data Analysis — Credit Card Fraud Detection Dataset.

Run:
    uv run python notebooks/01_eda.py

Outputs (saved to notebooks/figures/):
    - class_distribution.png   — fraud vs legit count + percentage
    - amount_distribution.png  — transaction amount by class (log scale)
    - feature_distributions.png — KDE plots for V1-V10 (sample)
    - correlation_heatmap.png  — correlation matrix of all features
    - time_distribution.png    — transactions over time by class
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RAW_DATA_PATH, TARGET_COL

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print(f"Loading dataset from {RAW_DATA_PATH} ...")
df = pd.read_csv(RAW_DATA_PATH)

# ---------------------------------------------------------------------------
# 1. Basic overview
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape:           {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage:    {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
print(f"Missing values:  {df.isnull().sum().sum()}")
print("\nClass distribution:")
counts = df[TARGET_COL].value_counts()
for cls, cnt in counts.items():
    label = "Fraud" if cls == 1 else "Legit"
    pct = cnt / len(df) * 100
    print(f"  {label} ({cls}): {cnt:,} ({pct:.4f}%)")

print(f"\nAmount stats (all):\n{df['Amount'].describe().round(2).to_string()}")
fraud_only = df[df[TARGET_COL] == 1]["Amount"].describe().round(2).to_string()
print(f"\nAmount stats (fraud only):\n{fraud_only}")

# ---------------------------------------------------------------------------
# 2. Class distribution bar chart
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels = ["Legit (0)", "Fraud (1)"]
counts_vals = [counts.get(0, 0), counts.get(1, 0)]
colors = ["#2196F3", "#F44336"]

axes[0].bar(labels, counts_vals, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Transaction Count by Class", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Number of Transactions")
for i, v in enumerate(counts_vals):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

# Pie chart
pcts = [c / len(df) * 100 for c in counts_vals]
axes[1].pie(
    counts_vals,
    labels=[f"{lbl}\n({p:.3f}%)" for lbl, p in zip(labels, pcts)],
    colors=colors,
    autopct="%1.4f%%",
    startangle=90,
    pctdistance=0.75,
)
axes[1].set_title("Class Distribution (%)", fontsize=14, fontweight="bold")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n✅ Saved: class_distribution.png")

# ---------------------------------------------------------------------------
# 3. Transaction Amount by class
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fraud = df[df[TARGET_COL] == 1]["Amount"]
legit = df[df[TARGET_COL] == 0]["Amount"]

# Log-scale histogram
axes[0].hist(legit + 1, bins=80, alpha=0.7, color="#2196F3", label="Legit", density=True)
axes[0].hist(fraud + 1, bins=80, alpha=0.7, color="#F44336", label="Fraud", density=True)
axes[0].set_xscale("log")
axes[0].set_xlabel("Amount (log scale)")
axes[0].set_ylabel("Density")
axes[0].set_title("Transaction Amount Distribution (log scale)", fontsize=13)
axes[0].legend()

# Box plots
axes[1].boxplot(
    [np.log1p(legit), np.log1p(fraud)],
    tick_labels=["Legit", "Fraud"],
    patch_artist=True,
    boxprops=dict(facecolor="#90CAF9"),
    medianprops=dict(color="red", linewidth=2),
)
axes[1].set_ylabel("log(Amount + 1)")
axes[1].set_title("Amount Boxplot (log scale)", fontsize=13)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✅ Saved: amount_distribution.png")

# ---------------------------------------------------------------------------
# 4. Feature distributions — V1 to V10 (sample)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

sample_features = [f"V{i}" for i in range(1, 11)]
for i, feat in enumerate(sample_features):
    fraud_vals = df[df[TARGET_COL] == 1][feat]
    legit_vals = df[df[TARGET_COL] == 0][feat].sample(
        n=min(5000, len(df[df[TARGET_COL] == 0])), random_state=42
    )

    axes[i].hist(legit_vals, bins=50, alpha=0.6, color="#2196F3", density=True, label="Legit")
    axes[i].hist(fraud_vals, bins=50, alpha=0.8, color="#F44336", density=True, label="Fraud")
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Value")
    if i == 0:
        axes[i].legend(fontsize=8)

plt.suptitle(
    "Feature Distributions: V1–V10 (Fraud vs Legit)", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✅ Saved: feature_distributions.png")

# ---------------------------------------------------------------------------
# 5. Correlation heatmap (V1–V10 + Amount + Class)
# ---------------------------------------------------------------------------

corr_cols = [f"V{i}" for i in range(1, 11)] + ["Amount", TARGET_COL]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation Heatmap — V1–V10, Amount, Class", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✅ Saved: correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 6. Transactions over Time
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].hist(
    df[df[TARGET_COL] == 0]["Time"] / 3600, bins=100, alpha=0.7, color="#2196F3", label="Legit"
)
axes[0].hist(
    df[df[TARGET_COL] == 1]["Time"] / 3600, bins=100, alpha=0.9, color="#F44336", label="Fraud"
)
axes[0].set_xlabel("Time (hours)")
axes[0].set_ylabel("Count")
axes[0].set_title("Transaction Distribution Over Time", fontsize=13)
axes[0].legend()

# Fraud rate over time (hourly)
df["hour"] = (df["Time"] / 3600).astype(int)
hourly = df.groupby("hour")[TARGET_COL].agg(["sum", "count"])
hourly["rate"] = hourly["sum"] / hourly["count"] * 100
axes[1].plot(hourly.index, hourly["rate"], color="#E91E63", linewidth=1.5)
axes[1].fill_between(hourly.index, hourly["rate"], alpha=0.3, color="#E91E63")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].set_title("Fraud Rate by Hour", fontsize=13)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "time_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✅ Saved: time_distribution.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETE")
print(f"All figures saved to: {FIGURES_DIR}")
print("=" * 60)

# Top features correlated with fraud
target_corr = (
    df[[c for c in df.columns if c != "hour"]]
    .corr()[TARGET_COL]
    .drop(TARGET_COL)
    .abs()
    .sort_values(ascending=False)
)
print("\nTop 10 features correlated with fraud (|correlation|):")
for feat, val in target_corr.head(10).items():
    print(f"  {feat:<12}: {val:.4f}")
