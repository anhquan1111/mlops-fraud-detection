# Model Card — Credit Card Fraud Detection

> **Format**: Inspired by [Mitchell et al., 2019 — Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

---

## 1. Model Details

| Field | Value |
|-------|-------|
| **Model name** | `fraud-detection-model` |
| **Champion variant** | `lgbm_regularized` |
| **Algorithm** | LightGBM (Gradient Boosting Decision Trees) |
| **MLflow alias** | `production` |
| **Registered model** | `fraud-detection-model@production` |
| **Training date** | August 2026 |
| **Framework** | LightGBM ≥ 4.5.0 · scikit-learn ≥ 1.5.0 |
| **Serving** | FastAPI + uvicorn · Docker |
| **Version** | 1.0.0 |

### Champion Hyperparameters (`lgbm_regularized`)

```python
{
    "n_estimators": 300,
    "max_depth": -1,          # no limit (LGBM default)
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 5,          # lower = less regularization, better for rare class
    "class_weight": "balanced",      # imbalance handling
    "metric": "average_precision",   # PR-AUC equivalent
    "early_stopping_rounds": 30,     # watched on the VALIDATION split
    "random_state": 42,
}
```

Stopped at iteration **74** of 300, chosen on validation.

> Note: `subsample` has no effect in LightGBM unless `subsample_freq > 0`, which is not set here.
> It is left in place so the grid stays comparable with the XGBoost configurations, but bagging is
> effectively disabled for every LightGBM run in this project.

---

## 2. Intended Use

### Primary Use Case

Real-time detection of fraudulent credit card transactions to flag suspicious activity for human review.

### Intended Users

- Fraud analysts reviewing flagged transactions
- MLOps engineers maintaining and retraining the pipeline
- Data scientists benchmarking new approaches

### Out-of-Scope Uses

- ⚠️ **Automated transaction blocking without human review** — false positive rate must be evaluated against customer experience impact
- ❌ **Non-financial fraud domains** — model was trained exclusively on credit card transaction patterns
- ❌ **Identity verification** — model predicts transaction patterns, not cardholder identity
- ❌ **Long-term deployment without retraining** — concept drift from evolving fraud patterns expected over time

---

## 3. Training Data

| Property | Detail |
|----------|--------|
| **Dataset** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Source** | ULB Machine Learning Group — European cardholders |
| **Period** | September 2013 (2 days) |
| **Total transactions** | 284,807 |
| **Fraud transactions** | 492 (0.173%) |
| **Legitimate transactions** | 284,315 (99.827%) |
| **Train split** | 227,845 samples (80%) — stratified |
| **Test split** | 56,962 samples (20%) — stratified |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `V1`–`V28` | float | PCA-transformed components (anonymized — original features confidential) |
| `Amount` | float | Transaction amount in EUR (scaled via StandardScaler) |
| `Time` | — | **Dropped** — timestamp not predictive for cross-day generalization |
| `Class` | int | Label: 0 = legitimate, 1 = fraud |

> ⚠️ **Privacy Note**: Features V1–V28 are PCA-transformed to protect cardholder privacy. Original features are not disclosed. This means feature importance interpretations refer to PCA components, not raw transaction attributes.

### Imbalance Handling

- **Strategy**: `class_weight='balanced'` (LightGBM sklearn API)
- **Effect**: Per-sample weights are inversely proportional to class frequency — fraud samples receive ~577× higher weight during training
- **Why not SMOTE**: Avoided to eliminate risk of data leakage if applied before train/test split

---

## 4. Evaluation Results

### Evaluation Protocol

Stratified **64 / 16 / 20** train / validation / test split (`random_state=42`), test carved out
first. Early stopping, champion selection and the validation gate all read the **validation**
split; the **test** split (56,962 rows, **98 frauds**) is scored once per run for reporting and
influences no decision. Before 2026-08-21 the pipeline early-stopped on the test set and fitted
the `Amount` scaler before splitting — see [`leakage_fix.md`](leakage_fix.md).

### Primary Metrics — all 7 runs

> **Business thresholds** (from AGENTS.md): Recall ≥ 0.80 · Precision ≥ 0.50, evaluated on **val**.

| Run | val PR-AUC | val Recall | val Prec | test PR-AUC | test Recall | test Prec | test F1 | TP | FP | FN | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lgbm_large` | 0.8160 | 0.7975 | 0.8289 | 0.8703 | 0.8469 | 0.7615 | 0.8019 | 83 | 26 | 15 | ❌ recall |
| `lgbm_default` | 0.8038 | 0.8228 | 0.4815 | 0.8496 | 0.8878 | 0.4652 | 0.6105 | 87 | 100 | 11 | ❌ precision |
| `xgb_default` | 0.7899 | 0.7848 | 0.8267 | 0.8604 | 0.8265 | 0.7714 | 0.7980 | 81 | 24 | 17 | ❌ recall |
| **`lgbm_regularized`** ⭐ | **0.7407** | **0.8354** | **0.5238** | 0.7462 | 0.8878 | 0.4555 | 0.6021 | 87 | 104 | 11 | ✅ **champion** |
| `xgb_regularized` | 0.7135 | 0.7848 | 0.4593 | 0.7096 | 0.8571 | 0.4200 | 0.5638 | 84 | 116 | 14 | ❌ both |
| `xgb_deep` | 0.6831 | 0.7342 | 0.5000 | 0.6932 | 0.8061 | 0.4647 | 0.5896 | 79 | 91 | 19 | ❌ recall |
| Logistic Regression | 0.6755 | 0.8861 | 0.0591 | 0.7105 | 0.9082 | 0.0606 | 0.1137 | 89 | 1379 | 9 | ❌ precision |

The gate rejects **6 of 7** runs. Note that **ROC-AUC would rank these very differently** — every
run here scores above 0.95 on ROC-AUC, including the Logistic Regression that raises 1,379 false
alarms to catch 89 frauds. That divergence is the clearest argument in this project for ranking on
PR-AUC rather than ROC-AUC.

### Champion vs Baseline (held-out test)

| Metric | Logistic Regression | `lgbm_regularized` | Δ |
|--------|--------------------|--------------------|--------------|
| PR-AUC | 0.7105 | **0.7462** | +0.0357 (+5.0%) |
| Recall | 0.9082 | 0.8878 | −0.0204 (2 fewer frauds caught) |
| Precision | 0.0606 | **0.4555** | **+7.5×** |
| F1 | 0.1137 | **0.6021** | **+5.3×** |
| TP / FP | 89 / 1,379 | 87 / **104** | **−1,275 false alarms** |

The baseline catches 2 more frauds and raises **thirteen times** as many false alarms doing it.
At any realistic review capacity it is unusable, which is why the gate's precision floor rejects
it automatically.

### Why not `lgbm_large`?

`lgbm_large` has the best test numbers in the table (PR-AUC 0.8703, precision 0.7615, 26 false
positives) but is **rejected by the gate**: its validation recall is 0.7975 against a required
0.80 — short by 0.0025, roughly one fraud out of the 79 in the validation split. Selecting it
anyway would mean overriding the gate on the basis of test numbers, which is precisely the
selection leak the pipeline was rewritten to remove. The honest reading is that the thresholds
themselves need re-deriving from real review capacity; see §6 and `leakage_fix.md` §5.

### Decision Threshold Analysis

Measured on the held-out test set (56,962 transactions, **98 frauds**) with the registered
champion `lgbm_regularized`. Actual sweep results, not estimates:

| Threshold | Recall | Precision | F1 | TP | FP | FN | Use case |
|-----------|--------|-----------|-----|----|----|----|----------|
| 0.1 | 0.9184 | 0.1100 | 0.1965 | 90 | 728 | 8 | Unusable — 1 alert in 9 is real |
| 0.2 | 0.8980 | 0.2085 | 0.3385 | 88 | 334 | 10 | |
| 0.3 | 0.8878 | 0.2979 | 0.4462 | 87 | 205 | 11 | |
| 0.4 | 0.8878 | 0.3718 | 0.5241 | 87 | 147 | 11 | |
| **0.5** (deployed) | **0.8878** | **0.4555** | **0.6021** | **87** | **104** | **11** | **Neutral default** |
| 0.6 | 0.8878 | 0.5472 | 0.6770 | 87 | 72 | 11 | Clears the 0.50 precision floor |
| 0.7 | 0.8878 | 0.6259 | 0.7342 | 87 | 52 | 11 | **Same recall, half the false alarms** |
| 0.8 | 0.8673 | 0.7025 | 0.7763 | 85 | 36 | 13 | High precision |
| 0.9 | 0.8265 | 0.7570 | 0.7902 | 81 | 26 | 17 | Max precision |

Three things worth stating plainly:

1. **Recall is completely flat from 0.3 to 0.7** at 0.8878 — the same 87 frauds — while false
   positives fall from 205 to 52. Between 0.5 and 0.7 this model gives up *nothing* and halves the
   review workload twice over. The deployed default of 0.5 is not where this model belongs.
2. **0.5 does not clear the project's own precision floor on test** (0.4555 < 0.50), even though
   it cleared it on validation (0.5238). Threshold 0.6 or 0.7 would. This is discussed in
   [`leakage_fix.md` §5](leakage_fix.md) — it is a symptom of thresholds calibrated against
   pre-leak-fix numbers, not of the model being broken.
3. **The threshold is nevertheless not changed here.** Pricing a missed fraud against an analyst's
   review time is a business input this project does not have, and `AGENTS.md` requires the
   decision be taken with the project owner rather than read off a metric.

For contrast, the Logistic Regression baseline at threshold 0.5 produces **TP=89, FP=1,379,
FN=9** — it catches 2 more frauds at the cost of ~13× more false alarms.

> ⚠️ Threshold change is a **business decision** — contact project owner before modifying.
> `DECISION_THRESHOLD` lives in `src/config.py`.

---

## 5. Model Tracking & Reproducibility

All experiments are tracked in **MLflow**:

- **Tracking URI**: `sqlite:///mlflow.db` (local SQLite)
- **Experiment**: `fraud-detection`
- **Total runs**: 7 (1 baseline + 3 XGBoost variants + 3 LightGBM variants)
- **Parameters logged**: all hyperparameters, threshold, imbalance strategy, train/test sizes
- **Metrics logged**: PR-AUC, Recall, Precision, F1, ROC-AUC, best_iteration
- **Artifacts logged**: trained model, PR curve figures

To view all experiments:
```bash
uv run mlflow ui  # → http://localhost:5000
```

---

## 6. Limitations & Known Issues

### Data Limitations

- **Temporal scope**: Dataset covers only 2 days in September 2013 — fraud patterns may have evolved significantly
- **Geographic scope**: European cardholders only — may not generalize to other regions
- **Historical data**: Fraud patterns evolve — model expects **quarterly retraining** at minimum
- **PCA anonymization**: V1–V28 cannot be interpreted in domain terms — limits debugging of edge cases

### Methodological Limitations

Two data-leakage defects were present until 2026-08-21 and have been **fixed**: the `Amount`
scaler was fitted before the train/test split, and early stopping used the test set as its watch
list. Full diagnosis, before/after measurements and the reasoning are in
[`leakage_fix.md`](leakage_fix.md). On identical test rows, removing them moved the same
configuration (`lgbm_large`) from PR-AUC 0.8770 / precision 0.8485 / 15 FP to 0.8703 / 0.7615 / 26.

What remains open, honestly stated:

- **Selection runs on a single validation split.** 45,569 rows containing only **79 frauds** is a
  noisy basis for a precision estimate, and taking the maximum over seven candidates biases the
  winner upward. It shows: the promoted champion clears the precision floor on validation
  (0.5238) and misses it on test (0.4555). Stratified k-fold on train+val, selecting on the mean,
  is the change most likely to close that gap.
- **The gate's thresholds were calibrated against pre-fix numbers.** `MIN_RECALL = 0.80` and
  `MIN_PRECISION = 0.50` were set when metrics were inflated. Under honest measurement the two
  models with the best test performance are rejected for missing recall by 0.0025 and 0.0152.
  Re-deriving both from real review capacity is a business conversation, not a code change.
- **No refit on train+val after selection.** The champion is trained on 64% of the data and
  shipped as-is. Refitting the chosen configuration on train+val at the selected iteration count
  would use 80% and typically helps, but it forfeits the ability to re-verify the exact artifact
  against validation, so it was left out deliberately.
- **The deployed artifact is a bare estimator, not a pipeline.** `src/api.py` re-applies the
  `Amount` scaling from constants in `src/config.py`. A test asserts those constants match the
  scaler fitted from the real training split, but bundling the scaler with the model would remove
  the failure mode entirely.

### Model Limitations

- **Calibration**: Probability outputs are directionally correct but not perfectly calibrated — use Platt scaling for exact probability interpretation
- **Concept drift**: Fraud tactics change over time; model performance will degrade without retraining
- **Rare event sensitivity**: At 0.173% base rate, even small threshold changes significantly impact operational metrics

### Infrastructure Limitations

- **No online learning**: Model is retrained in batch — fraud patterns between training runs are not captured
- **Single model serving**: No ensemble or model versioning at serve time

---

## 7. Ethical Considerations

### False Positive Impact (legitimate transaction flagged as fraud)

- Customer experience degradation: card declined, transaction disrupted
- Customer support burden
- **Mitigation**: Precision ≥ 0.85 threshold maintained in champion model

### False Negative Impact (fraud transaction missed)

- Direct financial loss to cardholders or issuing bank
- **Mitigation**: Recall ≥ 0.85 maintained; model used as decision support, not sole arbiter

### Potential Biases

- **Demographic bias**: PCA anonymization hides potential demographic correlates in V1–V28. If original features included spending patterns correlated with demographics, the model may exhibit disparate impact — **cannot be verified without original features**
- **Temporal bias**: Model trained on 2013 data — spending patterns differ significantly from present day

### Fairness Recommendation

> Before production deployment in a real financial system, conduct a **disparate impact audit** using original non-anonymized features if available, assessing fraud detection rates across demographic groups.

---

## 8. Serving & Deployment

### Inference Pipeline

```
Raw transaction (JSON)
    ↓
Pydantic validation (FastAPI)
    ↓
Feature preprocessing:
    - V1–V28: pass-through (already PCA-scaled)
    - Amount: StandardScaler (μ=88.35, σ=250.12)
    ↓
LightGBM.predict_proba() → fraud probability [0,1]
    ↓
Threshold (0.5) → is_fraud: bool
    ↓
JSON response: { fraud_probability, is_fraud, threshold, model_name }
```

### Model Loading Strategy (priority order)

1. `MODEL_PATH` env var → local `.pkl` file (Docker with bundled model)
2. `HF_REPO_ID` env var → download from HuggingFace Hub (Render deploy)
3. Fallback → MLflow Registry alias `production` (local dev)

### Performance

Measured in-process against the registered champion via `TestClient` (300 iterations after
warm-up, request handling + preprocessing + inference; excludes network transit):

| Endpoint | Median | p95 | Per transaction |
|----------|--------|-----|-----------------|
| `POST /predict` (1 tx) | **1.58 ms** | 2.45 ms | 1.58 ms |
| `POST /predict/batch` (10 tx) | 2.33 ms | 3.88 ms | 0.233 ms |
| `POST /predict/batch` (100 tx) | 5.99 ms | 7.94 ms | **0.060 ms** |

Batching is worth using: 100 transactions cost 5.99 ms in one call versus ~158 ms as 100 separate
calls, because `predict_proba` is invoked once for the whole frame rather than once per row. The
per-request fixed cost — HTTP handling, Pydantic validation, DataFrame construction — dominates
single-transaction latency, not the model itself.

- **Batch cap**: 100 transactions per `/predict/batch` call (returns 422 above that)
- **Deployment**: single uvicorn worker on Render free tier; expect a cold start on first request
  after idle, since the model is downloaded from Hugging Face Hub at startup

---

## 9. Maintenance & Retraining

### When to Retrain

| Trigger | Action |
|---------|--------|
| Recall drops below 0.75 in production monitoring | Urgent retraining |
| PR-AUC drops below 0.80 | Scheduled retraining |
| New fraud patterns identified by analysts | Feature engineering review + retraining |
| Quarterly schedule | Preventive retraining |

### Retraining Procedure

```bash
# 1. Download fresh data
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip

# 2. Run training pipeline
uv run python src/train.py

# 3. Select best model (validation gate runs automatically)
uv run python scripts/select_best_model.py

# 4. Verify new champion meets thresholds
# → MLflow Registry: fraud-detection-model@production updated
```

---

## 10. Citation

If you use this pipeline or model card in research:

```bibtex
@misc{fraud-detection-mlops-2026,
  author = {anhquan1111},
  title = {MLOps Fraud Detection Pipeline},
  year = {2026},
  url = {https://github.com/anhquan1111/mlops-fraud-detection}
}
```

**Dataset citation**:
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
