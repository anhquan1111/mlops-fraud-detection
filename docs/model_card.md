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
positives) but is **rejected by the gate**: validation recall 0.7975 against a required 0.80.

That looks like a miss of 0.0025, which is misleading. With 79 frauds in the validation split,
recall can only take multiples of 1/79 = 0.0127. `lgbm_large` caught **63 of 79**; the next
attainable value is 64/79 = 0.8101. Nothing can score between them, so a floor of 0.80 is in
practice a floor of 64/79 — and `lgbm_large` missed it by **exactly one fraud case**.

It is not promoted anyway. Selecting it would mean overriding the gate on the strength of test
numbers, which is precisely the selection leak the pipeline was rewritten to remove, and it would
reduce the gate to something that only binds when it is convenient. The right response is a
recall estimate stable enough to make the boundary meaningful — stratified k-fold over train+val
would put ~394 frauds behind it instead of 79 — not a lower bar. See
[`leakage_fix.md` §5.4](leakage_fix.md).

### Decision Threshold Analysis

The operating point is chosen on **validation** and verified on test exactly once.
An earlier version of this section swept thresholds on the test set and drew a
recommendation from the result — that is model selection on test, the same defect as
Leak B, and it is recorded in [`leakage_fix.md` §5.1](leakage_fix.md). Reproduce with
`uv run python scripts/select_threshold.py`, which fixes the criterion in code before
producing any number.

**Selection (validation only — 45,569 rows, 79 frauds).** Criterion: recall ≥ 0.80,
then maximise precision; tie-break to the lower threshold. 84 of 99 grid points were
eligible. **Chosen: 0.81** — validation recall 0.8101 (64/79), precision 0.7805,
TP 64, FP 18, FN 15.

**Verification (test — 56,962 rows, 98 frauds — scored once):**

| Threshold | Recall | Precision | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| **0.50** (deployed default) | 0.8878 | 0.4555 ❌ below the 0.50 floor | 0.6021 | 87 | 104 | 11 |
| **0.81** (selected on val) | 0.8673 | **0.7083** ✅ | 0.7798 | 85 | **35** | 13 |

Moving 0.50 → 0.81 costs **2 frauds** and removes **69 false alarms**, and lifts test
precision from below the project's own floor to comfortably above it.

Note the optimism in the val estimate: validation precision 0.7805 against test
precision 0.7083. That gap is the selection bias of taking a maximum over 84
candidates on a 79-fraud split — treat 0.7805 as an overestimate, not a forecast.

> ⚠️ **`DECISION_THRESHOLD` remains 0.50.** The table above is a measurement, not a
> change. Pricing a missed fraud against an analyst's review time is a business input
> this project does not have, and `AGENTS.md` requires the decision be taken with the
> project owner. The analysis exists so the conversation can start from evidence.

For contrast, the Logistic Regression baseline at threshold 0.5 produces **TP=89,
FP=1,379, FN=9** — 2 more frauds caught at roughly 13× the false alarms.

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
- **The gate's recall floor is finer than the data can resolve.** With 79 frauds in the
  validation split, recall moves in steps of 1/79 = 0.0127, so a floor of 0.80 is
  operationally a floor of 64/79 = 0.8101. `lgbm_large` was rejected for missing by
  **exactly one fraud case** (63/79), `xgb_default` by two. The fix is a better estimator —
  k-fold would put ~394 frauds behind the estimate instead of 79 — not a lower bar.
  Details in [`leakage_fix.md` §5.4](leakage_fix.md).
- **The gate's thresholds were calibrated against pre-fix numbers.** `MIN_RECALL = 0.80` and
  `MIN_PRECISION = 0.50` were set when metrics were inflated. Re-deriving both from real
  review capacity is a business conversation, not a code change.
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
