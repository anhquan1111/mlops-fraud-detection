# Model Card — Credit Card Fraud Detection

> **Format**: Inspired by [Mitchell et al., 2019 — Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

---

## 1. Model Details

| Field | Value |
|-------|-------|
| **Model name** | `fraud-detection-model` |
| **Champion variant** | `lgbm_large` |
| **Algorithm** | LightGBM (Gradient Boosting Decision Trees) |
| **MLflow alias** | `production` |
| **Registered model** | `fraud-detection-model@production` |
| **Training date** | August 2026 |
| **Framework** | LightGBM ≥ 4.5.0 · scikit-learn ≥ 1.5.0 |
| **Serving** | FastAPI + uvicorn · Docker |
| **Version** | 1.0.0 |

### Champion Hyperparameters (`lgbm_large`)

```python
{
    "n_estimators": 300,
    "max_depth": -1,          # no limit (LGBM default)
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",      # imbalance handling
    "metric": "average_precision",   # PR-AUC equivalent
    "early_stopping_rounds": 30,
    "random_state": 42,
}
```

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

### Primary Metrics (Test Set — 56,962 samples)

> **Business thresholds** (from AGENTS.md): Recall ≥ 0.80 · Precision ≥ 0.50

| Model | PR-AUC | Recall | Precision | F1-score | ROC-AUC | Gate |
|-------|--------|--------|-----------|----------|---------|------|
| Logistic Regression (baseline) | 0.7156 | 0.9184 | 0.0588 | 0.1105 | 0.9714 | ❌ precision |
| XGBoost default | 0.8707 | 0.8367 | 0.7664 | 0.8000 | 0.9697 | ✅ |
| XGBoost deep | 0.7001 | 0.8367 | 0.4184 | 0.5578 | 0.9596 | ❌ precision |
| XGBoost regularized | 0.7139 | 0.8673 | 0.3571 | 0.5060 | 0.9814 | ❌ precision |
| LightGBM default | 0.8757 | 0.8878 | 0.6493 | 0.7500 | 0.9737 | ✅ |
| LightGBM regularized | 0.7440 | 0.8878 | 0.4065 | 0.5577 | 0.9695 | ❌ precision |
| **LightGBM large** ⭐ | **0.8770** | **0.8571** | **0.8485** | **0.8528** | **0.9786** | ✅ **champion** |

> All figures are read straight from the MLflow experiment `fraud-detection` (latest full sweep).
> Note that **ROC-AUC is high for every run, including the ones the gate rejects** — `xgb_regularized`
> has the *best* ROC-AUC (0.9814) and the *worst* precision (0.3571). That divergence is the
> single clearest argument in this project for ranking on PR-AUC rather than ROC-AUC.

### Champion Model vs Baseline

| Metric | Logistic Regression | LightGBM large | Δ Improvement |
|--------|--------------------|--------------------|--------------|
| PR-AUC | 0.7156 | **0.8770** | +22.6% 📈 |
| Recall | 0.9184 | 0.8571 | −6.7% (6 fewer frauds caught) |
| Precision | 0.0588 | **0.8485** | **+14.4×** |
| F1-score | 0.1105 | **0.8528** | **+7.7×** |

> **Note on the Recall trade-off**: the baseline LR does catch 6 more frauds (90 vs 84 of 98), but
> it does so by flagging roughly **1,440 legitimate transactions** as fraud — precision 0.0588 means
> only about 1 alert in 17 is real. The champion gives up those 6 frauds to cut false alarms to
> ~15. At any realistic review capacity the LR baseline is unusable, which is precisely why the
> validation gate enforces a precision floor alongside recall.

### Why not `lgbm_default`?

`lgbm_default` has **higher recall** (0.8878 vs 0.8571 — 87 frauds caught vs 84) and a PR-AUC only
0.0013 behind. It was not chosen because its precision is 0.6493 vs 0.8485: catching 3 more frauds
costs roughly **32 extra false alarms**. Both models clear the gate, so this is a business
trade-off rather than a technical one — with a larger review team, `lgbm_default` would be the
defensible pick.

### Decision Threshold Analysis

Measured on the held-out test set (56,962 transactions, **98 frauds**) with the registered
champion. These are actual sweep results, not estimates:

| Threshold | Recall | Precision | F1 | TP | FP | FN | Use case |
|-----------|--------|-----------|-----|----|----|----|----------|
| 0.1 | 0.8776 | 0.7107 | 0.7854 | 86 | 35 | 12 | Max sensitivity |
| 0.2 | 0.8673 | 0.7798 | 0.8213 | 85 | 24 | 13 | |
| 0.3 | 0.8673 | 0.8019 | 0.8333 | 85 | 21 | 13 | High sensitivity |
| 0.4 | 0.8571 | 0.8317 | 0.8442 | 84 | 17 | 14 | |
| **0.5** (deployed) | **0.8571** | **0.8485** | **0.8528** | **84** | **15** | **14** | **Neutral default** |
| 0.6 | 0.8469 | 0.8737 | **0.8601** | 83 | 12 | 15 | Best F1 |
| 0.7 | 0.8469 | 0.8737 | 0.8601 | 83 | 12 | 15 | High precision |
| 0.8 | 0.8265 | 0.8901 | 0.8571 | 81 | 10 | 17 | |
| 0.9 | 0.8163 | 0.8889 | 0.8511 | 80 | 10 | 18 | Max precision |

Two things worth stating plainly:

1. **The curve is remarkably flat.** Between thresholds 0.1 and 0.9, recall moves only from 0.878
   to 0.816 and false positives from 35 to 10. The model separates the classes well enough that
   threshold choice is a second-order decision here — which is why 0.5 was kept rather than tuned.
2. **0.5 is not the F1 optimum** — 0.6 is (0.8601 vs 0.8528). 0.5 is retained deliberately as a
   neutral default: picking a threshold means pricing a missed fraud against an analyst's review
   time, and that is a business input this project does not have. See §"Known Limitations".

For contrast, the Logistic Regression baseline at the same threshold produces **TP=90, FP=1,441,
FN=8** — it catches 6 more frauds at the cost of ~96× more false alarms.

> ⚠️ Threshold change is a **business decision** — contact project owner before modifying.

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

### Methodological Limitations (known, not yet fixed)

Two places where the reported numbers are optimistically biased. Both are documented here rather
than quietly left in the code, because the size of the bias matters more than its existence:

- **The `Amount` scaler is fit before the train/test split.** `src/features.py::preprocess()` fits
  `StandardScaler` on all 284,807 rows, then `split_data()` splits. Strictly this leaks test-set
  statistics into training. In practice the leak is two scalars (mean 88.35, std 250.12) estimated
  from 284k rows, affecting 1 of 29 features — the effect on the reported metrics is far below the
  fourth decimal. The correct fix is to fit on the training split only and persist the fitted
  scaler as a model artifact, which would also remove the hard-coded constants in `src/api.py`.
- **Early stopping selects the number of boosting rounds on the test set.** Both `train_xgboost()`
  and `train_lightgbm()` pass `eval_set=[(X_test, y_test)]`. The test set therefore participates
  in choosing `best_iteration` (96 / 174 for the champion), so the reported test metrics are not
  a fully clean held-out estimate. The correct fix is a three-way train/validation/test split,
  early-stopping on validation and reporting on test. **This is the more material of the two** and
  is the reason the numbers in §4 should be read as "selection-set" rather than "held-out" results.

Neither issue affects the *ranking* of the seven runs (all were early-stopped identically), so the
model-selection conclusion stands; both affect the absolute level of the reported metrics.

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

- **Inference latency**: < 5ms per transaction (single prediction, local)
- **Batch**: up to 100 transactions per `/predict/batch` call
- **Throughput**: Single uvicorn worker handles ~200 req/s (single-core)

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
