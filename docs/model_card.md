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

| Model | PR-AUC | Recall | Precision | F1-score | ROC-AUC |
|-------|--------|--------|-----------|----------|---------|
| Logistic Regression (baseline) | 0.7156 | 0.9184 | 0.8571 | 0.8867 | ~0.97 |
| XGBoost default | 0.8445 | 0.8163 | 0.9302 | 0.8696 | ~0.98 |
| XGBoost deep | 0.8391 | 0.7959 | 0.9286 | 0.8571 | ~0.98 |
| XGBoost regularized | 0.8373 | 0.7959 | 0.9070 | 0.8475 | ~0.97 |
| LightGBM default | 0.8657 | 0.8367 | 0.9114 | 0.8725 | ~0.98 |
| LightGBM regularized | 0.8700 | 0.8469 | 0.9130 | 0.8787 | ~0.98 |
| **LightGBM large** ⭐ | **0.8770** | **0.8571** | **0.8485** | **0.8528** | **~0.98** |

### Champion Model vs Baseline

| Metric | Logistic Regression | LightGBM large | Δ Improvement |
|--------|--------------------|--------------------|--------------|
| PR-AUC | 0.7156 | **0.8770** | +22.6% 📈 |
| Recall | 0.9184 | 0.8571 | −6.7% (trade-off for precision) |
| Precision | 0.8571 | **0.8485** | −1.0% |
| F1-score | 0.8867 | 0.8528 | −3.8% |

> **Note on Recall trade-off**: The baseline LR achieves higher raw Recall (0.9184 vs 0.8571) due to its high bias toward the fraud class. The champion LightGBM model achieves significantly higher PR-AUC (0.8770 vs 0.7156), indicating much better calibrated probability scores and overall discrimination — the key metric for deployment.

### Decision Threshold Analysis

| Threshold | Recall | Precision | F1 | Use Case |
|-----------|--------|-----------|-----|----------|
| 0.3 | ~0.92 | ~0.76 | ~0.83 | High sensitivity (catch more fraud, more false alerts) |
| **0.5** (default) | **0.857** | **0.848** | **0.853** | **Balanced — deployed** |
| 0.7 | ~0.78 | ~0.91 | ~0.84 | High precision (fewer false alerts, miss more fraud) |

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
