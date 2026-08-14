# Architecture Design — MLOps Fraud Detection Pipeline

Tài liệu thiết kế trước khi code. Trả lời 3 câu hỏi cốt lõi cho bài toán phát hiện gian lận trên dữ liệu mất cân bằng cực mạnh (~0.17% positive).

---

## 1. Metric nào là đúng?

### Vấn đề với Accuracy

Dataset có ~99.83% giao dịch hợp lệ và ~0.17% giao dịch gian lận. Một model "ngu" chỉ cần predict **tất cả là "not fraud"** sẽ đạt **99.83% accuracy** — nhưng **bỏ sót 100% giao dịch gian lận**, hoàn toàn vô dụng.

→ **Accuracy bị CẤM dùng làm metric chính** trong project này.

### Metrics được chọn

| Metric | Vai trò | Giải thích |
|--------|---------|------------|
| **PR-AUC** | Primary | Area Under Precision-Recall Curve — phản ánh khả năng phát hiện lớp hiếm (fraud) tốt hơn ROC-AUC trên imbalanced data |
| **Recall** | Target ≥ 0.80 | Tỉ lệ giao dịch gian lận **thật** được phát hiện — ưu tiên cao vì bỏ sót fraud gây thiệt hại tài chính |
| **Precision** | Target ≥ 0.50 | Tỉ lệ dự đoán "fraud" đúng — quá thấp sẽ gây quá nhiều false alarm, làm phiền khách hàng |
| **F1-score** | Secondary | Harmonic mean của Precision và Recall — metric tổng hợp để so sánh nhanh |
| **ROC-AUC** | Reference only | Ghi nhận để tham khảo, KHÔNG dùng để chọn model tốt nhất |

### Tại sao PR-AUC > ROC-AUC?

ROC-AUC sử dụng True Negative Rate (Specificity), mà với 99.83% negative, TNR gần như luôn rất cao bất kể model tốt hay dở → ROC-AUC bị "lạc quan giả". PR-AUC chỉ dùng Precision và Recall — cả hai đều tập trung vào **lớp positive (fraud)**, phản ánh đúng năng lực model.

### Ngưỡng quyết định (Decision Threshold)

- Mặc định: **0.5**
- Có thể điều chỉnh để tối ưu trade-off Precision/Recall, nhưng **phải hỏi trước** (quyết định nghiệp vụ, không phải kỹ thuật thuần)
- Threshold thấp hơn → Recall tăng, Precision giảm (bắt nhiều fraud hơn nhưng nhiều false alarm hơn)
- Threshold cao hơn → Precision tăng, Recall giảm (ít false alarm nhưng bỏ sót fraud nhiều hơn)

---

## 2. Xử lý mất cân bằng (Imbalance Handling)

### Giai đoạn 1: `class_weight='balanced'` (ưu tiên)

**Cách hoạt động**: Tự động điều chỉnh trọng số trong loss function — lớp hiếm (fraud) được gán trọng số cao hơn, buộc model "quan tâm" đến fraud hơn.

**Ưu điểm**:
- Có sẵn trong scikit-learn, XGBoost, LightGBM (tham số `scale_pos_weight`)
- Không tạo data giả → không rủi ro data leakage
- Đơn giản, dễ hiểu, dễ reproduce

**Áp dụng**:
```python
# scikit-learn
LogisticRegression(class_weight='balanced')

# XGBoost — scale_pos_weight = n_negative / n_positive
XGBClassifier(scale_pos_weight=ratio_negative_to_positive)
```

### Giai đoạn 2: SMOTE (chỉ khi cần)

Chỉ thử nếu `class_weight` chưa đạt ngưỡng target. **Quy tắc bắt buộc**:

```
⚠️ SMOTE phải áp dụng SAU khi split train/test
   KHÔNG BAO GIỜ áp dụng trước split — gây data leakage nghiêm trọng

   ❌ Sai:  SMOTE(data) → train_test_split → train
   ✅ Đúng: train_test_split → SMOTE(train_only) → train
```

---

## 3. Model Pipeline

### 3.1 Baseline: Logistic Regression

**Mục đích**: Tạo cột mốc so sánh (benchmark) — bắt buộc phải có trước khi build model phức tạp.

**Tại sao cần baseline?**
- Chứng minh model chính (XGBoost) thực sự **tốt hơn** chứ không chỉ "chạy được"
- Phát hiện bug: nếu XGBoost tệ hơn Logistic Regression → gần như chắc chắn pipeline có lỗi
- Nhanh, đơn giản, dễ debug

```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

### 3.2 Model chính: XGBoost / LightGBM

**Tại sao gradient boosting?**
- Thường thắng áp đảo trên dữ liệu dạng bảng (tabular data)
- Xử lý tốt feature interaction mà không cần feature engineering thủ công
- Hỗ trợ native `scale_pos_weight` cho imbalanced data
- Nhanh, có early stopping để tránh overfitting

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=ratio,   # xử lý imbalance
    eval_metric='aucpr',      # PR-AUC làm metric optimize
    early_stopping_rounds=10,
    random_state=42
)
```

### 3.3 Tracking: MLflow

Mọi run đều phải log đầy đủ:

```
mlflow.log_param()    → hyperparameters, class_weight, threshold
mlflow.log_metric()   → precision, recall, f1, pr_auc, roc_auc
mlflow.log_model()    → model artifact
mlflow.set_tag()      → model_type (baseline/xgboost/lightgbm), purpose (experiment/production)
```

Model tốt nhất (theo PR-AUC) được đăng ký vào **MLflow Model Registry** với alias `production`.

---

## 4. Pipeline tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                           │
│                                                                    │
│  Data (CSV)                                                        │
│    ↓                                                               │
│  Load & Validate (pandas + Pydantic schema check)                  │
│    ↓                                                               │
│  Feature Engineering (src/features.py)                             │
│    ↓                                                               │
│  Train/Test Split (stratified, giữ tỉ lệ fraud)                   │
│    ↓                                                               │
│  Train Model (class_weight='balanced')                             │
│    ↓                                                               │
│  Evaluate (Precision, Recall, F1, PR-AUC)                         │
│    ↓                                                               │
│  Log to MLflow (params + metrics + model artifact)                 │
│    ↓                                                               │
│  Compare with current "production" model                           │
│    ↓                                                               │
│  Promote if better → MLflow Model Registry (alias: production)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        SERVING PIPELINE                            │
│                                                                    │
│  FastAPI (src/api.py)                                              │
│    ↓                                                               │
│  Pydantic input validation                                         │
│    ↓                                                               │
│  Load model from MLflow Registry (alias: production)               │
│    ↓                                                               │
│  Predict → { probability: float, is_fraud: bool }                  │
│    ↓                                                               │
│  Return JSON response                                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CI/CD PIPELINE                              │
│                                                                    │
│  Push to GitHub                                                    │
│    ↓                                                               │
│  GitHub Actions                                                    │
│    ├── ruff check (lint)                                           │
│    ├── pytest (unit + integration tests)                           │
│    └── Train on sample data (pipeline smoke test)                  │
│    ↓                                                               │
│  Docker build & deploy (Hugging Face Spaces)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow

```
Kaggle Dataset (creditcard.csv)
  │
  ├── 284,807 giao dịch
  ├── 30 features: Time, V1–V28 (PCA), Amount
  ├── 1 label: Class (0 = legit, 1 = fraud)
  └── ~0.17% fraud (492 / 284,807)
  │
  ↓ train_test_split (stratify=Class, test_size=0.2)
  │
  ├── Train set (~227,845 samples) → dùng để train
  └── Test set (~56,962 samples)   → dùng để eval, KHÔNG BAO GIỜ dùng để train
```

---

## 6. Key Design Decisions Log

| # | Quyết định | Lý do | Thay đổi? |
|---|-----------|-------|-----------|
| 1 | PR-AUC làm primary metric | ROC-AUC bị lạc quan giả trên imbalanced data | Phải hỏi trước |
| 2 | class_weight='balanced' trước SMOTE | Đơn giản hơn, không rủi ro data leakage | Phải hỏi trước |
| 3 | Logistic Regression làm baseline | Cần benchmark để chứng minh model chính tốt hơn | Không |
| 4 | XGBoost/LightGBM làm model chính | Thường thắng trên tabular data | Phải hỏi trước |
| 5 | Threshold mặc định 0.5 | Điểm bắt đầu hợp lý, tune sau | Phải hỏi trước |
| 6 | Stratified split | Giữ tỉ lệ fraud trong train/test | Không |
