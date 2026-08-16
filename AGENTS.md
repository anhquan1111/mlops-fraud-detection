# AGENTS.md — MLOps Fraud Detection Pipeline

Hướng dẫn cho AI CLI (Gemini, Copilot, Cursor, v.v.) khi làm việc với repo này.

---

## 1. Project Overview

**Bài toán**: Phát hiện giao dịch thẻ tín dụng gian lận (Credit Card Fraud Detection).

- **Input**: 1 giao dịch (vector 30 features: V1–V28 từ PCA, Amount, Time)
- **Output**: Xác suất gian lận (0–1) + nhãn dự đoán (fraud / not fraud)
- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ~284,807 giao dịch, ~0.17% là gian lận (mất cân bằng cực mạnh)

### Tiêu chí thành công

| Metric | Ngưỡng tối thiểu |
|--------|-------------------|
| Recall | ≥ 0.80 |
| Precision | ≥ 0.50 |
| PR-AUC | Cao hơn baseline (Logistic Regression) |

> ⚠️ **KHÔNG dùng accuracy** làm thước đo chính. Với dataset 99.83% negative, predict "not fraud" cho mọi giao dịch đã đạt 99.83% accuracy mà hoàn toàn vô dụng.

---

## 2. Tech Stack

| Tool | Version | Mục đích |
|------|---------|----------|
| Python | 3.12 | Runtime |
| uv | latest | Package manager |
| scikit-learn | ≥1.5.0 | Baseline model, preprocessing, metrics |
| XGBoost | ≥2.1.0 | Model chính (gradient boosting) |
| LightGBM | ≥4.5.0 | Model chính (alternative) |
| MLflow | ≥2.17.0 | Experiment tracking, model registry |
| FastAPI | ≥0.115.0 | API serving |
| uvicorn | ≥0.32.0 | ASGI server |
| pandas | ≥2.2.0 | Data manipulation |
| numpy | ≥2.0.0 | Numerical operations |
| Pydantic | ≥2.9.0 | Request/response validation |
| pytest | ≥8.3.0 | Testing |
| ruff | ≥0.7.0 | Linting & formatting |
| Docker | latest | Containerization |

---

## 3. Common Commands

```bash
# Môi trường
uv sync                          # Cài dependencies
uv sync --extra dev              # Cài thêm dev dependencies
uv sync --extra notebook         # Cài thêm notebook dependencies

# Training
uv run python src/train.py       # Chạy training pipeline

# MLflow
uv run mlflow ui                 # Mở MLflow dashboard (http://localhost:5000)

# Testing & Linting
uv run pytest                    # Chạy tests
uv run ruff check src/ tests/    # Lint check
uv run ruff format src/ tests/   # Auto-format

# Docker
docker build -t fraud-detection .           # Build image
docker run -p 8000:8000 fraud-detection     # Run container

# API
uv run uvicorn src.api:app --reload         # Chạy API dev server
```

---

## 4. Permission Levels (3 mức quyền)

### ✅ Luôn được tự làm (không cần hỏi)

- Chạy training script, thêm/sửa test
- Sửa feature engineering code
- Refactor code, cải thiện type hints, docstrings
- Thêm logging, error handling
- Chạy linting, formatting
- Cập nhật documentation

### ⚠️ Phải hỏi trước khi làm

- **Đổi decision threshold** (ví dụ 0.5 → 0.3) — đây là quyết định nghiệp vụ, ảnh hưởng trade-off precision/recall
- **Đổi metric dùng để chọn "model tốt nhất"** (ví dụ từ PR-AUC sang F1) — ảnh hưởng toàn bộ chiến lược đánh giá
- **Thay đổi kiến trúc model** (ví dụ thêm neural network, đổi sang ensemble method mới)
- **Thay đổi chiến lược xử lý imbalance** (ví dụ từ class_weight sang SMOTE)
- **Promote model lên "production"** trong MLflow Registry — phải chạy full eval trên test set trước

### 🚫 Không bao giờ được làm

- **Commit file model** (`.pkl`, `.joblib`, `.h5`, `.onnx`) vào git — dùng MLflow artifact store hoặc `.gitignore`
- **Commit data thô** (`.csv`, `.parquet`) vào git — data tải về local, nằm trong `.gitignore`
- **Tự ý promote model** lên production mà chưa chạy evaluation đầy đủ trên test set
- **Tối ưu theo accuracy** — metric này vô nghĩa với bài toán imbalanced
- **Áp dụng SMOTE trước khi split** train/test — gây data leakage

---

## 5. Coding Conventions

- **Commit**: [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`
- **Formatting**: `ruff format` (line length 100)
- **Linting**: `ruff check` với rules `E`, `F`, `I`, `W`, `UP`
- **Type hints**: Bắt buộc cho function signatures
- **Docstrings**: Google style
- **File structure**:
  ```
  mlops-fraud-detection/
  ├── src/                  # Source code chính
  │   ├── __init__.py
  │   ├── train.py          # Training pipeline
  │   ├── evaluate.py       # Evaluation logic
  │   ├── features.py       # Feature engineering
  │   ├── api.py            # FastAPI app
  │   └── config.py         # Configurations
  ├── tests/                # Test files
  ├── notebooks/            # EDA notebooks
  ├── data/                 # Data (gitignored)
  ├── docs/                 # Documentation
  ├── .github/workflows/    # CI/CD
  ├── AGENTS.md
  ├── README.md
  ├── Dockerfile
  └── pyproject.toml
  ```

---

## 6. Non-obvious Patterns & Gotchas

### Tại sao dùng `class_weight='balanced'` thay vì SMOTE?

`class_weight='balanced'` tự điều chỉnh trọng số loss function theo tỉ lệ lớp — **đơn giản, không tạo data giả, không rủi ro data leakage**. SMOTE tạo synthetic samples, dễ gây leakage nếu áp dụng trước train/test split (vì synthetic samples có thể "nhìn thấy" test data). Bắt đầu bằng `class_weight`, chỉ thử SMOTE nếu kết quả chưa đạt ngưỡng.

### Tại sao PR-AUC quan trọng hơn ROC-AUC?

ROC-AUC bị "lạc quan giả" trên imbalanced data vì True Negative Rate (TNR) rất cao khi negative class áp đảo. PR-AUC chỉ tập trung vào Precision và Recall của **lớp hiếm (fraud)** — phản ánh đúng khả năng thật của model trong việc phát hiện gian lận.

### Tại sao cần baseline Logistic Regression?

Không có baseline thì không chứng minh được model phức tạp (XGBoost) "tốt hơn" bao nhiêu. Baseline còn giúp phát hiện bug: nếu XGBoost tệ hơn Logistic Regression, gần như chắc chắn có lỗi trong pipeline (data leakage, feature engineering sai, v.v.).

### MLflow tracking — log đầy đủ mỗi run

Mỗi lần train **phải** gọi:
- `mlflow.log_param()` — hyperparameters, class_weight, threshold
- `mlflow.log_metric()` — precision, recall, f1, pr_auc, roc_auc
- `mlflow.sklearn.log_model()` hoặc `mlflow.xgboost.log_model()` — lưu model artifact
- Tag run với tên model, mục đích (baseline/experiment/production)

---

## 7. Session Management Plan (Token Optimization)

Project này được chia thành **6 sessions** để tối ưu token và giữ context gọn. Mỗi session là 1 milestone độc lập, có thể bắt đầu chat mới mà không mất context nhờ `AGENTS.md` + `docs/architecture.md` + git history.

### Bảng phân chia sessions

| Session | Trạng thái | Scope | Deliverables |
|---------|-----------|-------|--------------|
| 1 | ✅ DONE | Setup + AGENTS.md + Architecture Design | `.gitignore`, `pyproject.toml`, `AGENTS.md`, `docs/architecture.md` |
| 2 | ✅ DONE | Data + EDA + Baseline Model + MLflow | `src/config.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `notebooks/01_eda.py`, baseline LR tracked in MLflow (PR-AUC=0.7156, Recall=0.9184) |
| 3 | ✅ DONE | FastAPI Serving + Deploy khung | `src/api.py`, `Dockerfile`, `render.yaml`, `scripts/export_model.py`, `scripts/register_model.py`, 18 tests passed, model bundled via HF Hub strategy |
| 4 | ✅ DONE | XGBoost/LightGBM + Tuning + MLflow Registry | `src/config.py` (XGB/LGBM params), `src/train.py` (7-run experiment grid), `scripts/select_best_model.py`, 7 MLflow runs, best model registered: `fraud-detection-model@production` (xgb_default: PR-AUC=0.8707, Recall=0.8367, Precision=0.7664) |
| 5 | ⬜ TODO | Validation Gate + Testing + CI/CD | `src/validate.py`, `tests/`, `.github/workflows/`, pytest + ruff + pipeline smoke test |
| 6 | ⬜ TODO | Final Deploy + README + Model Card | `README.md`, `docs/model_card.md`, deploy bản cuối, polish |

### Bootstrap Prompt cho mỗi session mới

Khi bắt đầu session mới, user sẽ paste prompt tương ứng:

**Session 2:**
```
Tôi tiếp tục project MLOps Fraud Detection. Đọc AGENTS.md và docs/architecture.md để nắm context.
Session 2: Setup môi trường uv, tải dataset Kaggle Credit Card Fraud, EDA nhanh, build baseline Logistic Regression + log MLflow.
Kiểm tra codebase hiện tại rồi bắt đầu.
```

**Session 3:**
```
Tôi tiếp tục project MLOps Fraud Detection. Đọc AGENTS.md và docs/architecture.md.
Session 3: Wrap model baseline vào FastAPI (src/api.py), tạo Dockerfile, deploy lên Hugging Face Spaces.
Kiểm tra codebase hiện tại rồi bắt đầu.
```

**Session 4:**
```
Tôi tiếp tục project MLOps Fraud Detection. Đọc AGENTS.md và docs/architecture.md.
Session 4: Build XGBoost/LightGBM model, xử lý imbalance, chạy nhiều runs MLflow, chọn best model → MLflow Registry.
Kiểm tra codebase hiện tại rồi bắt đầu.
```

**Session 5:**
```
Tôi tiếp tục project MLOps Fraud Detection. Đọc AGENTS.md và docs/architecture.md.
Session 5: Viết validation gate (so sánh model mới vs production), pytest cho features + API, CI/CD GitHub Actions.
Kiểm tra codebase hiện tại rồi bắt đầu.
```

**Session 6:**
```
Tôi tiếp tục project MLOps Fraud Detection. Đọc AGENTS.md và docs/architecture.md.
Session 6: Deploy bản cuối, viết README.md (bảng so sánh models, sơ đồ pipeline), viết model card, polish.
Kiểm tra codebase hiện tại rồi bắt đầu.
```

### ⚡ Quy tắc kết thúc session (BẮT BUỘC cho AI)

Khi hoàn thành **tất cả deliverables** của session hiện tại, AI **PHẢI**:

1. **Commit** tất cả thay đổi (Conventional Commits)
2. **Cập nhật bảng sessions** ở trên: đổi trạng thái session vừa xong từ `⬜ TODO` sang `✅ DONE`
3. **Thông báo cho user** bằng message có format sau:

```
✅ Session [N] hoàn thành!

Deliverables đã xong:
- [liệt kê]

👉 Bạn nên tạo NEW SESSION và paste prompt sau:
[copy bootstrap prompt của session tiếp theo]
```

> ⚠️ AI không được tự tiếp tục sang session tiếp theo mà không thông báo. User quyết định khi nào bắt đầu session mới.
