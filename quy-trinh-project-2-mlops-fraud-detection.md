# Quy trình chuẩn để làm Project 2 — MLOps Pipeline (Fraud Detection)

**Hạn gốc trong roadmap: 26/08/2026** (roadmap phân bổ 22–26/08, ~5 ngày, ngay sau khi xong Project 1 21/08). Nếu bạn bắt đầu trễ hơn mốc này thì cứ dời cả bảng lịch cuối bài, giữ nguyên tỉ lệ số ngày mỗi bước.

Cấu trúc quy trình giống hệt tinh thần Project 1 (định hình trước → khung chạy tối thiểu → deploy sớm → thêm độ phức tạp dần → test/CI song song → polish cuối), chỉ khác nội dung kỹ thuật vì đây là MLOps (train model thật) chứ không phải LLMOps (điều phối LLM có sẵn).

---

## Bước 0 — Setup & Planning (nửa ngày đầu)

1. **Spec 1 trang**: bài toán (phát hiện giao dịch gian lận), input/output (1 giao dịch → xác suất gian lận), tiêu chí "xong" — quan trọng: không chỉ là "chạy được" mà là **đạt ngưỡng recall/precision cụ thể** (ví dụ recall ≥ 0.8 ở precision ≥ 0.5), vì đây là bài toán mất cân bằng, "chạy được" mà không đo đúng chỉ số thì vô nghĩa.
2. **Chọn dataset**: khuyên dùng **Kaggle Credit Card Fraud Detection** — dữ liệu thật, nổi tiếng, ~0.17% giao dịch là gian lận (mất cân bằng cực mạnh, đúng chất bài toán fraud thật). Dùng dataset có sẵn để không tốn thời gian tìm/làm sạch dữ liệu, dồn hết thời gian cho phần pipeline — đúng tinh thần "infra/dataset vừa đủ, đầu tư vào phần lõi".
3. **Repo structure**: tương tự Project 1 (`src/`, `tests/`, `docs/`, `.github/workflows/`, `Dockerfile`, `AGENTS.md`, `README.md`), thêm `notebooks/` cho phần EDA và `mlruns/` (MLflow tự tạo, nhớ `.gitignore`).
4. **Môi trường + Git**: giữ nguyên thói quen từ Project 1 (uv/poetry, Conventional Commits, commit nhỏ sau mỗi bước chạy được).

---

## Bước 1 — Cập nhật `AGENTS.md` cho repo MLOps

Tạo file mới cho repo này (không dùng chung với Project 1), nhưng cấu trúc 6 phần đã học ở Project 1 giữ nguyên. Phần **khác biệt** cần thêm:

- **Tech stack riêng**: scikit-learn, XGBoost/LightGBM, MLflow, FastAPI, pandas — ghi rõ version.
- **Lệnh hay dùng**: `mlflow ui` (xem dashboard so sánh run), lệnh chạy training script, `pytest`, `docker build`.
- **3 mức quyền, chỉnh lại cho ML**:
  - ✅ Luôn được tự làm: chạy training, thêm test, sửa feature engineering
  - ⚠️ Phải hỏi trước: đổi threshold quyết định (0.5 → 0.3 chẳng hạn), đổi metric dùng để chọn model tốt nhất — đây là **quyết định nghiệp vụ**, không phải kỹ thuật thuần, AI không được tự quyết
  - 🚫 Không bao giờ: commit file model lớn (`.pkl`, `.joblib`) hoặc data thô vào git — dùng MLflow artifact store hoặc `.gitignore`
- **Pattern phi trực giác cần ghi rõ**: vì sao dùng `class_weight` thay vì accuracy làm metric chính (xem Bước 2) — nếu không ghi, AI CLI có thể "tối ưu nhầm" theo accuracy và báo cáo model "tốt" trong khi thực ra vô dụng.

---

## Bước 2 — Thiết kế trước khi code

Viết `docs/architecture.md` ngắn, trả lời trước 3 câu hỏi hay bị bỏ qua khi mới làm bài toán mất cân bằng:

1. **Metric nào là đúng?** Tuyệt đối không dùng accuracy làm thước đo chính — vì chỉ cần đoán "không gian lận" cho mọi giao dịch đã đạt ~99.8% accuracy mà vô dụng hoàn toàn. Dùng **Precision, Recall, F1, và đặc biệt PR-AUC** (quan trọng hơn ROC-AUC khi lớp hiếm) làm thước đo chính, ghi rõ trong AGENTS.md và README.
2. **Xử lý mất cân bằng bằng cách nào?** Bắt đầu bằng `class_weight='balanced'` (có sẵn trong XGBoost/LightGBM/sklearn, đơn giản, rủi ro thấp) trước khi thử resampling (SMOTE) — SMOTE dễ gây data leakage nếu áp dụng sai thứ tự (phải resample sau khi split train/test, không phải trước).
3. **Model nào?** Luôn có 1 **baseline đơn giản trước** (Logistic Regression) để có cột mốc so sánh, rồi mới build model chính (XGBoost/LightGBM — thường thắng áp đảo trên dữ liệu dạng bảng). Không có baseline thì không chứng minh được model chính "tốt hơn" bao nhiêu.

---

## Bước 3 — Build theo "walking skeleton"

1. **Ngày 1** — Setup repo + AGENTS.md + tải dataset + EDA nhanh (phân bố lớp, missing value) + baseline Logistic Regression chạy được, **log vào MLflow ngay** dù model còn thô. Mục tiêu: có 1 vòng train → track chạy trọn vẹn sớm nhất, chưa cần tối ưu.
2. **Ngày 2** — Bọc model baseline vào FastAPI, **deploy thử ngay** (dù model chưa tốt) lên cùng nền tảng đã quen ở Project 1 (Hugging Face Spaces, Docker SDK) — để có link sống sớm, không dồn deploy về cuối như quy trình Project 1 đã cảnh báo.
3. **Ngày 3** — Build model thật: XGBoost/LightGBM + xử lý mất cân bằng + feature engineering. Chạy nhiều lần, so sánh trong MLflow UI, chọn model tốt nhất theo PR-AUC/F1, **đăng ký vào MLflow Model Registry** (đánh dấu alias "production").
4. **Ngày 4** — Test + CI + Dockerize (chi tiết Bước 5-6).
5. **Ngày 5** — Deploy bản model tốt nhất, viết README + model card, polish.

---

## Bước 4 — Validation gate (điểm khác biệt cốt lõi của MLOps so với "chỉ train 1 lần")

Best practice 2026 của MLflow nhấn mạnh: version hoá mọi thứ (code, data, hyperparameter), và có **cổng kiểm định tự động** trước khi 1 model được lên "production" — đây chính là điều phân biệt 1 pipeline MLOps thật với 1 notebook train-rồi-thôi:

- Viết 1 script nhỏ: trước khi promote model mới, so sánh recall/F1/PR-AUC của nó với model đang là "production" trong MLflow Registry — chỉ promote nếu **không tệ hơn** model cũ. Đơn giản nhưng đủ để chứng minh bạn hiểu khái niệm validation gate.
- **Input validation ở API**: dùng Pydantic (đã quen từ Project 1) để validate schema request, từ chối request thiếu field/sai kiểu trước khi đưa vào model.
- MLflow tự động version hoá phần lớn (params, metrics, artifact) nếu bạn log đủ — chỉ cần nhớ gọi `mlflow.log_param`/`log_metric`/`log_model` đầy đủ mỗi lần train.

---

## Bước 5 — Testing

- `pytest` cho hàm feature engineering: input cố định → output cố định (deterministic, không cần mock gì phức tạp vì đây không phải LLM).
- Test API endpoint: gửi 1 giao dịch mẫu, kiểm tra response đúng schema (có field probability/label).
- Nếu còn thời gian: 1 test "chất lượng model" — assert F1/recall trên tập test cố định phải vượt 1 ngưỡng tối thiểu, để CI tự chặn nếu model bị làm tệ đi do sửa code nhầm.

---

## Bước 6 — CI/CD

- GitHub Actions: `ruff check` + `pytest` mỗi lần push (giống Project 1).
- **Điểm cộng riêng cho MLOps**: thêm 1 job chạy thử training script trên tập dữ liệu mẫu nhỏ (vài trăm dòng) để đảm bảo pipeline train không bị gãy khi code thay đổi — CI ở đây kiểm cả pipeline dữ liệu/model, không chỉ code như Project 1.

---

## Bước 7 — Docker & Deploy

- Dockerfile đóng gói FastAPI app, model load từ MLflow Model Registry hoặc file `.pkl` đóng kèm image.
- Deploy cùng nền tảng Project 1 (Hugging Face Spaces, Docker SDK) — tận dụng lại kinh nghiệm vừa làm, không mất công học nền tảng mới.
- MLflow tracking server không cần deploy public — chạy local lúc train, chụp lại screenshot dashboard (so sánh các run) để đưa vào README/model card, đủ để chứng minh bạn có track thật.

---

## Bước 8 — Tài liệu & Model Card

README nên có: mô tả bài toán, sơ đồ pipeline (ảnh đơn giản), **bảng so sánh các model đã thử** (Logistic Regression baseline vs XGBoost) kèm số liệu Precision/Recall/F1/PR-AUC, link demo, hướng dẫn chạy local.

Thêm 1 **model card ngắn** (chuẩn ngày càng phổ biến trong ML production, tách riêng khỏi README): model dùng để làm gì, giới hạn (dataset là dữ liệu công khai, không đại diện hết cho pattern gian lận thật ngoài đời — model không nên dùng thật để quyết định chặn giao dịch nếu chưa kiểm định kỹ hơn), input/output kỳ vọng. Mục này thể hiện tư duy trách nhiệm khi triển khai ML — điểm hay được hỏi khi phỏng vấn MLOps/Platform.

---

## Bước 9 — Làm việc với AI CLI (áp dụng riêng cho MLOps)

Thêm vào AGENTS.md của repo này 2 quy tắc riêng: **AI không được tự đổi threshold quyết định hay đổi metric dùng để chọn "model tốt nhất" mà không hỏi trước** (quyết định nghiệp vụ), và **luôn chạy lại full eval trên tập test trước khi promote 1 model lên "production" trong MLflow Registry** — đừng để AI tự tin báo "xong" chỉ dựa trên train loss thấp.

---

## Tóm tắt lịch 5 ngày

| Ngày | Việc chính |
|---|---|
| 1 | Repo + AGENTS.md + dataset + EDA + baseline model + track MLflow |
| 2 | FastAPI wrap model baseline + deploy khung sớm (link sống) |
| 3 | Model thật (XGBoost + xử lý imbalance) + nhiều run MLflow + chọn & đăng ký model tốt nhất |
| 4 | Test + CI (kèm test pipeline train) + Dockerize |
| 5 | Deploy bản cuối + README + model card + polish |

---

### Nguồn tham khảo
- [MLOps Pipeline Automation Best Practices in 2026 — MLflow official](https://mlflow.org/articles/mlops-pipeline-automation-best-practices-in-2026/)
- [MLOPS-Full-Data-Pipeline — ví dụ pipeline fraud detection đầy đủ (DVC, MLflow, CI/CD, Docker)](https://github.com/pacificrm/MLOPS-Full-Data-Pipeline)
