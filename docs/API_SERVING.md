# ⚡ Kiến thức cốt lõi: FastAPI Serving & Infrastructure Deployment (Session 3)

> [!info] Mục đích tài liệu
> Tài liệu này tổng hợp toàn bộ tư duy thiết kế hệ thống Phục vụ dự đoán (Model Serving), Đóng gói Container (Docker), Lưu trữ Model Cloud (Hugging Face Hub) và Cơ chế hoạt động của Render Platform. Được tối ưu theo chuẩn Obsidian Markdown.

---

## 1. 🌐 Tổng quan Kiến trúc API & Hạ tầng Deployment

Trong dự án Machine Learning thực tế, ứng dụng được tách thành **3 tầng độc lập**:

```
 ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
 │   Hugging Face Hub     │      │     Render Platform    │      │    Client Application  │
 │  (Model Storage Repo)  │      │  (Docker Container)    │      │    (Web / App / Bank)  │
 ├────────────────────────┤      ├────────────────────────┤      ├────────────────────────┤
 │ Lưu file:              │ ────►│ Nạp model lúc boot ➔   │ ◄─── │ Gửi JSON giao dịch ➔   │
 │ `baseline_lr.pkl`      │Download│ Chạy FastAPI uvicorn  │Request│ Nhận kết quả Fraud %  │
 └────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 2. ❓ Giải mã các thắc mắc cốt lõi về Hạ tầng MLOps

### 🏢 2.1 Tại sao phải đẩy Model lên Hugging Face Hub (Model Repository)?

> [!question] Tại sao không commit luôn file `.pkl` vào Git Repo chính để Render tự kéo về?
> 
> 1. **Quy tắc Vàng của Git**: Git chỉ được dùng để quản lý **Code dạng văn bản** (chữ/số). Git cực kỳ dở khi quản lý các file binary nặng như Model trọng số (`.pkl`, `.pt`, `.h5`). Commit model vào Git sẽ làm repo phình to hàng GB và gây nghẽn hệ thống.
> 2. **Phân tách trách nhiệm (Separation of Concerns)**: 
>    - **GitHub**: Chứa Source Code ứng dụng (`src/api.py`, `Dockerfile`).
>    - **Hugging Face Hub**: Đóng vai trò là **Kho lưu trữ Model tĩnh (Model Storage)** miễn phí.
> 3. **Cơ chế hoạt động của HF Hub**: HF Hub quản lý các file `.pkl` bằng cờ Git Commits. Mỗi khi ta train được model mới ngon hơn, ta chỉ cần upload đè file `baseline_lr.pkl` lên HF Hub. Đường link tải không bao giờ đổi!

---

### 🔄 2.2 Render hoạt động như thế nào? Cách nó tự động cập nhật khi Push Git

> [!abstract] Cơ chế GitHub Webhook
> Render không tự đoán khi nào bạn đổi code. Nó sử dụng một con robot theo dõi gọi là **Webhook**:

```
BẠN (gõ: git push)
   │
   ▼
GitHub Repository ────(WebHook Signal)────► Render Server
                                                 │
                                                 ├── 1. Kéo Code mới từ GitHub
                                                 ├── 2. Build Docker Container
                                                 └── 3. Restart Web App (Zero Downtime)
```

1. **Khi bạn `git push` lên GitHub**: GitHub lập tức gửi một tín hiệu (Webhook) báo cho Render: *"Repo này vừa có commit mới!"*.
2. **Render thực thi**: Render tải code mới nhất ➔ Đọc `Dockerfile` để đóng gói lại Image ➔ Khởi chạy Container mới ➔ Tắt Container cũ.

---

### 🔀 2.3 Cơ chế Nhận diện Model 3 Cấp (3-Priority Strategy)

Trong `src/api.py`, hàm `load_model()` quyết định môi trường chạy thông qua **Biến Môi Trường (`os.environ`)**:

```python
def load_model() -> tuple[Any, dict]:
    # 🥇 Ưu tiên 1: Đọc file .pkl local (Dành cho Docker Container hoặc Test máy nhà)
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return _load_from_local(model_path)

    # 🥈 Ưu tiên 2: Tải từ Hugging Face Hub (Dành cho Render Service khi không dùng Docker)
    hf_repo_id = os.environ.get("HF_REPO_ID")
    if hf_repo_id:
        return _load_from_hf_hub(hf_repo_id)

    # 🥉 Ưu tiên 3 (Fallback): Nạp từ MLflow Registry Local (Dành cho Data Scientist Dev)
    return _load_from_mlflow()
```

> [!success] Tại sao thiết kế này đỉnh cao?
> - **Chạy trong Docker (Render)**: Dockerfile cài sẵn `ENV MODEL_PATH=models/baseline_lr.pkl` ➔ Khớp ngay **Ưu tiên 1**.
> - **Chạy trên Render thuần (no Docker)**: Bạn cài trên Dashboard `HF_REPO_ID` ➔ Khớp ngay **Ưu tiên 2**.
> - **Chạy gõ lệnh test nhanh ở local**: Không cài biến gì ➔ Tự nhảy xuống **Ưu tiên 3** kết nối `mlflow.db`.
> ➔ **Chỉ 1 file code Python nhưng chạy hoàn hảo trên 3 môi trường khác nhau!**

---

## 3. 🧩 Mổ xẻ Chi tiết Code `src/api.py`

### ⚡ 3.1 Quản lý Vòng đời Server (`@asynccontextmanager lifespan`)

> [!danger] Sai lầm nghiêm trọng của Lập trình viên mới
> Nạp model inside hàm route `/predict`. Việc này khiến **mỗi khi khách hàng gửi request, server lại tốn 2 giây nạp file từ ổ cứng vào RAM**, gây nghẽn hệ thống!

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_info
    # 🚀 LÚC KHỞI ĐỘNG: Nạp model vào RAM đúng 1 lần duy nhất!
    _model, _model_info = load_model()
    yield  # Server bật công tắc nhận các Request từ khách hàng...
    # 🔻 LÚC TẮT SERVER: Giải phóng bộ nhớ RAM
    _model = None
```
- **Lợi ích**: Giúp thời gian xử lý dự đoán `/predict` giảm xuống chỉ còn **~0.001 giây (1 millisecond)** vì model đã nằm sẵn trên RAM từ trước!

---

### 🛡️ 3.2 Tầng Bảo Vệ Dữ Liệu: Pydantic Schema (`TransactionInput`)

FastAPI sử dụng Pydantic v2 để làm "người gác cổng" kiểm tra dữ liệu đầu vào:

```python
class TransactionInput(BaseModel):
    V1: float = Field(..., description="PCA component 1")
    ...
    Amount: float = Field(..., ge=0.0, description="Số tiền giao dịch (USD) >= 0")
```

- **`ge=0.0` (Greater than or equal to 0)**: Nếu kẻ xấu gửi số tiền âm (`Amount: -100`), Pydantic lập tức chặn lại và trả lỗi `422 Unprocessable Content` trước khi dữ liệu kịp đi vào model!
- **Tính đầy đủ**: Đảm bảo đủ 29 cột (`V1–V28` + `Amount`). Nếu thiếu bất kỳ cột nào, API sẽ chủ động báo lỗi rõ ràng.

---

### 🧮 3.3 Chuẩn hóa Dữ liệu Thực tế (`_preprocess_transaction`)

```python
_AMOUNT_MEAN = 88.3496
_AMOUNT_STD = 250.1201

def _preprocess_transaction(tx: TransactionInput) -> pd.DataFrame:
    features = []
    for col in FEATURE_COLS:
        val = getattr(tx, col)
        if col == "Amount":
            # Scale Amount theo Mean & Std tiêu chuẩn thu được từ EDA
            val = (val - _AMOUNT_MEAN) / _AMOUNT_STD
        features.append(val)
    return pd.DataFrame([features], columns=FEATURE_COLS)
```

- **Tại sao phải scale `Amount`?**
  Trong Kaggle dataset, `V1–V28` đã được PCA chuẩn hóa về dạng phân phối Gaussian, nhưng `Amount` ban đầu là số tiền thô (ví dụ `$149.62`). Hàm này dùng công thức $Z = \frac{X - \mu}{\sigma}$ để đưa `Amount` về cùng thang đo với `V1–V28` trước khi gọi `model.predict_proba()`.

---

### 🔌 3.4 Bảng Tổng hợp Endpoints

| Endpoint | Method | Vai trò trong MLOps |
| :--- | :--- | :--- |
| `GET /` | `GET` | **API Info**: Trả về thông tin phiên bản, threshold và nguồn model đang nạp. |
| `GET /health` | `GET` | **Liveness Probe**: Render dùng endpoint này mỗi 30s để kiểm tra API có bị treo không (`status: healthy`). |
| `POST /predict` | `POST` | **Single Inference**: Dự đoán 1 giao dịch đơn lẻ cho ứng dụng real-time. |
| `POST /predict/batch` | `POST` | **Batch Inference**: Dự đoán danh sách tối đa 100 giao dịch cùng lúc cho hệ thống xử lý lô. |

---

## 4. 🐳 Đóng gói Docker Container (`Dockerfile`)

```dockerfile
FROM python:3.12-slim

# Cài package manager uv (nhanh hơn pip 10-100 lần)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY models/baseline_lr.pkl ./models/baseline_lr.pkl

ENV MODEL_PATH=models/baseline_lr.pkl

EXPOSE 10000
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "10000"]
```

> [!tip] Điểm sáng trong Dockerfile
> 1. **Base Image `python:3.12-slim`**: Giúp container cực nhẹ (~150MB).
> 2. **Caching Layer**: Copy `pyproject.toml` và chạy `uv sync` trước khi copy code `src/`. Nhờ vậy những lần sau đổi code, Docker không cần phải cài lại thư viện!
> 3. **Bundled Model**: Copy file `baseline_lr.pkl` trực tiếp vào image giúp API tự đủ khả năng chạy độc lập mà không cần mạng internet để tải model.
