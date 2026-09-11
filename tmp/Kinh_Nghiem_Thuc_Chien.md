# Kinh Nghiệm Thực Chiến

Tài liệu tổng hợp các kinh nghiệm, kỹ thuật chuẩn và lưu ý thực tế được đúc kết trực tiếp trong quá trình đọc và chuẩn hóa mã nguồn dự án.

---

## 1. Quản Lý Cấu Hình: File `.env` & Pydantic `BaseSettings`

### File `.env` mẫu
```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DECISION_THRESHOLD=0.85
FRAUD_REPORTS_DIR=reports
UNWANTED_SYSTEM_KEY=xyz123
```

### Khai báo trong `src/config.py`
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MLFLOW_TRACKING_URI: str = "sqlite:///default.db"
    DECISION_THRESHOLD: float = 0.5
```

### Cơ chế nạp biến vào `settings`

| Biến trong `.env` | Khai báo trong class | Kết quả (`settings`) |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` | `str` | `settings.MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"` |
| `DECISION_THRESHOLD=0.85` | `float` | `settings.DECISION_THRESHOLD = 0.85` (tự ép sang float) |
| *(Không ghi trong .env)* | `default="sqlite:///default.db"` | Lấy giá trị default khai báo sẵn |
| `UNWANTED_SYSTEM_KEY=xyz123` | *(Không khai báo)* | Bỏ qua (nhờ `extra="ignore"`) |

### Tiền xử lý & Fallback an toàn: `@field_validator`

```python
@field_validator("MLFLOW_TRACKING_URI", mode="before")
@classmethod
def _default_mlflow_uri(cls, v: Any) -> str:
    if v and str(v).strip():
        return str(v).strip()
    return f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
```

#### Giải thích các thành phần
- **`@field_validator("...", mode="before")`**: Chạy **TRƯỚC KHI** Pydantic kiểm tra kiểu dữ liệu, cho phép can thiệp trực tiếp vào giá trị thô đọc từ `.env`.
- **`@classmethod` & `cls`**: Hàm validator chạy ở cấp độ Class (khi object `settings` chưa tạo xong), `cls` chính là class `AppSettings`.
- **`v` (Value)**: Giá trị thô được truyền vào từ `.env` (chuỗi text hoặc rỗng `""`).

#### Tại sao cần hàm này dù ở trên đã có `default`?
- **Cái bẫy chuỗi rỗng**: Nếu trong `.env` người dùng vô tình viết `MLFLOW_TRACKING_URI=` (bỏ trống sau dấu bằng), Pydantic thấy có giá trị nên **sẽ KHÔNG dùng `default`**, dẫn đến biến bị rỗng `""` và làm sập ứng dụng.
- **Vai trò validator**: Kiểm tra nếu `v` bị rỗng hoặc chỉ toàn khoảng trắng thì chủ động ép về giá trị mặc định an toàn (`sqlite://...`).

### Khởi tạo trong `src/config.py`
```python
# Tạo instance dùng chung cho toàn bộ dự án
settings = AppSettings()

# Xuất biến ra ngoài để dùng ngắn gọn
DECISION_THRESHOLD = settings.DECISION_THRESHOLD
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
```

### Cách import và sử dụng ở file `.py` khác
```python
# Cách 1: Import trực tiếp biến (ngắn gọn, khuyên dùng)
from src.config import DECISION_THRESHOLD, MLFLOW_TRACKING_URI

if probability >= DECISION_THRESHOLD:
    print("Giao dịch gian lận")

# Cách 2: Import cả object settings
from src.config import settings

print(settings.MLFLOW_TRACKING_URI)
```

### Thứ tự ưu tiên (Priority Order)
1. **Biến môi trường hệ điều hành (OS Environment Variable)**: Ưu tiên cao nhất, dùng khi deploy Docker hoặc CI/CD.
2. **Giá trị ghi trong file `.env`**: Ưu tiên thứ hai, phục vụ phát triển local.
3. **Giá trị mặc định (`default`) trong class Python**: Ưu tiên cuối cùng để fallback an toàn.

---

## 2. Quản Lý Tham Số: File YAML & Pydantic `BaseModel`

### File `configs/data.yaml` mẫu (rút gọn)
```yaml
raw_data_path: "data/raw/creditcard.csv"
target_col: "Class"
test_size: 0.2
pca_features: ["V1", "V2", "V3"]
amount_feature: "Amount"
```

### Khai báo `class DataConfig(BaseModel)` trong `src/config.py`
```python
from pydantic import BaseModel

class DataConfig(BaseModel):
    raw_data_path: str
    target_col: str = "Class"
    test_size: float = 0.2
    pca_features: list[str]
    amount_feature: str = "Amount"

    @property
    def numeric_features(self) -> list[str]:
        return self.pca_features + [self.amount_feature]
```

### Cơ chế nạp dữ liệu: `DataConfig(**_load_yaml(...))`
```python
def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

data_config = DataConfig(**_load_yaml(CONFIGS_DIR / "data.yaml"))
```

#### 1. Hàm `yaml.safe_load()` biến đổi dữ liệu thế nào?
- Toàn bộ file YAML được chuyển thành **1 `dict` duy nhất** trong Python.
- Bên trong `dict` đó, YAML tự động ép kiểu:
  - Cặp `key: value` thành chuỗi hoặc số: `"test_size": 0.2` (float), `"target_col": "Class"` (str).
  - Các dòng có gạch đầu dòng `-` thành danh sách: `"pca_features": ["V1", "V2", "V3"]` (list).
- **Ý nghĩa chữ `safe_`**: Chỉ bóc tách dữ liệu thuần túy (dict, list, số, chuỗi), chặn đứng việc thực thi các đoạn mã độc hại ẩn trong file YAML.

#### 2. Toán tử giải nén `**` (Dictionary Unpacking)
- `**` là **cú pháp cốt lõi của Python** (dùng cho mọi hàm trong Python, KHÔNG PHẢI chỉ dành cho Pydantic).
- Nó "bung" các cặp `key: value` trong `dict` thành các đối số đặt tên truyền vào hàm:
  ```python
  # Từ dict: {"raw_data_path": "...", "test_size": 0.2}
  # Lệnh: DataConfig(**dict)
  # Được Python bung ra tương đương:
  DataConfig(raw_data_path="...", test_size=0.2)
  ```
- Nhờ vậy, `BaseModel` nhận đúng từng tham số để kiểm tra kiểu dữ liệu và gán vào object `data_config`.

#### 3. `@property` dùng để làm gì?
1. **Truy cập như biến:** Cho phép gọi `data_config.numeric_features` trực tiếp bằng dấu chấm (không cần dấu ngoặc `()`).
2. **Khóa thuộc tính thành Chỉ đọc (Read-only):** Ngăn chặn việc ai đó từ module khác vô tình viết code ghi đè làm hỏng danh sách feature dùng chung của hệ thống.
3. **Quy tắc bắt buộc:** Hàm gắn `@property` chỉ nhận `self`, **tuyệt đối không được nhận thêm tham số truyền vào từ bên ngoài**.

---

## 3. Hệ Thống Logging Thực Chiến: `logger` vs `print` & AWS CloudWatch

### 1. So sánh `print()` vs `logger`
```python
# Cách dùng print (Chỉ dùng học tập, nháp nhanh):
print("Dataset loaded")
# Kết quả: Dataset loaded (không rõ thời gian, không biết file nào in)

# Cách dùng logger (Chuẩn Production):
import logging
logger = logging.getLogger(__name__)
logger.info(f"Dataset loaded: {n_total:,} rows")
# Kết quả: 2026-09-10 18:52:05 | INFO | src.features:55 | Dataset loaded: 284,807 rows
```

### 2. Các cấp độ log (Log Levels)
`logger` chỉ làm nhiệm vụ **ghi nhận và phân loại thông tin** (kể cả `logger.error` cũng chỉ in ra dòng cảnh báo đỏ chứ **KHÔNG làm sập chương trình** như lệnh `raise Exception`):

| Mức độ | Mục đích sử dụng | Hành vi chương trình |
| :--- | :--- | :--- |
| `DEBUG` | Chi tiết kỹ thuật khi tìm lỗi (tọa độ biến, shape ma trận) | Không ảnh hưởng, chạy tiếp |
| `INFO` | Các mốc hoạt động bình thường (đã tải xong data, đã fit scaler) | Không ảnh hưởng, chạy tiếp |
| `WARNING` | Cảnh báo nguy cơ tiềm ẩn (dung lượng ổ cứng sắp đầy, dữ liệu lệch nhẹ) | Không ảnh hưởng, chạy tiếp |
| `ERROR` | Lỗi xảy ra nhưng app vẫn tự cứu được (thử lại kết nối database lần 2) | Không dừng app, chạy tiếp |
| `CRITICAL` | Lỗi cực kỳ nghiêm trọng (hỏng file model trọng số) | Báo động đỏ, app có thể dừng |

### 3. Luồng dữ liệu từ Code -> Docker -> AWS CloudWatch
```text
[Ứng dụng Python / FastAPI]
       │
       ▼ logger.info("...") hoặc logger.error("...")
[Console stdout/stderr của Container]
       │
       ▼ Docker Daemon tự động hứng (xem qua lệnh: docker logs <container>)
       │
       ▼ Cấu hình Docker driver: awslogs
[AWS CloudWatch Logs]
       ├── Lưu trữ tập trung mọi container trên đám mây
       ├── CloudWatch Logs Insights: Tìm kiếm log bằng câu lệnh SQL
       └── CloudWatch Alarms: Tự động gửi email/Slack khi có dòng logger.error()
```

---

## 4. Dự Đoán Xác Suất: `predict()` vs `predict_proba()` (Binary & Multi-class)

### 1. Tại sao KHÔNG THỂ dùng `predict()` để xét ngưỡng?
- `model.predict(X)` đã **tự ý dùng ngưỡng mặc định 0.5 bên trong thuật toán** để ép kết quả thành số nguyên cụt lủn: `0` hoặc `1`.
- Toàn bộ thông tin xác suất mềm ban đầu (ví dụ: `0.51` hay `0.99`) **bị xóa sạch hoàn toàn**. Nếu muốn xét ngưỡng linh hoạt (ví dụ `>= 0.80`), bạn không thể dùng các số `0, 1` để lọc được.
- Các hàm vẽ đường cong `PR-AUC` và `ROC-AUC` **bắt buộc phải nhận xác suất mềm** từ `predict_proba()` để quét qua hàng nghìn ngưỡng khác nhau.

### 2. Cách bóc tách mảng 2 chiều trong bài toán Nhị phân (Binary - 2 nhãn)
Hàm `predict_proba()` trả về ma trận 2 cột: Cột 0 (Sạch) và Cột 1 (Gian lận).
```python
# Cú pháp cắt mảng NumPy: [dòng, cột]
# Dấu ":" là lấy tất cả các dòng, số "1" là chỉ lấy cột số 1 (xác suất Gian lận)
y_proba = model.predict_proba(X)[:, 1]

# So sánh với ngưỡng tùy biến (Threshold)
y_pred = (y_proba >= threshold).astype(int)
```

### 3. Mở rộng: Khi có 3 nhãn trở lên (Multi-class), `predict_proba` dùng thế nào?
Hoàn toàn dùng được và cực kỳ mạnh mẽ! Khi có 3 class (ví dụ: `0 = Thấp`, `1 = Trung bình`, `2 = Gian lận cao`):
- Ma trận trả về sẽ có **đúng 3 cột** (tương ứng 3 class), tổng mỗi hàng luôn = `1.0`:
  ```text
                Cột 0 (Thấp)    Cột 1 (Trung bình)    Cột 2 (Gian lận)
  Giao dịch 1:  [   0.80,              0.15,               0.05       ]
  Giao dịch 2:  [   0.10,              0.60,               0.30       ]
  Giao dịch 3:  [   0.02,              0.08,               0.90       ]
  ```
- **Lấy xác suất của từng nhóm cụ thể:**
  - `y_proba[:, 0]`: Lấy toàn bộ xác suất rủi ro thấp.
  - `y_proba[:, 2]`: Lấy toàn bộ xác suất gian lận cao.
- **Chọn nhãn có xác suất cao nhất:**
  ```python
  y_pred = np.argmax(y_proba, axis=1)  # Trả về chỉ số cột có giá trị lớn nhất: [0, 1, 2]
  ```
- **Áp dụng luật nghiệp vụ ưu tiên (Business Rule):**
  Trong ngân hàng, dù nhãn 1 cao hơn (60%), nhưng chỉ cần nhãn Gian lận vượt `0.25` (`y_proba[:, 2] >= 0.25`) là hệ thống đã kích hoạt khóa thẻ khẩn cấp để bảo vệ tài khoản khách hàng!

---

## 5. Kiến Trúc Pipeline Huấn Luyện & Kỹ Thuật Unpacking `*` vs `**`

### 1. Phân biệt toán tử Unpacking: `*` (Tuple/List) vs `**` (Dictionary)
- **`*` (1 dấu sao - Positional Unpacking):** Bung một `tuple` hoặc `list` theo đúng **thứ tự vị trí** vào các tham số của hàm:
  ```python
  # Trong main(): gom 6 biến dữ liệu vào 1 tuple duy nhất
  splits = (X_train, X_val, X_test, y_train, y_val, y_test)

  # Thay vì phải viết dài dòng dễ nhầm lẫn:
  train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, grid_params, scale_pos_weight)

  # Chỉ cần dùng dấu * để bung tự động vào 6 tham số đầu tiên:
  train_xgboost(*splits, grid_params, scale_pos_weight)
  ```
- **`**` (2 dấu sao - Keyword Unpacking):** Bung một `dict` thành các cặp đối số đặt tên dạng `key=value`:
  ```python
  # Gộp nhiều dict tham số thành 1 dict duy nhất:
  params = {**XGBOOST_BASE_PARAMS, **grid_params, "scale_pos_weight": scale_pos_weight}

  # Bung dict vào hàm khởi tạo mô hình:
  model = XGBClassifier(**params)  # Tương đương: XGBClassifier(max_depth=6, learning_rate=0.1, ...)
  ```

### 2. Thiết kế hàm huấn luyện linh hoạt (Config-driven Architecture)
- **Tách biệt dữ liệu cấu hình:** Mọi siêu tham số được khai báo trong `configs/models.yaml` (dưới dạng danh sách các dict trong `xgboost_grid`).
- **Hàm huấn luyện đơn nhiệm (Single-responsibility):**
  ```python
  def train_xgboost(..., grid_params: dict, scale_pos_weight: float) -> dict[str, float]:
      # Nhận 1 dict cấu hình bất kỳ, không bị gò bó (hard-code) tham số cụ thể
  ```
- **Vòng lặp trong `main()` & Vai trò của `copy.deepcopy`:**
  ```python
  import copy

  for grid_params in copy.deepcopy(XGBOOST_GRID):
      # Bắt buộc dùng deepcopy vì bên trong train_xgboost có lệnh grid_params.pop("run_name")
      # Nếu chỉ truyền trực tiếp, lệnh pop() sẽ xóa mất key trong cấu hình gốc dùng chung
      metrics = train_xgboost(*splits, grid_params, scale_pos_weight)
  ```

---

## 6. Cấu Trúc Dự Án: Thư Mục `src/` vs `scripts/` & Ứng Dụng Trong CI/CD

### 1. Bản chất: `src/` khác gì `scripts/`?
Về mặt kỹ thuật, cả hai đều chứa file Python `.py` bình thường. Sự khác biệt nằm ở **quy ước kiến trúc chuẩn của ngành phần mềm**:

| Tiêu chí | Thư mục `src/` (Core Package) | Thư mục `scripts/` (Operational Tooling) |
| :--- | :--- | :--- |
| **Vai trò** | Thư viện lõi (chứa hàm, class, logic tiền xử lý, định nghĩa API). | Kịch bản chạy tác vụ cụ thể (chọn model, xuất file `.pkl`, đo độ trễ API). |
| **Đóng gói** | Được đóng gói vào ứng dụng/Docker để các module khác import. | **Không** đóng gói; chỉ là file kịch bản chạy từ ngoài terminal. |
| **Quy tắc phụ thuộc** | **Cấm** import bất cứ thứ gì từ `scripts/` (tránh ô nhiễm kiến trúc). | **Được phép** import từ `src/` để ráp nối quy trình vận hành. |

### 2. Khi nào được sử dụng trong CI/CD?
- **Khi tạo Pull Request (CI - Kiểm tra):**
  - CI chạy linter (`ruff check scripts/`) để bảo đảm script không bị lỗi cú pháp.
  - CI chạy thử nghiệm mô phỏng với cờ `--dry-run` để test quy trình kiểm duyệt mà **không đụng vào Registry thật**.
- **Khi triển khai lên server (CD - Vận hành thật):**
  - Máy chủ CI/CD tự động gọi `python scripts/select_best_model.py` và `python scripts/export_model.py` để tự động bốc model vô địch và đóng gói vào Docker container trước khi deploy lên AWS EC2 hoặc Render.

### 3. Ví dụ đối chiếu trực quan: `src/validate.py` vs `scripts/select_best_model.py`

| Tiêu chí | `src/validate.py` (Lõi thẩm định - Core Library) | `scripts/select_best_model.py` (Kịch bản điều phối - Script) |
| :--- | :--- | :--- |
| **Hình tượng** | **"Vị Thẩm Phán"**: Chỉ quan tâm luật lệ, chấm điểm xem model có đủ chuẩn hay không. | **"Nhạc Trưởng"**: Đi tìm ứng viên sáng giá nhất, dắt tới Thẩm phán, rồi hoàn tất thủ tục đăng ký. |
| **Các hàm chính** | - `_check_minimum_thresholds()`: Soi ngưỡng sàn Recall >= 0.80, Precision >= 0.50.<br>- `_get_production_metrics()`: Lấy điểm của model Production cũ.<br>- `run_validation_gate()`: **Hàm trung tâm** so sánh Candidate vs Production và ra phán quyết.<br>- `print_validation_report()`: In bảng kết quả thẩm định. | - `get_all_runs()`: Quét toàn bộ các run trong MLflow, sắp xếp theo PR-AUC.<br>- `print_comparison_table()`: In bảng so sánh đối chiếu toàn bộ các run.<br>- `register_best_model()`: Đăng ký vào Registry và ghi Description chi tiết.<br>- `select_and_register()`: **Hàm điều phối** xâu chuỗi toàn bộ quy trình 5 bước. |
| **Sự liên kết** | Đứng yên một chỗ, cung cấp hàm `run_validation_gate()` cho bên ngoài gọi vào. | **Import và gọi hàm** của validate:<br>`from src.validate import run_validation_gate`<br>Chỉ khi Thẩm phán phán quyết PASS, script này mới gọi `register_best_model()`! |

### 4. Quy trình đưa 1 Run mới lên Production (Ví dụ kịch bản 9 Runs)

Giả sử trong MLflow hiện có **9 runs** đã train từ trước. Trong đó, **Run 3** từng là quán quân và đang giữ nhãn `alias="production"` với điểm số: `val_pr_auc = 0.8650`.

Khi bạn huấn luyện đợt mới (sinh ra thêm Run 10, 11, 12...), quy trình chuẩn diễn ra qua **3 bước tuần tự**:

```text
[Bước 1: Huấn luyện]            python -m src.train
                                  └──> Sinh ra Run 10, 11, 12 trong MLflow. (Production cũ Run 3 vẫn an toàn)
                                          │
                                          ▼
[Bước 2: Tuyển chọn & Thẩm định] python scripts/select_best_model.py
                                  ├──> Quét toàn bộ: Thấy Run 11 tốt nhất đợt mới (val_pr_auc = 0.8810)
                                  ├──> Kích hoạt Validation Gate so găng: 0.8810 (Run 11) >= 0.8650 (Run 3)
                                  └──> Kết quả: [PROMOTED] -> Chuyển tag alias='production' sang Run 11!
                                          │
                                          ▼
[Bước 3: Xuất & Triển khai]     python scripts/export_model.py
                                  └──> Kéo model mang tag 'production' (chính là Run 11) lưu thành file .pkl
                                  └──> FastAPI nạp file .pkl này để phục vụ dự đoán gian lận cho khách hàng.
```

- **Lưu ý thực tế:**
  - Bắt buộc phải chạy **Bước 1 (train)** trước để sinh ra run mới trong MLflow rồi mới chạy **Bước 2 (select)**.
  - Nếu chỉ muốn thử nghiệm riêng một run lẻ (`run_id="abc123"`), bạn có thể gọi thẳng lệnh thẩm định độc lập mà không cần quét lại cả kho:
    `python -m src.validate --run-id abc123`

---

## 7. Phân Biệt Pydantic Trong `config.py` vs Trong `api.py`

Cả hai file đều sử dụng Pydantic, nhưng phục vụ 2 vai trò hoàn toàn tách biệt trong kiến trúc hệ thống:

### 1. Bảng so sánh bản chất & Mục đích

| Tiêu chí | Pydantic trong `src/config.py` | Pydantic trong `src/api.py` |
| :--- | :--- | :--- |
| **Vai trò** | **Cấu hình hệ thống (System Config)** | **Hợp đồng dữ liệu mạng (API Data Contract)** |
| **Thời điểm chạy** | Chạy **1 lần duy nhất** lúc dự án/module khởi động. | Chạy **liên tục cho từng request** gửi tới qua internet. |
| **Nguồn dữ liệu vào** | File `.env`, file `data.yaml`, biến môi trường OS. | Gói tin JSON từ client (web frontend, app mobile, ngân hàng). |
| **Xử lý khi có lỗi** | Chặn đứng ứng dụng không cho khởi động (Fail-fast). | Trả về mã lỗi `HTTP 422 Unprocessable Entity` cho client. |

### 2. Ví dụ đối chiếu cụ thể: Ràng buộc & Điều kiện giá trị

#### A. Trong `src/config.py`: Quản lý danh mục đặc trưng & Tham số hệ thống
```python
# config.py chỉ định nghĩa khung danh mục và tham số tĩnh:
class DataConfig(BaseModel):
    raw_data_path: str
    target_col: str = "Class"
    pca_features: list[str] = ["V1", "V2", ..., "V28"]
    amount_feature: str = "Amount"

class AppSettings(BaseSettings):
    DECISION_THRESHOLD: float = 0.5  # Ngưỡng mặc định của toàn hệ thống
```
- **Ý nghĩa:** `config.py` chỉ định nghĩa **khung danh mục** (hệ thống cần những feature nào, file data ở đâu). Nó không biết và không quan tâm dữ liệu cụ thể của từng khách hàng.

#### B. Trong `src/api.py`: Ràng buộc giá trị chi tiết của từng giao dịch (Payload Validation)
```python
# api.py kiểm soát chi tiết từng con số của gói tin JSON khách hàng gửi lên:
class TransactionInput(BaseModel):
    V1: float
    V2: float
    ...
    # Ràng buộc điều kiện giá trị chi tiết cho từng trường:
    Amount: float = Field(..., ge=0.0, description="Số tiền giao dịch phải >= 0")

    model_config = {
        "allow_inf_nan": False,  # Chặn tuyệt đối NaN và Vô cực
    }
```
- **Ý nghĩa:** `api.py` trực tiếp cầm từng con số của khách hàng lên cân đo:
  + Nếu khách hàng gửi `Amount = -10` hoặc `Amount = "abc"`: Bị Pydantic trong `api.py` chặn đứng ngay tại cổng với lỗi HTTP 422.
  + Sau đó, `DECISION_THRESHOLD` từ `config.py` được gọi vào để so sánh: `is_fraud = bool(proba >= DECISION_THRESHOLD)`.

### 3. Mối liên kết giữa 2 tầng
`api.py` **không tự tiện khai báo lại từ đầu**, mà nó **kế thừa các hằng số** từ `config.py`:
- `_AMOUNT_MEAN` và `_AMOUNT_STD` được import từ `config.py` để chuẩn hóa cột `Amount`.
- `FEATURE_COLS` được import từ `config.py` để bảo đảm thứ tự 29 cột đưa vào `model.predict_proba()` luôn khớp 100% với lúc huấn luyện!

---

## 8. Phòng Thủ Chiều Sâu (Defense in Depth) & Cơ Chế Output Validation (`response_model`)

### 1. Tại sao đã có Pydantic mà vẫn cần Quality Gate và Evidently?

Đây là nguyên lý **Defense in Depth (Phòng thủ theo chiều sâu)** trong Enterprise MLOps. Ba công cụ này quản lý 3 tầng hoàn toàn khác biệt của luồng dữ liệu:

| Tiêu chí | Tầng 1: Pydantic (`TransactionInput`) | Tầng 2: Quality Gate (`src/quality.py`) | Tầng 3: Evidently (`src/monitor.py`) |
| :--- | :--- | :--- | :--- |
| **Phạm vi & Vị trí** | **Network / HTTP Layer** (Cửa ngõ API) | **Data Contract / Tabular Layer** (Bảng DataFrame trước Model) | **Statistical Distribution Layer** (Giám sát phân phối & Trôi dạt) |
| **Đối tượng kiểm tra** | Từng gói tin JSON đơn lẻ gửi qua mạng internet. | Cấu trúc bảng DataFrame và các mảng vector số. | Toàn bộ tập dữ liệu production qua thời gian vs Tập huấn luyện chuẩn. |
| **Nội dung kiểm tra** | - Kiểu dữ liệu cơ bản (float, int, string).<br>- Trường bắt buộc có đủ không.<br>- Chặn NaN/Inf trong JSON (`allow_inf_nan=False`).<br>- Giá trị đơn lẻ (`Amount >= 0`). | - Trùng lặp tên cột trong DataFrame (`df.columns.duplicated()`).<br>- Batch rỗng (`empty_batch`).<br>- Tỷ lệ Null/Missing trên toàn bộ batch.<br>- Infinite trên mảng numpy (`np.isfinite`). | - Phân phối thống kê của các đặc trưng V1-V28 và Amount có bị lệch không.<br>- Độ trôi dạt (Data Drift / Concept Drift).<br>- Thuật toán KS-test, Wasserstein distance, PSI. |
| **Khả năng tái sử dụng** | Chỉ chạy trong môi trường Web API (FastAPI). | **Chạy độc lập ở mọi nơi**: Pipeline train offline, Batch Inference, Monitoring job... (những nơi không có FastAPI/HTTP). | Chạy offline hoặc định kỳ (Scheduled cron / Batch job) để cảnh báo Retrain. |
| **Khi vi phạm thì sao?** | Trả về `HTTP 422 Unprocessable Entity` ngay lập tức. | Ném `ValueError` hoặc trả về danh sách `issues` để chặn tính toán ma trận. | Ghi log, bắn alert về Grafana/Slack để kỹ sư MLOps retrain model. |

#### Ví dụ thực tế về sự khác biệt giữa Tính Hợp Lệ (Validity) và Phân Phối (Distribution):
- Giả sử có một đợt tấn công gian lận tinh vi mới: 100% giao dịch gửi lên đều là số thực `float`, không hề bị null, số tiền `Amount > 0`.
- **Pydantic và Quality Gate**: Cho qua 100% vì dữ liệu hoàn toàn đúng cú pháp và đúng kiểu.
- **Evidently**: Phát hiện bất thường vì trước đây giao dịch trung bình chỉ $88, đợt này xuất hiện bất thường toàn giao dịch $9,999 hoặc phân phối các đặc trưng PCA $V1 - V28$ bị lệch hẳn sang một miền giá trị mới. Evidently phát hiện hiện tượng này để cảnh báo retrain trước khi model bị sụt giảm độ chính xác âm thầm (Silent Failure).

---

### 2. Cơ Chế Output Validation Với `response_model` Trong FastAPI

Trong `src/api.py`:
```python
@app.post(
    "/predict",
    response_model=PredictionResponse,  # Khai báo hợp đồng dữ liệu đầu ra
)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    ...
    return _predict_one(transaction)
```

`PredictionResponse` là một Pydantic BaseModel:
```python
class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold: float
    model_name: str
```

#### A. Nếu code backend tính toán sai kiểu hoặc thiếu trường thì sao?
- **FastAPI chặn đứng ngay tại cửa ra:** Pydantic sẽ ném ra ngoại lệ `ResponseValidationError`.
- **Khách hàng KHÔNG BAO GIỜ nhận được dữ liệu lỗi:** FastAPI tự động bắt lỗi này và chuyển đổi thành mã `HTTP 500 Internal Server Error`. Khách hàng chỉ thấy thông báo server gặp sự cố, tuyệt đối không nhận được dữ liệu rác hay định dạng méo mó.
- **Log hệ thống sẽ ghi rõ vị trí sai:** Server log sẽ in chi tiết trường nào bị sai kiểu để lập trình viên sửa lỗi ngay lập tức.

#### B. Nếu KHÔNG dùng `response_model` mà chỉ return dictionary bình thường thì sao?
- **Không có màng lọc đầu ra:** FastAPI chỉ đơn thuần biến dictionary đó thành chuỗi JSON và **bắn thẳng về cho khách hàng**!
- **Hậu quả nghiêm trọng:**
  1. **Làm sập hệ thống downstream của khách:** Nếu backend tính nhầm trả về `is_fraud = "yes"` (chuỗi string thay vì boolean `true/false`), hoặc `fraud_probability = None`, ứng dụng Mobile hoặc Cổng thanh toán của khách hàng đang chờ kiểu Boolean để chặn thẻ sẽ bị **crash runtime** ngay lập tức.
  2. **Rò rỉ thông tin nội bộ (Data Leakage):** Nếu dictionary nội bộ vô tình chứa các thông tin nhạy cảm (như database ID, token nội bộ, model weights), việc không có `response_model` sẽ khiến toàn bộ thông tin đó bị trả về cho client ngoài internet. Khi có `response_model`, Pydantic chỉ trích xuất và serialize đúng các trường đã được khai báo, tự động lọc bỏ mọi dữ liệu thừa thãi.

---

## 9. FastAPI Lifespan & Cơ Chế Quản Lý Vòng Đời Model Machine Learning

### 1. Bản chất của Lifespan là gì?

Trong FastAPI hiện đại, `lifespan` là chuẩn quản lý vòng đời ứng dụng (thay thế hoàn toàn cơ chế cũ `@app.on_event("startup")` và `shutdown` đã bị deprecated):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [PHẦN 1: STARTUP - CHẠY 1 LẦN DUY NHẤT KHI BẬT SERVER]
    # Nạp tài nguyên nặng từ ổ đĩa/mạng lên bộ nhớ RAM:
    _model, _model_info = load_model()
    
    yield  # [ỨNG DỤNG BẮT ĐẦU CHẠY VÀ PHỤC VỤ KHÁCH HÀNG TẠI ĐÂY]
    
    # [PHẦN 2: SHUTDOWN - CHẠY 1 LẦN DUY NHẤT KHI TẮT SERVER]
    # Dọn dẹp bộ nhớ, đóng kết nối:
    _model = None
```

Được đăng ký thẳng vào ứng dụng:
```python
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
```

### 2. Tại sao Model Machine Learning BẮT BUỘC phải nạp trong Lifespan?

- **Quy tắc vàng hiệu năng (Latency):**
  - Model máy học là tài nguyên nặng (từ hàng chục MB đến hàng GB). Việc đọc từ ổ cứng hoặc tải qua mạng mất từ 1 đến 5 giây.
  - **Nếu nạp sai chỗ (ví dụ nạp bên trong hàm `predict()`):** Cứ mỗi request gửi tới, server lại đọc đĩa nạp model lại từ đầu -> Độ trễ vọt lên vài giây/request, RAM bị phình to và server sập ngay lập tức!
  - **Nạp đúng chuẩn trong `lifespan`:** Model chỉ được nạp **1 lần duy nhất** khi khởi động server và lưu sẵn trên RAM (`_model`). Khi khách gửi request vào `/predict`, model đã sẵn sàng trên RAM nên tính toán trả kết quả chỉ mất **vài mili-giây**!

---

## 10. Kiến Trúc Giám Sát Thời Gian Thực: Prometheus Telemetry, ASGI Middleware & Scrape Pipeline

### 1. Bản chất của `HTTPMetricsMiddleware` & Đăng ký bằng `app.add_middleware`

Trong FastAPI / Starlette, có 2 cách viết Middleware:
1. **Cách cơ bản (Beginner):** Dùng decorator `@app.middleware("http")`. Cách này dễ viết nhưng có overhead hiệu năng và dễ bị lỗi luồng dữ liệu stream lớn.
2. **Cách chuẩn sản xuất Enterprise (Class-based ASGI Middleware):** Viết một class nhận `app: ASGIApp` và cài đặt hàm `async def __call__(self, scope, receive, send)`.

- **Khi gọi `app.add_middleware(HTTPMetricsMiddleware)`**:
  - Có hiệu lực **ngay lập tức** cho toàn bộ ứng dụng!
  - Nó đứng chặn ở tầng thấp nhất (tầng ASGI mạng) trước khi request chạm vào bất kỳ hàm logic nào.
  - Tự động bấm giờ bằng `time.perf_counter()` (độ chính xác micro-giây).
  - Tráo đổi chiếc loa phát thanh bằng `send_with_status` để bắt chính xác `status_code` (200, 422, 500) mà không làm gián đoạn hay biến dạng dữ liệu trả về cho khách.
  - **Chống lỗi High Cardinality (Tràn nhãn):** Chỉ lấy đường dẫn template (`/predict`, `/health`) chứ không lấy ID giao dịch động, bảo vệ RAM của Prometheus không bị phình to.

### 2. Bản chất câu lệnh `return Response(...)` tại Endpoint `/metrics`

```python
@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})
```

- **`Response` ở đâu ra?**: Được import trực tiếp từ FastAPI: `from fastapi import Response`. Đây là class phản hồi HTTP cơ bản nhất của FastAPI dùng để trả về chuỗi văn bản thô (Raw Text), thay vì tự động ép sang JSON như các hàm thông thường.
- **`generate_latest(REGISTRY)`**: Là hàm của thư viện `prometheus_client`. Nó đọc toàn bộ các số liệu đo lường đang lưu tạm trong RAM của biến `REGISTRY` và chuyển thành định dạng văn bản chuẩn **Prometheus Exposition Format (Line Protocol)**.
- **`headers={"Content-Type": CONTENT_TYPE_LATEST}`**: Báo cho người nhận biết đây là định dạng `text/plain; version=0.0.4; charset=utf-8`.
- **Tại sao không trả về JSON?**: Vì máy chủ Prometheus Server không đọc JSON! Chuẩn quốc tế của Prometheus quy định các dòng text có cấu trúc:
  ```text
  fraud_http_requests_total{method="POST",route="/predict",status_class="2xx"} 15.0
  fraud_http_request_duration_seconds_sum{method="POST",route="/predict"} 0.0637
  fraud_http_requests_in_flight 0.0
  ```

### 3. Chuỗi mắt xích toàn diện: Từ Code Python đến Prometheus Container & Grafana

Code Python tự nó **chỉ tính toán và lưu số liệu trong RAM**, chưa hề kết nối hay đẩy đi đâu cả. Để số liệu lên được biểu đồ Grafana, toàn bộ quy trình vận hành qua 3 bước:

```text
[Bước 1: Ứng dụng Python (Port 8000)]
FastAPI + HTTPMetricsMiddleware
  └── Tính toán độ trễ, đếm request, lưu số liệu vào RAM
  └── Mở sẵn cổng http://127.0.0.1:8000/metrics (chờ người đến lấy)
             │
             ▲ [Kéo dữ liệu định kỳ mỗi 15s - Pull Model]
             │
[Bước 2: Máy chủ Prometheus (Port 9090)]
Đọc cấu hình từ monitoring/prometheus.yml:
  global:
    scrape_interval: 15s
  scrape_configs:
    - job_name: fraud-api
      metrics_path: /metrics
      static_configs:
        - targets: ["127.0.0.1:8000"]
  └── Định kỳ mỗi 15s gửi GET /metrics vào Port 8000 để cào (scrape) số liệu
  └── Lưu toàn bộ lịch sử biến động vào Time-Series Database của Prometheus
             │
             ▼ [Truy vấn số liệu để vẽ biểu đồ]
             │
[Bước 3: Giao diện Grafana (Port 3000)]
Kết nối vào Prometheus bằng PromQL:
  └── rate(fraud_http_requests_total[1m]) -> Biểu đồ số request/giây
  └── histogram_quantile(0.95, sum(rate(fraud_http_request_duration_seconds_bucket[5m])) by (le)) -> Biểu đồ độ trễ p95
```








