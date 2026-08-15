# 🧠 Kiến thức cốt lõi: MLflow & Tư duy MLOps

> [!info] Mục đích tài liệu
> 
> Tài liệu này lưu trữ các khái niệm quan trọng ở Bước 1 & 2 của Project Fraud Detection, tập trung vào công cụ Tracking (MLflow) và Tư duy thiết kế hệ thống dễ mở rộng (Extensibility) khác biệt với cách làm Data Science truyền thống.

## 1. Mổ xẻ MLflow: "Cuốn nhật ký tự động" của Kỹ sư ML

> [!abstract] Định nghĩa ngắn gọn
> 
> Nếu **Git** dùng để quản lý phiên bản _Code_, thì **MLflow** dùng để quản lý phiên bản _Model_ (bao gồm Code, Data, Tham số, Điểm số và File model `.pkl`).

### 🕰️ MLflow "Xưa và Nay": File Store vs Database Backend

Trong các script cũ hoặc hướng dẫn trên mạng, bạn thường thấy MLflow tự tạo một thư mục tên là `mlruns/` và nhét hàng ngàn file vào đó. Đây là cách làm cũ và đã bị MLflow 3.x **đánh dấu lỗi thời (deprecated)**.

#### 🔴 Xưa: Cấu trúc thư mục `mlruns/` (File Store)

Khi không dùng Database, MLflow lưu mọi thứ thành từng file text/yaml nhỏ giọt.

```
mlruns/
├── 0/                                  # ID của Experiment
│   ├── bc6dc2a4f38d47b4b0c.../         # ID của một lần Run 
│   │   ├── metrics/                    
│   │   │   └── f1_score                # File text, mở ra chỉ ghi "0.85"
│   │   │   └── recall                  # File text, mở ra chỉ ghi "0.82"
│   │   ├── params/                     
│   │   │   ├── max_depth               # File text, mở ra ghi "5"
│   │   ├── artifacts/                  
│   │   │   └── model.pkl               # File mô hình thật
```

> [!danger] Hạn chế của cấu trúc File rời rạc:
> 
> - **Không có Indexing (Chỉ mục):** Nếu có 1,000 runs và bạn muốn tìm "model có F1 > 0.8", MLflow phải đi mở 1,000 cái file `f1_score.txt` lên để đọc thủ công -> **Giao diện UI cực kỳ chậm và giật lag.**
>     
> - **Tắc nghẽn I/O ổ cứng:** Sinh ra hàng ngàn thư mục và file rác. Nếu nhiều luồng chạy song song cùng lúc, rất dễ bị lỗi đụng độ file (file lock).
>     

#### 🟢 Nay: Database Backend (`sqlite:///mlflow.db`)

Để giải quyết vấn đề trên, chuẩn hiện tại là lưu **Metadata** (Tham số, Điểm số, Tags) vào một cơ sở dữ liệu. Ở local, ta dùng SQLite (`sqlite:///mlflow.db`).

Thay vì hàng vạn file rác, toàn bộ lịch sử được đóng gói vào **1 file `mlflow.db` duy nhất**. Bên trong file này là các bảng chuẩn SQL:

|   |   |   |   |
|---|---|---|---|
|**run_uuid**|**key**|**value**|**timestamp**|
|bc6dc2a4f...|f1_score|0.85|1692100000|
|bc6dc2a4f...|max_depth|5.0|1692100000|
|a1b2c3d4e...|f1_score|0.88|1692100500|

> [!success] Sự khác biệt mang tính cách mạng:
> 
> - **Truy vấn siêu tốc:** Việc lọc "F1 > 0.8" giờ đây chỉ là một câu lệnh SQL (`SELECT... WHERE key='f1_score' AND value > 0.8`). Hệ thống SQL xử lý việc này trong một phần nghìn giây. UI chạy mượt mà ngay cả với 10,000 runs.
>     
> - **Gọn gàng:** Chỉ 1 file quản lý thông số, dễ dàng chia sẻ hoặc xóa bỏ. (Lưu ý: Các file nặng như `.pkl` vẫn được lưu riêng trong thư mục `mlartifacts` để không làm nặng DB).
>     

![[Pasted image 20260815154708.png]]
### Cấu trúc Experiment và Run

> [!example] Ví dụ thực tế
> 
> Tưởng tượng **Experiment** là "Thư mục Dự án", còn **Run** là "Các lần làm bài kiểm tra".

- **Experiment:** `fraud-detection` (Dự án phát hiện gian lận thẻ).
    
    - **Run 1:** Train bằng Logistic Regression -> _F1: 0.6_ -> Lưu file `lr_v1.pkl`
        
    - **Run 2:** Train bằng XGBoost -> _F1: 0.8_ -> Lưu file `xgb_v1.pkl`
        
    - **Run 3:** Train XGBoost nhưng thêm dữ liệu -> _F1: 0.85_ -> Lưu file `xgb_v2.pkl`
        

Tất cả các Run này được gom chung vào bảng `fraud-detection`, giúp bạn dễ dàng mở biểu đồ so sánh đường hội tụ (loss curve) của chúng với nhau.

## 2. Tư duy thiết kế Feature: Code MLOps vs Code Notebook

Đây là phần tạo nên sự khác biệt giữa một "Sinh viên làm đồ án" và một "Kỹ sư MLOps thực thụ".

### Kịch bản thực tế

Hiện tại dữ liệu có biến `V1` đến `V28` và `Amount`. Tháng sau, team Data Engineer thu thập thêm một cột mới là `Device_Type` (Loại thiết bị: iOS, Android, PC) và muốn bạn đưa vào mô hình.

### 🔴 Nếu KHÔNG dùng tư duy MLOps (Cách làm cũ/Hạn chế)

Họ thường viết "Hardcode" (code chết) hoặc lọc cột dựa trên cảm tính trực tiếp vào file train:

```
# Cách 1: Thả trôi (Rất nguy hiểm)
X_train = data.drop(columns=['Class', 'Time']) 
```

> [!danger] Hạn chế của Cách 1:
> 
> Nếu Data Engineer vô tình chèn thêm cột `Customer_Name` vào database. Hàm `.drop()` không biết điều đó, nó đưa luôn Tên khách hàng vào huấn luyện -> Mô hình học vớ vẩn (Garbage in, garbage out).

```
# Cách 2: Hardcode rải rác
X_train = data[['V1', 'V2', 'V3', ..., 'Amount']] # Viết tay ở file train.py

# ... sang file app.py (API web) lại phải viết lại:
user_input = request_data[['V1', 'V2', 'V3', ..., 'Amount']]
```

> [!warning] Hạn chế của Cách 2:
> 
> Khi muốn thêm `Device_Type`, họ phải mở file `train.py` ra sửa. Sau đó mở file `evaluate.py` ra sửa. Sau đó mở `app.py` ra sửa. Nếu quên sửa ở API web -> Hệ thống sập (Crash) ngay khi đẩy lên Production vì input thực tế lệch với input model cần.

### 🟢 Khi dùng tư duy MLOps (Quản lý tập trung qua `config.py`)

Trong MLOps, mọi "sự thật" (Single Source of Truth) về schema dữ liệu chỉ được lưu ở **1 NƠI DUY NHẤT**:

```
# Trong config.py
FEATURE_COLS = ["V1", "V2", ..., "Amount", "Device_Type"] # <-- Chỉ thêm ở đây
```

Và ở **tất cả** các file khác (`train.py`, `evaluate.py`, `app.py`):

```
from src.config import FEATURE_COLS

X = data[FEATURE_COLS] # Tự động ăn theo config
```

> [!success] Lợi ích khổng lồ:
> 
> 1. **Kiểm soát rủi ro (Data Contract):** Nếu database có cột rác mới, hệ thống tự động phớt lờ vì nó chỉ lấy đúng các cột có trong `FEATURE_COLS`.
>     
> 2. **Dễ bảo trì (Maintainability):** Khi update dữ liệu, chỉ cần sửa đúng 1 dòng ở `config.py`. CI/CD pipeline tự động chạy lại, API tự động cập nhật schema mới. Bạn đi ngủ ngon giấc mà không sợ lỗi "lạc trôi" schema.

## 3. 🔍 Giải phẫu các lệnh Log: Tags, Params và Metrics

> [!info] Triết lý "Ghi chép trước, Huấn luyện sau"
> 
> Trong code MLOps, ta luôn gọi các lệnh log (`set_tag`, `log_param`) **trước** khi chạy hàm `model.fit()`. Điều này đảm bảo: Nếu quá trình train bị sập giữa chừng (do tràn RAM, lỗi data), hệ thống DB vẫn đã kịp ghi lại "nguyên liệu" gây ra lỗi đó để ta debug.

Dù `set_tag` và `log_param` có vẻ giống nhau, nhưng chúng phục vụ mục đích hoàn toàn khác biệt và được lưu ở 3 bảng riêng biệt trong SQLite theo mô hình **EAV (Entity-Attribute-Value)** để không làm phình to database:

### 🏷️ 1. `set_tag` (Gắn nhãn dán)

- **Bản chất:** Giống như dán tờ giấy note lên nồi phở (Ví dụ: "Phở của Quân", "Mô hình Baseline"). Việc bóc nhãn này ra không làm thay đổi chất lượng mô hình.
    
- **Mục đích:** Để **tìm kiếm và phân loại** trên giao diện UI (`tags.purpose = 'baseline'`).
    
- **Lưu trong SQLite:** Bảng `tags` gồm 3 cột: `[run_uuid, key, value]`.
    

### ⚙️ 2. `log_param` (Lưu công thức & Cấu hình)

- **Bản chất:** Giống như công thức nấu ăn (Ví dụ: 2 thìa muối, hầm 8 tiếng). Nếu thay đổi param (`max_iter`, `threshold`), mô hình sẽ thay đổi hoàn toàn.
    
- **Mục đích:** Để **so sánh** nguyên nhân tại sao mô hình A lại tốt hơn mô hình B.
    
- **Lưu trong SQLite:** Bảng `params` gồm 3 cột: `[run_uuid, key, value]`.
    

> [!tip] Kỹ thuật Dictionary Unpacking (`**`)
> 
> Thay vì viết tay từng param: `LogisticRegression(max_iter=1000, solver='lbfgs')`, MLOps dùng `LogisticRegression(**LR_PARAMS)`. Dấu `**` sẽ tự động "bung" từ điển trong file `config.py` ra và nhét vào hàm. Giúp file `train.py` sạch sẽ và không bao giờ phải sửa code khi đổi tham số.

### 📈 3. `log_metrics` (Lưu kết quả / Điểm số)

- **Bản chất:** Là kết quả bài kiểm tra sau khi mô hình đã học xong (Precision, Recall, PR-AUC).
    
- **Lưu trong SQLite:** Bảng `metrics`. Bảng này đặc biệt hơn vì có **5 cột**: `[run_uuid, key, value, timestamp, step]`.
    
- **Cột Step (Vòng lặp) để làm gì?** Dùng trong Deep Learning. Hệ thống sẽ lưu điểm số ở từng epoch (step 1, step 2...) để sau này giao diện MLflow tự động nối các điểm đó lại thành đồ thị học tập (Learning Curve).
    

## 4. 📦 Chiến lược lưu trữ Artifact: File rác vs. Package chuẩn

Artifact là các tệp tin vật lý (ảnh, file csv, file pkl) được lưu trên ổ cứng (thư mục `mlruns/`), không lưu trong SQLite. MLflow chia Artifact thành 2 đẳng cấp:

> [!warning] Cấp độ 1: `log_artifact` (Nhà kho chứa đồ câm)
> 
> - **Mục đích:** Lưu trữ các file phụ trợ như hình ảnh biểu đồ (`pr_curve.png`), báo cáo dạng text.
>     
> - **Đặc điểm:** Hệ thống chỉ đơn thuần copy file quăng vào thư mục. Nó phục vụ cho **mắt người** nhìn trên giao diện UI, máy móc không thể tự hiểu bên trong có gì.
>     

> [!success] Cấp độ 2: `log_model` (Trái tim của MLOps)
> 
> - **Mục đích:** Đóng gói mô hình thành một **Package phần mềm** hoàn chỉnh, sẵn sàng đem đi kiếm tiền (Production).
>     
> - **Khi chạy lệnh này, nó sinh ra 4 thứ:**
>     
>     1. `model.pkl`: File trọng số của mô hình.
>         
>     2. `requirements.txt`: Tự động quét và ghi lại phiên bản scikit-learn, pandas...
>         
>     3. `conda.yaml`: Cấu hình môi trường ảo.
>         
>     4. `MLmodel`: File định tuyến đặc biệt của MLflow.
>         
> - **Sức mạnh:** Tháng sau bạn chỉ cần gõ 1 lệnh `load_model()`, MLflow sẽ tự dựng lại y hệt môi trường ngày xưa và chạy dự đoán mà không sợ lỗi xung đột phiên bản thư viện.
>     

### ⏳ Khi nào thì lưu Model và Metrics?

Một sai lầm phổ biến là lưu mô hình ở mọi vòng lặp (epoch), làm ổ cứng nổ tung. Chuẩn MLOps quy định:

1. **Trong quá trình Train (Tại mỗi Epoch):**
    
    - **CHỈ LƯU METRICS:** Ghi lại loss/score bằng văn bản (rất nhẹ) để vẽ biểu đồ. Tuyệt đối không lưu file model.
        
2. **Sau khi Train kết thúc:**
    
    - **LƯU 1 MODEL DUY NHẤT:** Dùng kỹ thuật Checkpointing để trích xuất phiên bản mạnh nhất và dùng `log_model` đóng gói nó.
        
    - **VẼ 1 BIỂU ĐỒ DUY NHẤT:** Lấy model mạnh nhất đó làm bài thi trên `X_test`, vẽ ra 1 bức ảnh PR Curve và dùng `log_artifact` cất vào kho.


---

## 5. 🏷️ MLflow Model Registry & Vòng đời Production (Session 3)

> [!abstract] Định nghĩa Model Registry
> 
> Nếu **Experiment Tracking** là nơi lưu trữ nháp hàng ngàn thử nghiệm (Runs), thì **Model Registry** là "Kho hàng bảo vệ" tập trung chỉ chứa những Model **tốt nhất đã được tuyển chọn** để sẵn sàng đưa ra thị trường.

```
MLflow Experiments (Nháp/Thử nghiệm)             MLflow Model Registry (Kho Production)
┌──────────────────────────────────────┐          ┌─────────────────────────────────────────┐
│ Run 1: LR Baseline (PR-AUC 0.7156)   │ ───────► │ Model: "fraud-detection-baseline"       │
│ Run 2: XGBoost Bad (PR-AUC 0.6500)   │ Register │   └── Version 1                         │
│ Run 3: LightGBM Bad (PR-AUC 0.6800)  │          │         └── Alias: 🏷️ [production]       │
└──────────────────────────────────────┘          └─────────────────────────────────────────┘
```

---

### 🏛️ 5.1 Đóng gói & Quản lý vòng đời (2 File Scripts Cốt Lõi)

#### 1️⃣ File `scripts/register_model.py` — Người Tuyển Trạch Model

> [!tip] Nhiệm vụ chính
> 
> Tách biệt giữa **lúc train** và **lúc đăng ký**. Không tự động đăng ký mọi model rác trong lúc train. Chỉ chọn đúng Run ID tốt nhất (`b989f579...`) để đưa vào Registry và gán nhãn `production`.

- **Cú pháp ảo URI Scheme (`runs:/`)**:
  `model_uri = f"runs:/{BEST_RUN_ID}/model"` ➔ Chỉ định cho MLflow tự truy vấn vào SQLite database để tìm đúng thư mục lưu trữ trọng số của Run ID đó.
  
- **Cơ chế `mlflow.register_model()`**:
  Đưa Model vào Registry. Nếu chưa tồn tại tên `fraud-detection-baseline`, nó tạo mới **Version 1**. Nếu đã có, nó tự tăng lên **Version 2, 3...**

- **Sức mạnh của Aliases (`client.set_registered_model_alias`)**:
  > [!important] Tại sao dùng Alias `production` thay vì Stage cũ?
  > 
  > - **Cách làm cũ (Stages)**: Gán cứng vào `Staging` / `Production`. Bị gượng ép và bị đánh dấu lỗi thời từ MLflow 2.0+.
  > - **Cách làm mới (Aliases)**: Gán chiếc thẻ tên tự do `production` vào Version 1. 
  > - **Lợi ích**: Code API Serving chỉ cần gọi nhãn `@production`. Khi tháng sau ta train được Version 2 ngon hơn, ta chỉ cần chuyển thẻ nhãn `production` sang Version 2, API Server sẽ tự động ăn theo **mà không cần sửa 1 dòng code API hay restart server!**

---

#### 2️⃣ File `scripts/export_model.py` — Người Xuất Bản & Đóng Gói Cloud

> [!question] Tại sao phải Export ra file Pickle (`.pkl`) khi đã có MLflow Registry?
> 
> Các máy chủ Web Serving (Render, Kubernetes Cluster) hoàn toàn không có quyền truy cập vào file database `mlflow.db` local của bạn vì lý do bảo mật và hạ tầng. Do đó, ta cần đóng gói model thành file tĩnh `models/baseline_lr.pkl` (~1.5 KB) để mang đi deploy.

- **Cú pháp nạp theo Alias (`models:/`)**:
  `model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"` ➔ Kết nối vào Registry lấy đúng model đang mang nhãn `production` mà không cần quan tâm nó là Version mấy.
  
- **Cơ chế Đa Môi Trường (Local Export ➔ Hugging Face Hub)**:
  - **Local Export**: Nạp model từ RAM và dùng `joblib.dump()` nén thành file `models/baseline_lr.pkl` nhẹ gọn trên ổ cứng.
  - **Cloud Upload (`--upload`)**: Sử dụng thư viện `huggingface_hub` (gồm `HfApi`) và mã `HF_TOKEN` (quyền `Write`) để đẩy file `.pkl` cùng trang tài liệu **Model Card (`README.md`)** lên Hugging Face Model Repository hoàn toàn tự động.

---

### ⚡ 5.2 Chiến lược Nạp Model 3 Cấp (3-Priority Loading Strategy) trong `src/api.py`

Khi ứng dụng FastAPI khởi động trong hàm `lifespan(app)`, hàm `load_model()` sẽ kiểm tra theo 3 nấc ưu tiên giúp code chạy mượt trên **mọi môi trường mà không bị lỗi**:

```python
def load_model():
    # Nấc 1: Đọc file .pkl local (Môi trường Docker Container / Local Test)
    if os.environ.get("MODEL_PATH"):
        return _load_from_local(...)
    
    # Nấc 2: Tải từ Hugging Face Hub (Môi trường Render Cloud Service)
    if os.environ.get("HF_REPO_ID"):
        return _load_from_hf_hub(...)
    
    # Nấc 3 (Fallback): Nạp từ MLflow Registry Local (Môi trường Data Scientist Dev)
    return _load_from_mlflow()
```

> [!success] Tóm tắt tư duy MLOps
> 
> 1. **Code độc lập với Môi trường**: Nhờ biến môi trường (`os.environ`), code Python không bao giờ bị hardcode đường dẫn.
> 2. **Model dán nhãn động**: Nhờ MLflow Alias `production`, việc cập nhật model mới chỉ là thao tác gán lại nhãn.
> 3. **Tự động hóa hoàn toàn**: Từ train ➔ log metrics ➔ register ➔ export ➔ upload cloud ➔ serving đều thực thi bằng câu lệnh CLI ngắn gọn.