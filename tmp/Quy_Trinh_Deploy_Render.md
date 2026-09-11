# QUY TRÌNH TRIỂN KHAI HOÀN CHỈNH: HUGGING FACE MODEL HUB & RENDER BLUEPRINT

Tài liệu này hợp nhất toàn bộ quy trình triển khai Chặng 1: Từ việc xuất và lưu trữ model trên Hugging Face Model Hub, đến việc tự động hóa triển khai dịch vụ FastAPI lên Render thông qua Blueprint (`render.yaml`), cùng giải thích chi tiết về cơ chế mạng và định tuyến endpoint.

---

## 1. TỔNG QUAN KIẾN TRÚC & PHÂN TÁCH TRÁCH NHIỆM

```mermaid
flowchart LR
    subgraph GitHub["1. GitHub Repository"]
        Code["Source Code Python"]
        DF["Dockerfile"]
        RYaml["render.yaml (Blueprint)"]
    end

    subgraph HF["2. Hugging Face Hub"]
        Model["baseline_lr.pkl (Trọng số model)"]
        MCard["README.md (Model Card tự động)"]
    end

    subgraph Render["3. Render Cloud Platform"]
        Builder["Docker Engine (Tự build image)"]
        Container["Container API (Port 10000)"]
    end

    GitHub -->|1. Webhook Auto-Deploy| Builder --> Container
    Container -.->|2. Khởi động: Tải model về RAM| HF
    Client["Client / Trình duyệt"] -->|3. Gọi HTTPS (Định tuyến mọi endpoint)| Container
```

### Điểm cốt tử giải phóng sự phức tạp:
1. **Model tách rời khỏi Code:** Tệp nhị phân `baseline_lr.pkl` bị `.gitignore` trên GitHub, được đưa lên Hugging Face Model Hub để làm kho lưu trữ tĩnh miễn phí.
2. **Tại sao chưa cần điền Access Token của Hugging Face vào Render?**  
   Vì kho model `votrananhquan/fraud-detection-model` được tạo ở chế độ **Public (công khai)**. Bất kỳ ai trên Internet (bao gồm cả server Render) đều có thể tải file `baseline_lr.pkl` về mà **không cần mật khẩu hay token xác thực**. Chỉ khi nào bạn để repo ở chế độ **Private (bí mật)** thì mới cần điền `HF_TOKEN` vào Render.
3. **Tại sao Render cũng không cần cấu hình kết nối gì phức tạp?**  
   Render kết nối trực tiếp với GitHub thông qua **GitHub App** (chỉ cần cấp quyền 1 lần lúc bấm *Configure account*). Kể từ đó, mỗi khi bạn `git push`, GitHub tự động báo cho Render kéo code về build lại (Auto-Deploy).

---

## 2. PHẦN 1: XUẤT VÀ TẢI MODEL LÊN HUGGING FACE HUB

### Bước 1: Xác thực máy tính cá nhân (1 lần duy nhất)
Chạy lệnh xác thực qua trình duyệt (OAuth Device Flow):
```powershell
uv run python -c "from huggingface_hub import login; login()"
```
Trình duyệt mở ra -> Bấm **Authorize** -> Hugging Face tự động sinh ra một OAuth Token (`hf_oauth_...`) và lưu vĩnh viễn vào `C:\Users\Admin\.cache\huggingface\token`. Toàn bộ các project trên máy bạn từ nay về sau tự động nhận diện tài khoản `votrananhquan`.

### Bước 2: Chạy script xuất model từ MLflow và tải lên Hub
Đoạn code trong [scripts/export_model.py](file:///D:/Documents/Project/mlops-fraud-detection/scripts/export_model.py) thực hiện trọn gói:
1. Đọc model Champion từ MLflow Registry cục bộ và lưu ra `models/baseline_lr.pkl`.
2. Tự động đọc chỉ số PR-AUC, Recall, Precision từ MLflow run để sinh nội dung Markdown cho **Model Card**.
3. Gọi `huggingface_hub.HfApi` tự động tạo repo (nếu chưa có) và upload cả 2 tệp: `baseline_lr.pkl` và `README.md`.

```powershell
# Chạy trong PowerShell tại thư mục dự án
$env:HF_REPO_ID = "votrananhquan/fraud-detection-model"
$env:PYTHONIOENCODING = "utf-8"
uv run python scripts/export_model.py --upload
```

Kiểm tra tại: `https://huggingface.co/votrananhquan/fraud-detection-model` thấy đầy đủ file trọng số và trang Model Card giới thiệu.

---

## 3. PHẦN 2: TRIỂN KHAI FASTAPI LÊN RENDER QUA BLUEPRINT

### Bước 1: Tệp cấu hình Blueprint `render.yaml`
Nằm sẵn tại thư mục gốc dự án, đóng vai trò là "Bản thiết kế hạ tầng" (IaC):

```yaml
services:
  - type: web
    name: fraud-detection-api
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: free
    healthCheckPath: /health
    envVars:
      - key: HF_REPO_ID
        value: votrananhquan/fraud-detection-model
```

### Bước 2: Thao tác trên Dashboard Render
1. Truy cập [dashboard.render.com](https://dashboard.render.com) -> Bấm **New +** -> Chọn **Blueprint**.
2. Kết nối repo: `anhquan1111/mlops-fraud-detection`.
3. Điền thông tin:
   - **Blueprint Name:** `mlops-fraud-detection` (Tên dự án hiển thị trên Dashboard).
   - **Branch:** `master`.
   - **Blueprint Path:** `render.yaml`.
4. Bấm nút **Apply**: Render tự động đọc file YAML, khởi động Docker Builder, cài đặt môi trường, kéo model từ Hugging Face về RAM, và hoàn tất kiểm tra `/health`.

---

## 4. PHẦN 3: ĐỊNH TUYẾN MẠNG — TẠI SAO 1 LINK RENDER LẠI CUNG CẤP ĐƯỢC CẢ `/docs`, `/health`, `/predict`?

### Bản chất: Tên miền cơ sở (Base Domain) vs. Đường dẫn con (Routes)

Nhiều người lầm tưởng: *"Render cấp 1 link nghĩa là chỉ gọi được đúng 1 trang web duy nhất"*. Thực tế hoàn toàn ngược lại:

```
                  CƠ CHẾ ĐỊNH TUYẾN CỦA RENDER & FASTAPI

  Client gõ trên Trình duyệt
  https://fraud-detection-api.onrender.com/docs
                    |
                    v
  [ Render Reverse Proxy (HTTPS Edge) ]
  Nhìn thấy domain `fraud-detection-api.onrender.com`
  -> Chuyển tiếp toàn bộ request (kèm đường dẫn `/docs`) vào Container
                    |
                    v
  [ Docker Container: FastAPI App ]
  Bộ định tuyến (Router) của FastAPI nhận request và phân luồng:
  
  +--- Nếu đường dẫn là `/health`   --> Trả về JSON trạng thái sống ({status: healthy})
  +--- Nếu đường dẫn là `/docs`     --> Trả về Giao diện tương tác Swagger UI
  +--- Nếu đường dẫn là `/predict`  --> Gọi hàm dự đoán gian lận thẻ tín dụng
  +--- Nếu đường dẫn là `/metrics`  --> Trả về dữ liệu cho Prometheus cào
```

Render chỉ làm nhiệm vụ **cấp chiếc cổng nhà (Tên miền chính)**:
👉 `https://fraud-detection-api.onrender.com`

Còn bên trong nhà có bao nhiêu phòng (bao nhiêu endpoint con như `/docs`, `/health`, `/predict`, `/metrics`), thì **bộ định tuyến của FastAPI trong `src/api.py` toàn quyền xử lý**:

1. **`GET /health`:** Kiểm tra container còn sống và model đã nạp vào RAM chưa.
2. **`GET /docs`:** Xem Swagger UI trực quan, bấm chạy thử nghiệm dự đoán ngay trên web.
3. **`POST /predict`:** Nhận dữ liệu giao dịch 29 đặc trưng và trả về xác suất gian lận.
4. **`GET /metrics`:** Cung cấp số liệu thời gian thực cho Prometheus cào dữ liệu từ xa.
