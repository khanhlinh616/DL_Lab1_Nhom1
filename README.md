# FashionMNIST Deep Learning Classification

Dự án phân loại ảnh thời trang 10 lớp sử dụng **PyTorch**, tuân thủ nghiêm ngặt theo quy trình phát triển **Machine Learning / Deep Learning chuẩn 21 bước** với tính module hóa cao, cấu hình tập trung (config-driven), khả năng tái lập (reproducibility) và kiểm chứng thực nghiệm đầy đủ.

---

## 1. Mục Tiêu và Bài Toán

- **Bài toán:** Phân loại ảnh thời trang đơn sắc (grayscale $28 \times 28$ pixel) thành 10 lớp thời trang.
- **Tập dữ liệu:** FashionMNIST (Zalando Research) gồm **60,000 ảnh huấn luyện** và **10,000 ảnh kiểm thử**.
- **10 lớp trang phục:** `T-shirt/top` (Áo phông), `Trouser` (Quần), `Pullover` (Áo len chui đầu), `Dress` (Váy), `Coat` (Áo khoác), `Sandal` (Dép xăng-đan), `Shirt` (Áo sơ mi), `Sneaker` (Giày thể thao), `Bag` (Túi xách), `Ankle boot` (Bốt cổ thấp).
- **Phương pháp tiếp cận:** Đối chứng khoa học giữa mô hình cơ sở nhiều tham số (`BaselineMLP` - 203.5K params) và mô hình tích chập tối ưu (`SimpleCNN` - 38.1K params).

---

## 2. Kiến Trúc Dự Án Chi Tiết (Project Architecture)

Dự án áp dụng nguyên lý thiết kế **Separation of Concerns (Tách biệt mối quan tâm)**, chia tách toàn bộ hệ thống thành các tầng chức năng độc lập:

```text
fashionmnist-deep-learning/
├── configs/                  # TẦNG CẤU HÌNH (CONFIG-DRIVEN ARCHITECTURE)
│   ├── default.yaml          # Cấu hình gốc mặc định (seed, batch size, epochs, paths)
│   └── experiments.yaml      # Cấu hình ghi đè cho từng thí nghiệm (MLP, CNN, LR, Batch size)
│
├── src/fashionmnist/         # TẦNG MÃ NGUỒN CHÍNH (CORE LIBRARY PACKAGE)
│   ├── config.py             # Parser đọc YAML, validate kiểu dữ liệu và merge cấu hình thí nghiệm
│   ├── data/                 # Phân hệ dữ liệu & tiền xử lý
│   │   ├── dataset.py        # Tải FashionMNIST và kiểm tra tính toàn vẹn (integrity check)
│   │   ├── eda.py            # Phân tích khám phá dữ liệu (EDA) độc lập trước khi chia tập
│   │   ├── transforms.py     # Pipeline chuẩn hóa Z-score và data augmentation (RandomFlip)
│   │   └── split.py          # Chia tập phân tầng (Stratified split) chống rò rỉ dữ liệu (leak-free)
│   ├── models/               # Phân hệ mạng nơ-ron sâu
│   │   └── network.py        # Định nghĩa BaselineMLP, SimpleCNN và Factory tạo mô hình
│   ├── training/             # Phân hệ vòng lặp huấn luyện & tối ưu
│   │   ├── trainer.py        # Vòng lặp train/val epoch, early stopping, Cosine LR scheduler
│   │   └── checkpoint.py     # Quản lý lưu trữ/khôi phục state_dict, optimizer và best checkpoint
│   ├── evaluation/           # Phân hệ đánh giá & chẩn đoán lỗi
│   │   ├── metrics.py        # Tính Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix
│   │   └── error_analysis.py # Trích xuất và phân tích các cặp lớp nhầm lẫn nhiều nhất
│   ├── visualization/        # Phân hệ trực quan hóa dữ liệu & đồ thị
│   │   └── plots.py          # Vẽ biểu đồ EDA, đường cong học tập, ma trận nhầm lẫn, sample grid
│   └── utils/                # Phân hệ tiện ích hệ thống
│       ├── reproducibility.py# Cố định random seed (Python, NumPy, PyTorch CPU/CUDA)
│       └── logging.py        # Ghi nhận và kết xuất nhật ký thí nghiệm chuẩn JSON
│
├── scripts/                  # TẦNG KỊCH BẢN ĐẦU CUỐI (CLI ENTRYPOINTS)
│   ├── download_and_eda.py   # Bước 1: Tải dữ liệu, kiểm tra toàn vẹn và xuất toàn bộ đồ thị EDA
│   ├── train.py              # Bước 2: Huấn luyện mô hình theo thí nghiệm chỉ định
│   ├── evaluate.py           # Bước 3: Đánh giá mô hình tốt nhất trên tập Test độc lập
│   ├── predict.py            # Bước 4: Chạy suy luận (inference) thực tế trên batch ảnh
│   └── generate_complete_notebook.py # Tự động tạo và nhúng đồ thị vào Jupyter Notebook
│
├── notebooks/                # TẦNG BÁO CÁO & TRỰC QUAN HÓA TOÀN DIỆN
│   └── fashionmnist_complete_pipeline.ipynb # Notebook hoàn chỉnh từ EDA đến Error Analysis
│
├── tests/                    # TẦNG KIỂM THỬ TỰ ĐỘNG (46 TEST CASES - PYTEST)
│   ├── test_dataset.py       # Kiểm thử shape tensor, dtype, dải nhãn, biến đổi transforms
│   ├── test_model.py         # Kiểm thử logits đầu ra, không chứa Softmax ngầm, số lượng tham số
│   ├── test_training.py      # Kiểm thử optimizer, gradient backward pass, chế độ train/eval
│   └── test_checkpoint.py    # Kiểm thử lưu và nạp lại checkpoint không làm mất mát trọng số
│
├── artifacts/                # TẦNG LƯU TRỮ SẢN PHẨM THỰC NGHIỆM
│   ├── checkpoints/          # Trọng số tối ưu best_model.pt cho từng thí nghiệm
│   ├── eda/                  # Thống kê phân phối JSON (mean, std, class distribution)
│   ├── experiments/          # Lịch sử training loss, accuracy, thời gian qua từng epoch (.json)
│   ├── figures/              # Toàn bộ hình ảnh biểu đồ PNG phân giải cao
│   ├── metrics/              # Báo cáo đánh giá classification_report.txt và test_metrics.json
│   └── predictions/          # Ảnh kết quả suy luận thực tế với xác suất dự đoán
│
├── docs/                     # TẦNG TÀI LIỆU KỸ THUẬT CHUẨN 21 BƯỚC (01 ĐẾN 08)
│   ├── 01_project_overview.md
│   ├── 02_dataset_and_eda.md
│   ├── 03_preprocessing_and_splitting.md
│   ├── 04_model_architecture.md
│   ├── 05_training_and_experiments.md
│   ├── 06_evaluation_and_error_analysis.md
│   ├── 07_reproducibility_and_limitations.md
│   └── 08_results_and_conclusion.md
│
├── pyproject.toml            # Cấu hình cài đặt package và môi trường dev
└── .gitignore                # Loại trừ file nhị phân thô và checkpoints trung gian
```

---

## 3. Giải Thích Chi Tiết Từng Phân Hệ

### 3.1 Tầng Cấu Hình (Configs)
- **Cơ chế Override:** Toàn bộ siêu tham số được quản lý qua `configs/default.yaml`. Khi người dùng chỉ định thí nghiệm cụ thể qua tham số `--experiment baseline_mlp` hoặc `--experiment exp_cnn_small`, hệ thống sẽ nạp cấu hình mặc định rồi tự động ghi đè các trường tương ứng từ `configs/experiments.yaml`. Không có bất kỳ giá trị số nào bị hardcode trong logic xử lý.

### 3.2 Tầng Dữ Liệu & Tiền Xử Lý (Data & Preprocessing)
1. **Kiểm tra tính toàn vẹn (Integrity Check):** Đảm bảo tập train có đủ 60,000 ảnh, test có đủ 10,000 ảnh, 100% nhãn nằm trong đoạn $[0, 9]$, không có giá trị NaN hay nhãn khuyết tật.
2. **EDA độc lập trước khi chia tập:** Tính toán các chỉ số thống kê cơ bản toàn tập ($\mu = 0.2860, \sigma = 0.3530$).
3. **Chia tập Stratified chống rò rỉ (Leak-Free Split):** Tách 60,000 ảnh ban đầu thành **50,000 ảnh Train** và **10,000 ảnh Validation** bằng phương pháp phân tầng (mỗi lớp có đúng 5,000 ảnh train và 1,000 ảnh val). Hàm `verify_no_overlap()` đảm bảo giao chỉ số giữa Train và Val là tập rỗng (`set(train_idx) ∩ set(val_idx) = ∅`).
4. **Data Transforms:** Áp dụng `ToTensor()`, chuẩn hóa Z-Score với $\mu=0.2860, \sigma=0.3530$, và tăng cường dữ liệu `RandomHorizontalFlip(p=0.5)` chỉ trên tập Train nhằm giúp mô hình học tính bất biến đối xứng gương của trang phục.

### 3.3 Tầng Kiến Trúc Mô Hình Nơ-ron (Model Architectures)
1. **`BaselineMLP` (Mô hình đối chứng):**
   $$\text{Input } (1 \times 28 \times 28) \xrightarrow{\text{Flatten}} 784 \xrightarrow{\text{Linear}} 256 \xrightarrow{\text{ReLU}} \text{Dropout} \xrightarrow{\text{Linear}} 10$$
   - Tổng tham số: **203,530**.
   - Hạn chế: Ép phẳng ảnh 2D thành vector 1D làm phá vỡ tương quan vị trí không gian của các điểm ảnh lân cận.
2. **`SimpleCNN` (Kiến trúc Deep Learning chính):**
   - **Khối Conv 1:** `Conv2D(1→32, k=3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2, 2)` (kích thước ảnh giảm $28 \times 28 \rightarrow 14 \times 14$).
   - **Khối Conv 2:** `Conv2D(32→64, k=3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2, 2)` (kích thước ảnh giảm $14 \times 14 \rightarrow 7 \times 7$).
   - **Khối Phân loại (Classifier):** `Flatten(64 * 7 * 7 = 3136)` $\rightarrow$ `Linear(3136, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, 10)`.
   - Tổng tham số: **38,122** (nhẹ hơn **5.3 lần** so với MLP).
   - Nguyên tắc ngõ ra: Xuất **Raw Logits** (không dùng hàm kích hoạt Softmax ở lớp cuối) nhằm tương thích tối ưu với `nn.CrossEntropyLoss()` của PyTorch, giúp tránh hiện tượng tràn số và ổn định đạo hàm.

### 3.4 Tầng Huấn Luyện & Checkpoint (Training & Checkpoints)
- **Tách biệt vòng lặp:** `train_one_epoch()` thực hiện lan truyền xuôi, tính loss, lan truyền ngược `loss.backward()` và cập nhật trọng số `optimizer.step()`. Trong khi đó, `evaluate_one_epoch()` bọc trong `@torch.no_grad()` và `model.eval()`.
- **Lịch trình học (LR Scheduler):** Tích hợp `CosineAnnealingLR` giúp giảm dần tốc độ học mềm mại về cuối chu kỳ, hỗ trợ mô hình tìm kiếm điểm cực tiểu tốt hơn.
- **Lưu Checkpoint an toàn:** Lưu riêng `best_model.pt` mỗi khi Validation Accuracy vượt kỷ lục cũ, kèm theo các checkpoint định kỳ 5 epoch/lần.

### 3.5 Tầng Trực Quan Hóa (Visualization)
Sử dụng 2 thư viện chuẩn mực:
- **`matplotlib`:** Vẽ phổ cường độ pixel, lưới ảnh mẫu 4×10, đường cong học tập train/val loss và accuracy, biểu đồ cột F1-Score từng lớp và lưới mẫu phân tích ca đoán sai (Error Analysis Grid).
- **`seaborn`:** Vẽ Heatmap ma trận nhầm lẫn (Confusion Matrix) chuẩn hóa tỉ lệ %.

---

## 4. Bảng Kết Quả Thực Nghiệm Đối Chứng

Mô hình tốt nhất được lựa chọn dựa trên **Validation Accuracy** (chọn mô hình trước khi cho đánh giá trên Test Set):

| Experiment | Kiến Trúc (Architecture) | Số Tham Số | Val Accuracy | Test Accuracy | Macro F1 | Weighted F1 | Thời Gian Huấn Luyện |
|---|---|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 203,530 | 82.16% | 82.07% | 0.8204 | 0.8204 | 192.5s |
| `exp_cnn_small` | **SimpleCNN** | **38,122** | **84.83%** | **84.75%** | **0.8468** | **0.8468** | 829.9s |

### Nhận xét cốt lõi:
1. **Sức mạnh của tích chập:** SimpleCNN đạt độ chính xác cao hơn rõ rệt (**+2.68% Test Accuracy**, **+2.64% F1-score**) dù số lượng trọng số nhẹ hơn **5.3 lần** so với MLP.
2. **Cải thiện trên các lớp khó:** Lớp `Pullover` (Áo len) tăng vọt F1-score từ **68.44% lên 79.20% (+10.76%)**; `Sneaker` đạt 93.04%; `Sandal` đạt 94.42%.
3. **Giảm thiểu lỗi:** Tổng số mẫu phân loại sai trên toàn bộ 10,000 ảnh kiểm thử giảm từ **1,793 ca xuống 1,525 ca** (giảm 268 trường hợp phân loại sai).

---

## 5. Phân Tích Sai Số Chuyên Sâu (Error Analysis)

Dựa trên ma trận nhầm lẫn của `SimpleCNN`, 5 cặp lớp nhầm lẫn nhiều nhất gồm:
1. **`Shirt` nhầm thành `T-shirt/top` (176 ca):** Cả 2 đều có dáng thân trên tương đồng; ở độ phân giải 28×28 grayscale, các chi tiết hàng cúc và cổ áo sơ mi bị tiêu biến.
2. **`Coat` nhầm thành `Shirt` (124 ca):** Áo khoác mở vạt và áo sơ mi có viền cổ chữ V tương tự nhau.
3. **`Pullover` nhầm thành `Shirt` (106 ca):** Áo len và áo sơ mi cùng có chiều dài tay và thân áo giống nhau.
4. **`Shirt` nhầm thành `Coat` (105 ca):** Hình bóng trang phục ngoài tương đồng khi không có thông tin màu sắc và độ dày vải.
5. **`Coat` nhầm thành `Pullover` (105 ca):** Tương tự, thiếu độ tương phản chất liệu dạ/len.

---

## 6. Hướng Dẫn Cài Đặt và Thực Thi

### 6.1 Cài đặt môi trường
```bash
# Cài đặt package ở chế độ editable
py -3 -m pip install -e . --no-deps
py -3 -m pip install pytest nbformat
```

### 6.2 Chạy toàn bộ quy trình từ dòng lệnh (CLI)
```bash
# Bước 1: Tải dữ liệu, kiểm tra toàn vẹn và trích xuất EDA
py -3 scripts/download_and_eda.py

# Bước 2: Huấn luyện mô hình (chọn baseline_mlp hoặc exp_cnn_small)
py -3 scripts/train.py --experiment baseline_mlp
py -3 scripts/train.py --experiment exp_cnn_small

# Bước 3: Đánh giá mô hình trên tập Test độc lập
py -3 scripts/evaluate.py --experiment exp_cnn_small

# Bước 4: Chạy suy luận mẫu thực tế
py -3 scripts/predict.py --experiment exp_cnn_small --n_samples 8

# Chạy kiểm thử tự động toàn hệ thống
py -3 -m pytest
```

### 6.3 Xem báo cáo trực quan hoàn chỉnh
Mở file Jupyter Notebook tại:
`notebooks/fashionmnist_complete_pipeline.ipynb`
File notebook này đã được nhúng sẵn toàn bộ biểu đồ EDA, đường cong học tập, ma trận nhầm lẫn, phân tích lỗi và giải thích chuyên sâu từng bước.
