# Đồ Án Deep Learning: Phân Loại Ảnh Thời Trang FashionMNIST Bằng PyTorch

Dự án phân loại hình ảnh thời trang gồm 10 lớp sử dụng PyTorch. Mã nguồn được chia thành các module độc lập, siêu tham số quản lý qua tệp YAML, seed cố định để tái lập kết quả, và có 46 bài kiểm thử tự động bằng Pytest.

---

## 1. Mục Tiêu và Tổng Quan Bài Toán

- **Bài toán:** Phân loại ảnh chữ nhật đơn sắc (grayscale $28 \times 28$ pixel) vào 10 danh mục trang phục thời trang.
- **Tập dữ liệu:** FashionMNIST (Zalando Research) gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm thử.
- **Danh sách 10 lớp:**
  0. T-shirt/top (Áo phông)
  1. Trouser (Quần dài)
  2. Pullover (Áo len chui đầu)
  3. Dress (Váy liền)
  4. Coat (Áo khoác ngoài)
  5. Sandal (Dép xăng-đan)
  6. Shirt (Áo sơ mi)
  7. Sneaker (Giày thể thao)
  8. Bag (Túi xách)
  9. Ankle boot (Ủng / Bốt cổ thấp)
- **Phương pháp:** So sánh hai kiến trúc mạng nơ-ron:
  - BaselineMLP (mạng kết nối đầy đủ, 203.530 tham số) làm mốc tham chiếu.
  - SimpleCNN (mạng tích chập, 38.122 tham số) là mô hình chính.

---

## 2. Kiến Trúc Dự Án và Giải Thích Chi Tiết Từng Tệp

Cấu trúc thư mục:

```text
fashionmnist-deep-learning/
├── configs/                  # Quản lý siêu tham số và thí nghiệm
│   ├── default.yaml          # Cấu hình gốc (mặc định)
│   └── experiments.yaml      # Cấu hình ghi đè cho từng thí nghiệm (MLP, CNN, LR, Batch size)
├── src/fashionmnist/         # Core Library (gói mã nguồn chính)
│   ├── config.py             # Đọc và validate cấu hình YAML
│   ├── data/                 # Module xử lý dữ liệu và tiền xử lý
│   │   ├── dataset.py        # Tải FashionMNIST và kiểm tra tính toàn vẹn
│   │   ├── eda.py            # Phân tích khám phá dữ liệu (EDA) trước khi chia tập
│   │   ├── transforms.py     # Pipeline chuẩn hóa (ToTensor, Normalize, Augmentation)
│   │   └── split.py          # Stratified split và khởi tạo DataLoader
│   ├── models/               # Định nghĩa các kiến trúc mạng nơ-ron
│   │   └── network.py        # BaselineMLP, SimpleCNN và Model Factory
│   ├── training/             # Logic huấn luyện mô hình
│   │   ├── trainer.py        # Vòng lặp train/val epoch, early stopping, LR scheduler
│   │   └── checkpoint.py     # Lưu/load state_dict mô hình, optimizer, epoch tốt nhất
│   ├── evaluation/           # Đánh giá và phân tích lỗi
│   │   ├── metrics.py        # Tính Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix
│   │   └── errors.py         # Trích xuất và phân tích các cặp lớp nhầm lẫn nhiều nhất
│   ├── visualization/        # Vẽ biểu đồ trực quan hóa
│   │   └── plots.py          # Biểu đồ EDA, Confusion Matrix, Learning curves, Sample grid
│   └── utils/                # Tiện ích bổ trợ
│       ├── reproducibility.py# Cố định random seed và phát hiện thiết bị tính toán (CPU/CUDA)
│       └── logging.py        # Ghi nhật ký thí nghiệm dạng JSON
├── scripts/                  # CLI Entrypoints (các script thực thi đầu cuối)
│   ├── download_and_eda.py   # Tải dữ liệu, kiểm tra dữ liệu và chạy EDA
│   ├── train.py              # Huấn luyện mô hình theo thí nghiệm chỉ định
│   ├── evaluate.py           # Đánh giá mô hình tốt nhất trên tập Test độc lập
│   └── predict.py            # Chạy suy luận (inference) trên ảnh thực tế
├── tests/                    # 46 kiểm thử tự động với Pytest
│   ├── test_dataset.py       # Kiểm thử shape, dtype, dải nhãn, transforms, leak-free split
│   ├── test_model.py         # Kiểm thử logits đầu ra, số lượng tham số
│   ├── test_training.py      # Kiểm thử optimizer, gradient update, chế độ train/eval
│   └── test_checkpoint.py    # Kiểm thử lưu/khôi phục checkpoint không làm mất mát trọng số
├── artifacts/                # Nơi lưu trữ toàn bộ kết quả thực nghiệm
│   ├── checkpoints/          # Các file trọng số mô hình tốt nhất (.pt)
│   ├── eda/                  # Thống kê phân phối dữ liệu (.json)
│   ├── experiments/          # Lịch sử và thông số các lần huấn luyện (.json)
│   ├── figures/              # Toàn bộ hình ảnh biểu đồ phân tích (.png)
│   ├── metrics/              # Báo cáo đánh giá chi tiết (.txt, .json)
│   └── predictions/          # Ảnh kết quả suy luận thực tế (.png)
└── docs/                     # 8 tài liệu kỹ thuật (từ 01 đến 08)
```

### 2.1 Thư mục Cấu hình (configs/)

- `configs/default.yaml`: Chứa toàn bộ các thông số cấu hình mặc định cho dự án, bao gồm: giá trị random seed (42), tỷ lệ tách tập validation (0.1667), kích thước batch size (128), thiết bị tính toán (tự động phát hiện CUDA/CPU), cấu hình bộ tối ưu hóa Adam, tốc độ học ban đầu (0.001), hệ số suy giảm trọng số (weight decay = 0.0001), số lượng epoch tối đa và đường dẫn thư mục lưu trữ dữ liệu, kết quả.
- `configs/experiments.yaml`: Định nghĩa cấu hình ghi đè cho từng kịch bản thực nghiệm cụ thể (ví dụ: `baseline_mlp`, `exp_cnn_small`, `exp_lr_01`, `exp_batch_256`). Khi gọi một kịch bản, hệ thống sẽ tự động hòa trộn cấu hình mặc định với các thông số ghi đè của kịch bản đó.

### 2.2 Thư mục Mã Nguồn Lõi (src/fashionmnist/)

- `src/fashionmnist/__init__.py`: Tệp khởi tạo package Python `fashionmnist`.
- `src/fashionmnist/config.py`: Tải, xác thực và kết hợp các tệp cấu hình YAML thành một dictionary dùng cho toàn bộ pipeline.

#### Phân hệ Dữ liệu (src/fashionmnist/data/)
- `src/fashionmnist/data/__init__.py`: Khởi tạo module xử lý dữ liệu.
- `src/fashionmnist/data/dataset.py`: Tải FashionMNIST từ `torchvision.datasets`. Hàm `verify_dataset_integrity()` kiểm tra số lượng ảnh (60.000 train, 10.000 test), dải nhãn $[0, 9]$ và phát hiện nhãn bất thường.
- `src/fashionmnist/data/eda.py`: Hàm `run_eda()` phân tích dữ liệu thô trước khi chia tập: tính trung bình pixel ($\mu \approx 0.2860$), độ lệch chuẩn ($\sigma \approx 0.3530$), phân phối mẫu mỗi lớp, và xuất các đồ thị phân tích.
- `src/fashionmnist/data/transforms.py`: Định nghĩa `get_train_transform()` và `get_eval_transform()`. Train: `RandomHorizontalFlip(p=0.5)` + `RandomCrop(28, padding=4)` + `ToTensor()` + `Normalize(0.2860, 0.3530)`. Eval: `ToTensor()` + `Normalize`.
- `src/fashionmnist/data/split.py`: Chia 60.000 ảnh train thành 50.000 Train + 10.000 Validation theo stratified split (giữ tỷ lệ lớp). `verify_no_overlap()` kiểm tra giao giữa hai tập bằng rỗng. Sau đó khởi tạo DataLoader.

#### Phân hệ Mô hình (src/fashionmnist/models/)
- `src/fashionmnist/models/__init__.py`: Khởi tạo module kiến trúc mạng nơ-ron.
- `src/fashionmnist/models/network.py`: 
  - `BaselineMLP`: Flatten(784) → Linear(784, 256) → ReLU → Dropout → Linear(256, 10). Tổng cộng 203.530 tham số.
  - `SimpleCNN`: 2 khối Conv (1→32→64, kernel 3×3, padding 1, BatchNorm, ReLU, MaxPool 2×2) + GAP + Linear(64, 256) → ReLU → Dropout(0.3) → Linear(256, 10). Tổng cộng 38.122 tham số.
  - `count_parameters()`: Đếm tổng số tham số trainable.
  - `build_model()`: Factory khởi tạo mô hình từ cấu hình.
  - Cả hai mô hình xuất raw logits (không Softmax ở lớp cuối) vì `nn.CrossEntropyLoss()` đã tích hợp log-softmax bên trong.

#### Phân hệ Huấn luyện (src/fashionmnist/training/)
- `src/fashionmnist/training/__init__.py`: Khởi tạo module huấn luyện.
- `src/fashionmnist/training/trainer.py`:
  - `build_optimizer()`: Khởi tạo Adam hoặc SGD với `weight_decay`.
  - `build_scheduler()`: Khởi tạo `CosineAnnealingLR` hoặc `StepLR`.
  - `train_one_epoch()`: Forward → `CrossEntropyLoss` → `loss.backward()` → `optimizer.step()`. Trả về loss trung bình và accuracy.
  - `evaluate_one_epoch()`: Chạy dưới `@torch.no_grad()` và `model.eval()`, trả về val loss và accuracy.
  - `train()`: Vòng lặp huấn luyện chính với early stopping và lưu checkpoint định kỳ.
- `src/fashionmnist/training/checkpoint.py`: Lưu `model_state_dict`, `optimizer_state_dict`, epoch, best metric vào `best_model.pt`. Hàm `load_checkpoint()` nạp lại trạng thái cho inference hoặc tiếp tục huấn luyện.

#### Phân hệ Đánh giá (src/fashionmnist/evaluation/)
- `src/fashionmnist/evaluation/__init__.py`: Khởi tạo module đánh giá.
- `src/fashionmnist/evaluation/metrics.py`: Tính Accuracy, Macro/Weighted F1, Precision, Recall và confusion matrix.
- `src/fashionmnist/evaluation/error_analysis.py`: Tìm các mẫu phân loại sai, xếp hạng cặp lớp nhầm lẫn nhiều nhất, và xuất lưới ảnh lỗi kèm nhãn thật/dự đoán.

#### Phân hệ Trực quan hóa (src/fashionmnist/visualization/)
- `src/fashionmnist/visualization/__init__.py`: Khởi tạo module đồ thị.
- `src/fashionmnist/visualization/plots.py`: Các hàm vẽ đồ thị (`matplotlib` + `seaborn`):
  - `plot_training_history()`: Đường cong Loss và Accuracy qua các epoch.
  - `plot_confusion_matrix()`: Heatmap ma trận nhầm lẫn (chuẩn hóa hoặc số lượng).
  - `plot_per_class_f1()`: Biểu đồ F1-Score từng lớp.
  - `plot_experiment_comparison()`: So sánh kết quả giữa các thí nghiệm.

#### Phân hệ Tiện ích (src/fashionmnist/utils/)
- `src/fashionmnist/utils/__init__.py`: Khởi tạo module tiện ích.
- `src/fashionmnist/utils/reproducibility.py`: `set_seed()` cố định seed cho `random`, `numpy`, `torch`, `torch.cuda` và bật `cudnn.deterministic`. `get_device()` tự động chọn GPU nếu có.
- `src/fashionmnist/utils/logging.py`: Ghi lịch sử huấn luyện, siêu tham số và kết quả đánh giá vào JSON (`artifacts/experiments/`).

---

### 2.3 Thư mục Kịch Bản Thực Thi (scripts/)

- `scripts/download_and_eda.py`: Tải tập dữ liệu về `data/raw/`, kiểm tra tính toàn vẹn và tạo các biểu đồ EDA vào `artifacts/figures/`.
- `scripts/train.py`: Nhận `--experiment <tên>`, xây dựng mô hình, huấn luyện, lưu learning curve và `best_model.pt`.
- `scripts/evaluate.py`: Nạp `best_model.pt`, đánh giá trên 10.000 mẫu test, tạo `classification_report.txt`, confusion matrix và biểu đồ phân tích lỗi.
- `scripts/predict.py`: Tải checkpoint, chạy suy luận trên ảnh ngẫu nhiên từ tập test, lưu kết quả vào `artifacts/predictions/`.
- `scripts/generate_complete_notebook.py`: Tạo file `notebooks/fashionmnist_complete_pipeline.ipynb` với mã nguồn, biểu đồ nhúng sẵn và giải thích.

---

### 2.4 Thư mục Báo Cáo Thực Nghiệm (notebooks/)

- `notebooks/fashionmnist_complete_pipeline.ipynb`: Notebook tổng hợp gồm EDA, so sánh kiến trúc, learning curves, confusion matrix, phân tích lỗi, kèm hình ảnh nhúng sẵn và giải thích từng phần.

---

### 2.5 Thư mục Kiểm Thử Tự Động (tests/)

46 bài kiểm thử tự động (`pytest`):
- `tests/test_dataset.py` (17 test): Shape ảnh, dtype, dải nhãn, transforms, kích thước split, giao tập train/val bằng rỗng.
- `tests/test_model.py` (11 test): Output logits shape $[B, 10]$, không có Softmax ngầm, giá trị hữu hạn, số tham số.
- `tests/test_training.py` (11 test): Optimizer Adam/SGD, Cosine scheduler, trọng số cập nhật sau backward, chuyển đổi train/eval mode.
- `tests/test_checkpoint.py` (7 test): Lưu/nạp `state_dict` đúng, optimizer state, checkpoint chỉ cập nhật khi val accuracy cải thiện.

---

### 2.6 Thư mục Sản Phẩm Thực Nghiệm (artifacts/)

- `artifacts/checkpoints/`: Chứa các tệp trọng số tối ưu `best_model.pt` của các mô hình đã huấn luyện.
- `artifacts/eda/`: Chứa tệp `eda_statistics.json` ghi lại các chỉ số thống kê phân phối của tập dữ liệu.
- `artifacts/experiments/`: Chứa các tệp nhật ký JSON ghi nhận chi tiết loss và accuracy của từng epoch huấn luyện.
- `artifacts/figures/`: Chứa toàn bộ các tệp biểu đồ dạng PNG phân giải cao (phân phối lớp, sample grid, histogram, ma trận nhầm lẫn, learning curve, per-class F1, lưới ảnh dự đoán sai).
- `artifacts/metrics/`: Chứa các báo cáo đánh giá định lượng: `classification_report_exp_cnn_small.txt` và `test_metrics_exp_cnn_small.json`.
- `artifacts/predictions/`: Chứa ảnh kết quả dự đoán thực tế `predictions_exp_cnn_small.png`.

---

### 2.7 Thư mục Tài Liệu Hệ Thống (docs/)

Gồm 8 tài liệu phân tích kỹ thuật:
- `docs/01_project_overview.md`: Tổng quan dự án, phạm vi bài toán và mục tiêu thực nghiệm.
- `docs/02_dataset_and_eda.md`: Nguồn dữ liệu, quy trình EDA và thống kê cường độ pixel.
- `docs/03_preprocessing_and_splitting.md`: Chi tiết phương pháp chuẩn hóa và chia tách tập dữ liệu không rò rỉ.
- `docs/04_model_architecture.md`: Chi tiết toán học và cấu trúc các lớp của BaselineMLP và SimpleCNN.
- `docs/05_training_and_experiments.md`: Chiến lược tối ưu hóa, hàm mất mát và siêu tham số huấn luyện.
- `docs/06_evaluation_and_error_analysis.md`: Phương pháp luận đánh giá và phân tích nguyên nhân sai số.
- `docs/07_reproducibility_and_limitations.md`: Báo cáo về tính tái lập, phiên bản thư viện và giới hạn kỹ thuật của hệ thống.
- `docs/08_results_and_conclusion.md`: Bảng tổng hợp số liệu thực nghiệm đối chứng và phương hướng phát triển.

---

### 2.8 Các Tệp Cấu Hình Gốc

- `pyproject.toml`: Khai báo package (PEP 517/518, `setuptools`), liệt kê dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pyyaml`, `tqdm`) và cấu hình `pytest`.
- `.gitignore`: Loại trừ `__pycache__`, `.venv`, `data/raw/` và checkpoint epoch trung gian.

---

## 3. Bảng Kết Quả Thực Nghiệm Đối Chứng

Mô hình được chọn theo **Validation Accuracy**. Test set chỉ dùng cho đánh giá cuối cùng:

| Kịch Bản (Experiment) | Kiến Trúc Mô Hình | Số Tham Số | Val Accuracy | Test Accuracy | Macro F1 | Weighted F1 | Thời Gian Chạy |
|---|---|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 203.530 | 82.16% | 82.07% | 0.8204 | 0.8204 | 192.5s |
| `exp_cnn_small` | **SimpleCNN** | **38.122** | **84.83%** | **84.75%** | **0.8468** | **0.8468** | 829.9s |

### Nhận xét:
1. **CNN vs MLP:** SimpleCNN đạt +2.68% Test Accuracy và +2.64% F1-score so với BaselineMLP, với số tham số nhỏ hơn 5.3 lần. Các lớp tích chập khai thác được đặc trưng không gian cục bộ mà MLP bỏ lỡ khi flatten ảnh thành vector.
2. **Cải thiện trên các lớp khó:** Pullover tăng F1 từ 68.44% lên 79.20% (+10.76%); Sneaker đạt 93.04%; Sandal đạt 94.42%.
3. **Số lỗi giảm:** Từ 1.793 (BaselineMLP) xuống 1.525 (SimpleCNN) trên 10.000 mẫu test, giảm 268 lỗi.

---

## 4. Phân Tích Sai Số (Error Analysis)

Dựa trên ma trận nhầm lẫn của mô hình `SimpleCNN`, 5 cặp lớp xảy ra nhầm lẫn nhiều nhất gồm:
1. `Shirt` nhầm thành `T-shirt/top` (176 mẫu): Cả hai đều có cấu trúc áo thân trên tương đồng. Khi chuyển về ảnh grayscale độ phân giải thấp $28 \times 28$, các chi tiết đặc trưng như cúc áo và đường xẻ cổ áo sơ mi bị tiêu biến.
2. `Coat` nhầm thành `Shirt` (124 mẫu): Áo khoác mở vạt trước tạo thành nếp viền cổ chữ V tương tự như áo sơ mi phanh ngực.
3. `Pullover` nhầm thành `Shirt` (106 mẫu): Áo len dài tay có chiều dài thân và tay áo tương đương áo sơ mi.
4. `Shirt` nhầm thành `Coat` (105 mẫu): Thiếu thông tin về độ dày của chất liệu vải và màu sắc khiến mô hình khó phân biệt dáng áo ngoài và áo mặc trong.
5. `Coat` nhầm thành `Pullover` (105 mẫu): Tương tự, thiếu độ tương phản bề mặt chất liệu dạ/len.

Ngược lại, các lớp có cấu trúc hình học bên ngoài đặc thù đạt độ chính xác rất cao: `Trouser` (F1: 97.05%), `Bag` (F1: 96.04%), `Sandal` (F1: 94.42%), `Ankle boot` (F1: 94.16%).

---

## 5. Hướng Dẫn Cài Đặt và Vận Hành

### 5.1 Cài đặt môi trường
```bash
# Cài đặt gói mã nguồn ở chế độ editable
py -3 -m pip install -e . --no-deps

# Cài đặt các thư viện kiểm thử và tạo notebook nếu cần
py -3 -m pip install pytest nbformat
```

### 5.2 Thực thi quy trình từ dòng lệnh
```bash
# Bước 1: Tải dữ liệu, kiểm tra toàn vẹn và tạo biểu đồ EDA
py -3 scripts/download_and_eda.py

# Bước 2: Huấn luyện mô hình theo kịch bản chỉ định
py -3 scripts/train.py --experiment baseline_mlp
py -3 scripts/train.py --experiment exp_cnn_small

# Bước 3: Đánh giá mô hình trên tập kiểm thử độc lập
py -3 scripts/evaluate.py --experiment exp_cnn_small

# Bước 4: Chạy thử nghiệm suy luận trên nhóm mẫu ngẫu nhiên
py -3 scripts/predict.py --experiment exp_cnn_small --n_samples 8

# Thực thi kiểm thử tự động toàn bộ 46 test cases
py -3 -m pytest
```

### 5.3 Báo cáo thực nghiệm
Báo cáo đầy đủ kèm hình ảnh và phân tích được lưu trữ trong tệp:
`notebooks/fashionmnist_complete_pipeline.ipynb`
