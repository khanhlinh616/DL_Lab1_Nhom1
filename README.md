# Đồ Án Deep Learning: Phân Loại Ảnh Thời Trang FashionMNIST Bằng PyTorch

Dự án nghiên cứu và triển khai hệ thống phân loại hình ảnh thời trang gồm 10 lớp sử dụng thư viện PyTorch. Dự án được tổ chức theo tiêu chuẩn kỹ nghệ phần mềm chuyên nghiệp, cấu trúc module hóa độc lập, quản lý siêu tham số tập trung qua tệp cấu hình, đảm bảo tính tái lập của thực nghiệm và có hệ thống kiểm thử tự động toàn diện.

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
- **Phương pháp thực nghiệm:** Thực hiện đối chứng khoa học giữa hai kiến trúc mạng nơ-ron:
  - Mạng kết nối đầy đủ cơ sở (BaselineMLP - 203.530 tham số).
  - Mạng tích chập tối ưu (SimpleCNN - 38.122 tham số).

---

## 2. Kiến Trúc Dự Án và Giải Thích Chi Tiết Từng Tệp

Toàn bộ dự án được phân cấp rõ ràng theo các tầng chức năng:

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
└── docs/                     # 8 tài liệu kỹ thuật chuẩn mực (từ 01 đến 08)
```

### 2.1 Thư mục Cấu hình (configs/)

- `configs/default.yaml`: Chứa toàn bộ các thông số cấu hình mặc định cho dự án, bao gồm: giá trị random seed (42), tỷ lệ tách tập validation (0.1667), kích thước batch size (128), thiết bị tính toán (tự động phát hiện CUDA/CPU), cấu hình bộ tối ưu hóa Adam, tốc độ học ban đầu (0.001), hệ số suy giảm trọng số (weight decay = 0.0001), số lượng epoch tối đa và đường dẫn thư mục lưu trữ dữ liệu, kết quả.
- `configs/experiments.yaml`: Định nghĩa cấu hình ghi đè cho từng kịch bản thực nghiệm cụ thể (ví dụ: `baseline_mlp`, `exp_cnn_small`, `exp_lr_01`, `exp_batch_256`). Khi gọi một kịch bản, hệ thống sẽ tự động hòa trộn cấu hình mặc định với các thông số ghi đè của kịch bản đó.

### 2.2 Thư mục Mã Nguồn Lõi (src/fashionmnist/)

- `src/fashionmnist/__init__.py`: Tệp khởi tạo package Python `fashionmnist`, định nghĩa phiên bản của gói (`__version__ = "1.0.0"`).
- `src/fashionmnist/config.py`: Đảm nhiệm việc tải, xác thực dữ liệu và kết hợp các tệp cấu hình YAML thành một dictionary hoàn chỉnh phục vụ toàn bộ pipeline.

#### Phân hệ Dữ liệu (src/fashionmnist/data/)
- `src/fashionmnist/data/__init__.py`: Khởi tạo module xử lý dữ liệu.
- `src/fashionmnist/data/dataset.py`: Tải tập dữ liệu FashionMNIST nguyên bản từ `torchvision.datasets` và thực hiện hàm `verify_dataset_integrity()` để kiểm tra độ toàn vẹn: xác nhận số lượng ảnh chính xác (60.000 train, 10.000 test), dải giá trị nhãn hợp lệ trong đoạn $[0, 9]$, không có nhãn bị lỗi hay giá trị NaN.
- `src/fashionmnist/data/eda.py`: Chứa hàm `run_eda()` thực hiện phân tích khám phá dữ liệu ngay trên dữ liệu thô trước khi chia tập. Hàm tính toán trung bình pixel ($\mu \approx 0.2860$), độ lệch chuẩn pixel ($\sigma \approx 0.3530$), phân phối số lượng mẫu mỗi lớp và kết xuất các đồ thị phân tích.
- `src/fashionmnist/data/transforms.py`: Định nghĩa các hàm tiền xử lý `get_train_transforms()` và `get_eval_transforms()`. Áp dụng `ToTensor()`, chuẩn hóa Z-score theo channel đơn `(pixel - 0.2860) / 0.3530`, và tăng cường dữ liệu lật ngang ngẫu nhiên (`RandomHorizontalFlip(p=0.5)`) dành riêng cho tập huấn luyện.
- `src/fashionmnist/data/split.py`: Thực hiện phân chia tập huấn luyện 60.000 ảnh ban đầu thành 50.000 ảnh Train và 10.000 ảnh Validation bằng phương pháp phân tầng (`stratified_split`), đảm bảo mỗi lớp có tỷ lệ phân bố chính xác (5.000 ảnh train và 1.000 ảnh val mỗi lớp). Hàm `verify_no_overlap()` xác thực giao giữa hai tập là tập rỗng nhằm loại bỏ hoàn toàn rủi ro rò rỉ dữ liệu (data leakage). Sau đó, module khởi tạo các DataLoader với `pin_memory=True` và `batch_size` tương ứng.

#### Phân hệ Mô hình (src/fashionmnist/models/)
- `src/fashionmnist/models/__init__.py`: Khởi tạo module kiến trúc mạng nơ-ron.
- `src/fashionmnist/models/network.py`: 
  - `BaselineMLP`: Mô hình mạng nhiều tầng kết nối đầy đủ. Đầu vào ảnh $1 \times 28 \times 28$ được làm phẳng thành vector 784 chiều, đi qua lớp ẩn 256 nơ-ron, hàm kích hoạt ReLU, lớp Dropout và lớp Linear cuối cùng ra 10 lớp ngõ ra (tổng cộng 203.530 tham số).
  - `SimpleCNN`: Mô hình mạng tích chập gồm 2 khối Conv (Conv2D $1 \rightarrow 32$ và $32 \rightarrow 64$, kernel 3x3, padding 1, ReLU, MaxPool 2x2) giảm kích thước ảnh về $7 \times 7$, tiếp nối bởi lớp Linear 256 nơ-ron, Dropout(0.3) và Linear ra 10 ngõ ra (tổng cộng 38.122 tham số).
  - `count_parameters()`: Hàm tính toán tổng số lượng tham số có thể huấn luyện của mô hình.
  - `build_model()`: Hàm Factory xây dựng mô hình dựa trên cấu hình chỉ định.
  - Quy tắc thiết kế: Cả hai mô hình đều xuất Raw Logits (không đặt Softmax ở lớp cuối) để kết hợp với `nn.CrossEntropyLoss()` của PyTorch, đảm bảo tính ổn định số học và tránh tràn số.

#### Phân hệ Huấn luyện (src/fashionmnist/training/)
- `src/fashionmnist/training/__init__.py`: Khởi tạo module huấn luyện.
- `src/fashionmnist/training/trainer.py`: Điều phối toàn bộ quá trình huấn luyện:
  - `build_optimizer()`: Khởi tạo bộ tối ưu hóa (Adam hoặc SGD) kèm tham số suy giảm trọng số `weight_decay`.
  - `build_scheduler()`: Khởi tạo bộ điều chỉnh tốc độ học theo chu kỳ Cosine (`CosineAnnealingLR`) hoặc theo bước (`StepLR`).
  - `train_one_epoch()`: Thực hiện lan truyền tiến (forward pass), tính hàm mất mát `CrossEntropyLoss`, lan truyền ngược (backward pass), cập nhật trọng số `optimizer.step()`, tính loss trung bình và độ chính xác epoch.
  - `evaluate_one_epoch()`: Đánh giá mô hình trên tập validation hoặc test dưới chế độ `@torch.no_grad()` và `model.eval()`.
  - `train()`: Vòng lặp huấn luyện chính, tích hợp cơ chế dừng sớm (Early Stopping) dựa trên số epoch liên tiếp không cải thiện và lưu checkpoint định kỳ.
- `src/fashionmnist/training/checkpoint.py`: Quản lý lưu trữ trạng thái mô hình (`model_state_dict`), trạng thái bộ tối ưu (`optimizer_state_dict`), số epoch và giá trị accuracy tốt nhất vào tệp `best_model.pt`. Cung cấp hàm nạp checkpoint phục vụ cho việc suy luận hoặc đánh giá độc lập.

#### Phân hệ Đánh giá (src/fashionmnist/evaluation/)
- `src/fashionmnist/evaluation/__init__.py`: Khởi tạo module đánh giá.
- `src/fashionmnist/evaluation/metrics.py`: Tính toán các thước đo định lượng bao gồm: Accuracy, Macro-averaged F1, Weighted F1, Macro-averaged Precision, Macro-averaged Recall, cùng ma trận nhầm lẫn (Confusion Matrix) dạng số lượng và dạng chuẩn hóa tỉ lệ.
- `src/fashionmnist/evaluation/error_analysis.py`: Thực hiện bóc tách các trường hợp phân loại sai trên tập kiểm thử, tìm ra các cặp lớp bị nhầm lẫn nhiều nhất kèm số lượng lỗi cụ thể và trích xuất dữ liệu ảnh lỗi phục vụ việc trực quan hóa.

#### Phân hệ Trực quan hóa (src/fashionmnist/visualization/)
- `src/fashionmnist/visualization/__init__.py`: Khởi tạo module đồ thị.
- `src/fashionmnist/visualization/plots.py`: Chứa các hàm vẽ đồ thị sử dụng hai thư viện `matplotlib` và `seaborn`:
  - `plot_class_distribution()`: Vẽ phân phối số lượng mẫu mỗi lớp bằng biểu đồ thanh ngang.
  - `plot_sample_grid()`: Vẽ ma trận ảnh mẫu kích thước 4x10 từ tập dữ liệu thô.
  - `plot_pixel_histogram()`: Vẽ biểu đồ phân bố cường độ pixel từ 0.0 đến 1.0.
  - `plot_class_mean_images()`: Vẽ hình ảnh trung bình đại diện cho từng lớp.
  - `plot_training_history()`: Vẽ đường cong suy giảm Loss và tăng trưởng Accuracy qua các epoch.
  - `plot_confusion_matrix()`: Vẽ Heatmap ma trận nhầm lẫn chuẩn hóa và số lượng mẫu.
  - `plot_per_class_f1()`: Vẽ biểu đồ so sánh chỉ số F1-Score giữa các lớp.
  - `plot_error_analysis_grid()`: Vẽ lưới các mẫu dự đoán sai kèm nhãn gốc (màu xanh lá) và nhãn dự đoán (màu đỏ).
  - `plot_predictions()`: Vẽ kết quả dự đoán ngẫu nhiên kèm điểm tin cậy phần trăm.

#### Phân hệ Tiện ích (src/fashionmnist/utils/)
- `src/fashionmnist/utils/__init__.py`: Khởi tạo module tiện ích.
- `src/fashionmnist/utils/reproducibility.py`: Chứa hàm `set_seed()` thiết lập seed cố định cho `random`, `numpy`, `torch` và `torch.cuda`, đồng thời cấu hình `cudnn.deterministic = True` nhằm đảm bảo mọi kết quả thực nghiệm đều có khả năng tái lập bit-for-bit trên CPU. Cung cấp hàm `get_device()` tự động chọn GPU khi khả dụng.
- `src/fashionmnist/utils/logging.py`: Lưu trữ toàn bộ lịch sử huấn luyện, các giá trị siêu tham số và kết quả đánh giá thành tệp JSON trong thư mục `artifacts/experiments/`.

---

### 2.3 Thư mục Kịch Bản Thực Thi (scripts/)

- `scripts/download_and_eda.py`: Kịch bản thực thi bước 1: tải tập dữ liệu về thư mục `data/raw/`, chạy kiểm tra độ toàn vẹn và tạo toàn bộ biểu đồ EDA trong `artifacts/figures/`.
- `scripts/train.py`: Kịch bản thực thi bước 2: nhận tham số `--experiment <tên_kịch_bản>` (mặc định là kịch bản trong `configs/experiments.yaml`), xây dựng mô hình, thực hiện huấn luyện, lưu biểu đồ learning curve và lưu checkpoint tối ưu `best_model.pt`.
- `scripts/evaluate.py`: Kịch bản thực thi bước 3: nạp mô hình tốt nhất từ `best_model.pt`, chạy đánh giá trên 10.000 mẫu của tập kiểm thử độc lập, tạo báo cáo phân loại chi tiết `classification_report.txt`, xuất ma trận nhầm lẫn và biểu đồ phân tích lỗi.
- `scripts/predict.py`: Kịch bản thực thi bước 4: tải checkpoint đã lưu và chạy thử nghiệm suy luận trên một nhóm ảnh ngẫu nhiên từ tập kiểm thử, lưu kết quả thành tệp ảnh trực quan `artifacts/predictions/`.
- `scripts/generate_complete_notebook.py`: Kịch bản tự động khởi tạo và cập nhật file Jupyter Notebook `notebooks/fashionmnist_complete_pipeline.ipynb` với đầy đủ mã nguồn, biểu đồ hình ảnh được nhúng sẵn và lời giải thích chi tiết.

---

### 2.4 Thư mục Báo Cáo Thực Nghiệm (notebooks/)

- `notebooks/fashionmnist_complete_pipeline.ipynb`: Tệp sổ tay Jupyter Notebook hoàn chỉnh chứa toàn bộ quy trình từ tải dữ liệu, phân tích khám phá (EDA), so sánh kiến trúc mô hình, biểu đồ đường cong học tập, ma trận nhầm lẫn đến phân tích lỗi chuyên sâu. Tệp đã được xuất kèm hình ảnh đồ thị phân giải cao và diễn giải chi tiết cho từng phần.

---

### 2.5 Thư mục Kiểm Thử Tự Động (tests/)

Toàn bộ hệ thống được bảo đảm bởi 46 bài kiểm thử tự động sử dụng thư viện `pytest`:
- `tests/test_dataset.py` (17 bài test): Kiểm tra kích thước tập train (60.000), test (10.000), shape ảnh $[1, 28, 28]$, dtype `torch.float32`, dải giá trị $[0.0, 1.0]$, dải nhãn $[0, 9]$, tính đúng đắn của transforms, việc chia tập train/val đúng tỷ lệ 50.000/10.000 và đảm bảo giao giữa hai tập bằng rỗng (không rò rỉ dữ liệu).
- `tests/test_model.py` (11 bài test): Kiểm tra kích thước ngõ ra logits $[B, 10]$ cho cả BaselineMLP và SimpleCNN, xác nhận không chứa Softmax ngầm ở lớp cuối, kiểm tra tính hữu hạn của giá trị và số lượng tham số mô hình.
- `tests/test_training.py` (11 bài test): Kiểm tra bộ tối ưu hóa Adam/SGD, lịch trình Cosine Annealing, đảm bảo trọng số được cập nhật sau mỗi bước lan truyền ngược (`loss.backward()` và `optimizer.step()`), kiểm tra việc chuyển đổi chính xác giữa chế độ `model.train()` và `model.eval()`.
- `tests/test_checkpoint.py` (7 bài test): Kiểm tra tính toàn vẹn khi lưu và nạp `state_dict`, optimizer state, epoch lưu trữ, đảm bảo checkpoint chỉ được cập nhật khi chỉ số validation accuracy thực sự cải thiện.

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

- `pyproject.toml`: Khai báo thông tin dự án, cấu hình đóng gói package theo chuẩn PEP 517/518 (sử dụng `setuptools.build_meta`), liệt kê danh sách thư viện phụ thuộc (`torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pyyaml`, `tqdm`) và cấu hình thực thi cho `pytest`.
- `.gitignore`: Cấu hình danh sách tệp không theo dõi trên Git: mã nhị phân Python (`__pycache__`), môi trường ảo (`.venv`), dữ liệu thô kích thước lớn (`data/raw/`) và các checkpoint epoch trung gian.

---

## 3. Bảng Kết Quả Thực Nghiệm Đối Chứng

Mô hình tốt nhất được chọn dựa trên **Validation Accuracy** trước khi tiến hành đánh giá trên tập Kiểm thử (Test Set):

| Kịch Bản (Experiment) | Kiến Trúc Mô Hình | Số Tham Số | Val Accuracy | Test Accuracy | Macro F1 | Weighted F1 | Thời Gian Chạy |
|---|---|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 203.530 | 82.16% | 82.07% | 0.8204 | 0.8204 | 192.5s |
| `exp_cnn_small` | **SimpleCNN** | **38.122** | **84.83%** | **84.75%** | **0.8468** | **0.8468** | 829.9s |

### Nhận xét phân tích kỹ thuật:
1. **Hiệu quả của phép tích chập:** SimpleCNN đạt độ chính xác cao hơn rõ rệt (+2.68% Test Accuracy, +2.64% F1-score) dù số lượng trọng số nhỏ hơn 5.3 lần so với BaselineMLP. Điều này chứng minh khả năng học các đặc trưng không gian cục bộ (spatial features) và tính chia sẻ trọng số của các bộ lọc tích chập.
2. **Cải thiện trên các lớp khó:** Lớp `Pullover` (Áo len chui đầu) tăng F1-score từ 68.44% lên 79.20% (+10.76%); lớp `Sneaker` đạt 93.04%; lớp `Sandal` đạt 94.42%.
3. **Giảm thiểu số lượng lỗi:** Tổng số trường hợp dự đoán sai trên 10.000 mẫu kiểm thử giảm từ 1.793 ca (BaselineMLP) xuống còn 1.525 ca (SimpleCNN), tương đương mức giảm 268 lỗi.

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
