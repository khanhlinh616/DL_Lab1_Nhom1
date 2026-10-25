# Phân Loại Ảnh Thời Trang FashionMNIST Bằng PyTorch

Dự án xây dựng và đánh giá hệ thống phân loại hình ảnh thời trang 10 lớp trên tập dữ liệu FashionMNIST sử dụng PyTorch. Dự án so sánh thực nghiệm giữa mạng nơ-ron kết nối đầy đủ (BaselineMLP) và mạng nơ-ron tích chập (SimpleCNN), cấu hình hóa qua tệp YAML, đảm bảo tính tái lập và tích hợp bộ kiểm thử tự động.

---

## 1. Tổng Quan Bài Toán

- **Bài toán:** Phân loại ảnh đơn sắc (grayscale $28 \times 28$ pixel) vào 10 danh mục trang phục thời trang.
- **Tập dữ liệu:** FashionMNIST (Zalando Research) gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm thử.
- **Danh mục 10 lớp:**
  - `0`: T-shirt/top (Áo phông)
  - `1`: Trouser (Quần dài)
  - `2`: Pullover (Áo len chui đầu)
  - `3`: Dress (Váy liền)
  - `4`: Coat (Áo khoác ngoài)
  - `5`: Sandal (Dép xăng-đan)
  - `6`: Shirt (Áo sơ mi)
  - `7`: Sneaker (Giày thể thao)
  - `8`: Bag (Túi xách)
  - `9`: Ankle boot (Ủng / Bốt cổ thấp)
- **Mô hình đối chứng:**
  - **BaselineMLP:** 203.530 tham số (Flatten 784 → Linear 256 → ReLU → Linear 10).
  - **SimpleCNN:** 38.122 tham số (2 Conv Blocks + BatchNorm + MaxPool + Global Average Pooling + Linear 256 + Dropout 0.3 + Linear 10).

---

## 2. Kiến Trúc Thư Mục và Tệp Tin

Toàn bộ hệ thống được phân chia theo cấu trúc module:

```text
fashionmnist-deep-learning/
├── .gitignore                          # Cấu hình bỏ qua tệp nhị phân, cache, dữ liệu thô và checkpoints tạm
├── pyproject.toml                      # Cấu hình đóng gói package (PEP 517/518), dependencies và pytest
├── README.md                           # Tài liệu tổng quan và hướng dẫn vận hành dự án
├── configs/                            # Quản lý cấu hình và siêu tham số thực nghiệm
│   ├── default.yaml                    # Cấu hình gốc (seed, split ratio, batch size, optimizer mặc định)
│   └── experiments.yaml                # Cấu hình ghi đè cho từng thực nghiệm (baseline_mlp, exp_cnn_small, ...)
├── docs/                               # Tài liệu phân tích kỹ thuật chuyên sâu
│   ├── 01_project_overview.md          # Tổng quan dự án, phạm vi bài toán và mục tiêu thực nghiệm
│   ├── 02_dataset_and_eda.md           # Nguồn dữ liệu, quy trình EDA và thống kê phân phối pixel
│   ├── 03_preprocessing_and_splitting.md# Tiền xử lý, chuẩn hóa Z-score và phân chia tập không rò rỉ
│   ├── 04_model_architecture.md        # Kiến trúc toán học chi tiết của BaselineMLP và SimpleCNN
│   ├── 05_training_and_experiments.md  # Chiến lược tối ưu, hàm mất mát và siêu tham số huấn luyện
│   ├── 06_evaluation_and_error_analysis.md# Phương pháp luận đánh giá và phân tích nguyên nhân sai số
│   ├── 07_reproducibility_and_limitations.md# Báo cáo tính tái lập, phiên bản thư viện và giới hạn hệ thống
│   └── 08_results_and_conclusion.md    # Bảng tổng hợp thực nghiệm đối chứng và hướng phát triển
├── notebooks/                          # Báo cáo thực nghiệm trực quan
│   └── fashionmnist_complete_pipeline.ipynb # Notebook hoàn chỉnh kèm biểu đồ nhúng và phân tích chi tiết
├── scripts/                            # CLI entrypoints điều phối quy trình đầu cuối
│   ├── download_and_eda.py             # Tải dữ liệu, kiểm tra toàn vẹn và tạo biểu đồ EDA
│   ├── train.py                        # Huấn luyện mô hình theo experiment chỉ định
│   ├── evaluate.py                     # Đánh giá checkpoint tốt nhất trên tập Test độc lập
│   ├── predict.py                      # Chạy suy luận (inference) trên ảnh thực tế
│   └── generate_complete_notebook.py   # Tự động khởi tạo notebook với đầy đủ biểu đồ nhúng sẵn
├── src/fashionmnist/                   # Gói mã nguồn chính (core package)
│   ├── __init__.py                     # Khởi tạo package
│   ├── config.py                       # Nạp và validate cấu hình YAML
│   ├── data/                           # Module dữ liệu và tiền xử lý
│   │   ├── __init__.py
│   │   ├── dataset.py                  # Tải FashionMNIST và kiểm tra tính toàn vẹn
│   │   ├── eda.py                      # Khám phá dữ liệu (EDA), tính mean/std trước khi chia tập
│   │   ├── transforms.py               # Pipeline chuẩn hóa Z-score và data augmentation
│   │   └── split.py                    # Stratified split (50k/10k) và khởi tạo DataLoader
│   ├── models/                         # Module kiến trúc mạng nơ-ron
│   │   ├── __init__.py
│   │   └── network.py                  # Định nghĩa BaselineMLP, SimpleCNN và build_model factory
│   ├── training/                       # Module huấn luyện và quản lý checkpoint
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Vòng lặp train/val, Cosine LR scheduler, early stopping
│   │   └── checkpoint.py               # Lưu và nạp state_dict (model, optimizer, epoch, metrics)
│   ├── evaluation/                     # Module đánh giá và phân tích sai số
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Tính Accuracy, Precision, Recall, Macro/Weighted F1, CM
│   │   └── error_analysis.py           # Bóc tách mẫu đoán sai và xếp hạng cặp lớp nhầm lẫn
│   ├── visualization/                  # Module trực quan hóa
│   │   ├── __init__.py
│   │   └── plots.py                    # Biểu đồ EDA, learning curves, confusion matrix, error grid
│   └── utils/                          # Module tiện ích bổ trợ
│       ├── __init__.py
│       ├── reproducibility.py          # Cố định random seed (CPU/CUDA) và chọn thiết bị tính toán
│       └── logging.py                  # Ghi nhật ký thực nghiệm (JSON metrics/history)
├── tests/                              # 46 kiểm thử tự động với Pytest
│   ├── test_dataset.py                 # Kiểm thử shape, dtype, dải nhãn, transforms, split không rò rỉ
│   ├── test_model.py                   # Kiểm thử logits output, không có Softmax ngầm, số tham số
│   ├── test_training.py                # Kiểm thử optimizer, gradient update, chuyển đổi train/eval
│   └── test_checkpoint.py              # Kiểm thử lưu và nạp state_dict checkpoint chính xác
└── artifacts/                          # Nơi lưu trữ sản phẩm thực nghiệm
    ├── checkpoints/                    # Trọng số mô hình tối ưu (best_model.pt) và epoch checkpoints
    ├── eda/                            # Thống kê phân phối pixel và tính toàn vẹn dữ liệu (.json)
    ├── experiments/                    # Nhật ký lịch sử huấn luyện chi tiết theo từng epoch (.json)
    ├── figures/                        # Toàn bộ biểu đồ phân tích trực quan độ phân giải cao (.png)
    ├── metrics/                        # Báo cáo đánh giá phân loại định lượng (.txt, .json)
    ├── predictions/                    # Ảnh trực quan hóa kết quả suy luận thực tế (.png)
    └── reports/                        # Báo cáo kỹ thuật tổng kết nghiệm thu dự án (.md)
```

---

## 3. Hướng Dẫn Cài Đặt và Thực Thi

### 3.1 Cài đặt môi trường

```bash
# Cài đặt gói mã nguồn ở chế độ editable
py -3 -m pip install -e . --no-deps

# Cài đặt các thư viện kiểm thử và tạo notebook nếu chưa có
py -3 -m pip install pytest nbformat
```

### 3.2 Vận hành pipeline qua dòng lệnh

```bash
# 1. Tải tập dữ liệu, kiểm tra tính toàn vẹn và tạo biểu đồ EDA
py -3 scripts/download_and_eda.py

# 2. Huấn luyện mô hình
py -3 scripts/train.py --experiment baseline_mlp
py -3 scripts/train.py --experiment exp_cnn_small

# 3. Đánh giá mô hình tốt nhất trên tập Test độc lập (10.000 mẫu)
py -3 scripts/evaluate.py --experiment exp_cnn_small

# 4. Chạy suy luận (inference) trên mẫu ảnh thực tế
py -3 scripts/predict.py --experiment exp_cnn_small --n_samples 8

# 5. Khởi tạo / cập nhật notebook tổng hợp có nhúng hình ảnh
py -3 scripts/generate_complete_notebook.py

# 6. Chạy toàn bộ 46 bài kiểm thử tự động
py -3 -m pytest tests/ -v
```

---

## 4. Kết Quả Thực Nghiệm Đối Chứng

Mô hình tốt nhất được lựa chọn dựa trên **Validation Accuracy** trước khi đánh giá một lần duy nhất trên tập Kiểm thử (Test Set):

| Kịch Bản | Kiến Trúc | Số Tham Số | Val Accuracy | Test Accuracy | Macro F1 | Weighted F1 | Thời Gian Chạy |
|---|---|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 203.530 | 82.16% | 82.07% | 0.8204 | 0.8204 | 192.5s |
| `exp_cnn_small` | **SimpleCNN** | **38.122** | **84.83%** | **84.75%** | **0.8468** | **0.8468** | 829.9s |

### Phân tích kết quả:
1. **Hiệu quả cấu trúc tích chập:** SimpleCNN đạt độ chính xác cao hơn rõ rệt (+2.68% Test Accuracy, +2.64% F1-score) so với BaselineMLP dù số lượng tham số ít hơn 5.3 lần. Các lớp tích chập giữ được cấu trúc không gian 2D cục bộ thay vì dàn phẳng thành vector như MLP.
2. **Cải thiện trên các lớp phức tạp:** Lớp `Pullover` tăng F1-score từ 68.44% lên 79.20% (+10.76%); lớp `Sneaker` đạt 93.04%; lớp `Sandal` đạt 94.42%.
3. **Giảm thiểu số lượng lỗi:** Tổng số mẫu dự đoán sai trên 10.000 mẫu kiểm thử giảm từ 1.793 ca (BaselineMLP) xuống còn 1.525 ca (SimpleCNN), tương đương mức giảm 268 lỗi.

---

## 5. Phân Tích Sai Số (Error Analysis)

Dựa trên ma trận nhầm lẫn của mô hình `SimpleCNN`, 5 cặp lớp xảy ra nhầm lẫn nhiều nhất gồm:

| Cặp Lớp Nhầm Lẫn | Số Lượng Lỗi | Nguyên Nhân Kỹ Thuật |
|---|---|---|
| `Shirt` → `T-shirt/top` | 176 mẫu | Cấu trúc thân áo tương đồng; ở độ phân giải 28×28 grayscale, cúc áo và đường xẻ cổ áo sơ mi bị tiêu biến |
| `Coat` → `Shirt` | 124 mẫu | Áo khoác mở vạt trước tạo viền cổ chữ V tương tự áo sơ mi phanh ngực |
| `Pullover` → `Shirt` | 106 mẫu | Áo len dài tay có chiều dài thân và tay áo tương đương áo sơ mi |
| `Shirt` → `Coat` | 105 mẫu | Thiếu thông tin về độ dày chất liệu vải và màu sắc ở ảnh grayscale |
| `Coat` → `Pullover` | 105 mẫu | Thiếu độ tương phản bề mặt chất liệu dạ/len |

Ngược lại, các lớp có hình thái biên dạng đặc thù đạt độ chính xác cao: `Trouser` (F1: 97.05%), `Bag` (F1: 96.04%), `Sandal` (F1: 94.42%), `Ankle boot` (F1: 94.16%).

---

## 6. Danh Mục Tài Liệu Kỹ Thuật

| Tài Liệu | Nội Dung Chính |
|---|---|
| [01_project_overview.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/01_project_overview.md) | Tổng quan dự án, phạm vi bài toán và mục tiêu thực nghiệm |
| [02_dataset_and_eda.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/02_dataset_and_eda.md) | Phân tích khám phá dữ liệu (EDA) và thống kê pixel |
| [03_preprocessing_and_splitting.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/03_preprocessing_and_splitting.md) | Tiền xử lý, chuẩn hóa Z-score và chiến lược chia tập chống rò rỉ |
| [04_model_architecture.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/04_model_architecture.md) | Chi tiết toán học các lớp mạng của BaselineMLP và SimpleCNN |
| [05_training_and_experiments.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/05_training_and_experiments.md) | Vòng lặp huấn luyện, optimizer, Cosine LR scheduler, early stopping |
| [06_evaluation_and_error_analysis.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/06_evaluation_and_error_analysis.md) | Phương pháp luận đánh giá, confusion matrix và phân tích lỗi |
| [07_reproducibility_and_limitations.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/07_reproducibility_and_limitations.md) | Cấu hình tái lập (seed 42), môi trường thử nghiệm và giới hạn |
| [08_results_and_conclusion.md](file:///c:/Project/DL/fashionmnist-deep-learning/docs/08_results_and_conclusion.md) | Bảng tổng hợp số liệu thực nghiệm đối chứng và kết luận |
| [final_evaluation_report.md](file:///c:/Project/DL/fashionmnist-deep-learning/artifacts/reports/final_evaluation_report.md) | Báo cáo kỹ thuật tổng kết nghiệm thu dự án |
| [fashionmnist_complete_pipeline.ipynb](file:///c:/Project/DL/fashionmnist-deep-learning/notebooks/fashionmnist_complete_pipeline.ipynb) | Sổ tay thực nghiệm hoàn chỉnh kèm biểu đồ nhúng trực quan |
