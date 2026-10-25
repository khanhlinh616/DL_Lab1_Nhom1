# FashionMNIST Deep Learning Classification

## Mục tiêu

Phân loại ảnh thời trang thành 10 lớp sử dụng PyTorch, theo quy trình Machine Learning 21 bước đầy đủ.

## Dataset

- **Nguồn:** `torchvision.datasets.FashionMNIST` (tải tự động)
- **Quy mô:** 60 000 train + 10 000 test
- **Ảnh:** Grayscale 28×28 pixel
- **10 lớp:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Kiến trúc Thư mục

```
fashionmnist-deep-learning/
├── configs/          # default.yaml, experiments.yaml
├── data/raw/         # FashionMNIST raw files (auto-downloaded)
├── docs/             # 8 tài liệu chi tiết (01–08)
├── notebooks/        # Notebooks EDA và experiments
├── scripts/          # download_and_eda.py | train.py | evaluate.py | predict.py
├── src/fashionmnist/ # Package nguồn chính
│   ├── config.py
│   ├── data/         # dataset, transforms, eda, split
│   ├── models/       # network.py (BaselineMLP, SimpleCNN)
│   ├── training/     # trainer.py, checkpoint.py
│   ├── evaluation/   # metrics.py, error_analysis.py
│   ├── visualization/# plots.py
│   └── utils/        # reproducibility.py, logging.py
├── tests/            # test_dataset, test_model, test_training, test_checkpoint
├── artifacts/        # EDA, figures, metrics, predictions, experiments, checkpoints
└── pyproject.toml
```

## Các Thành phần Chính

| Thành phần | Mô tả |
|---|---|
| `BaselineMLP` | MLP đơn giản, baseline |
| `SimpleCNN` | 2 Conv blocks + GAP, mô hình chính |
| `Trainer` | Train/eval loop, early stopping, checkpoint |
| `EDA` | Thống kê, phân bố lớp, ảnh mẫu, histogram |
| `Metrics` | Accuracy, F1 macro/weighted, confusion matrix |
| `ErrorAnalysis` | Grid ảnh sai, top confused pairs |

## Sản phẩm Đầu ra

- `artifacts/eda/` — thống kê JSON và biểu đồ EDA
- `artifacts/figures/` — training curves, confusion matrix, F1 bar chart
- `artifacts/checkpoints/` — `best_model.pt` mỗi experiment
- `artifacts/metrics/` — JSON metrics và classification report
- `artifacts/experiments/` — log JSON từng experiment
- `artifacts/predictions/` — grid dự đoán

## Trạng thái và Giới hạn

- **Cài đặt:** PyTorch 2.14.1 (CPU), torchvision 0.29.1 — đã cài đặt và cấu hình
- **EDA/Training/Evaluation:** Đã hoàn thành toàn bộ 21 bước pipeline và lưu trữ artifacts
- **GPU:** Hỗ trợ CUDA nếu có, tự động fallback về CPU
- **Reproducibility:** Cố định seed = 42 trên toàn bộ thư viện (Python, NumPy, PyTorch)
- **Test set:** Chỉ đánh giá sau khi chọn mô hình tốt nhất dựa trên Validation Accuracy

## Kết quả Thực nghiệm

| Experiment | Kiến Trúc | Số Tham Số | Val Acc | Test Acc | Macro F1 | Weighted F1 | Thời Gian Huấn Luyện |
|---|---|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 203,530 | 82.16% | 82.07% | 0.8204 | 0.8204 | 192.5s |
| `exp_cnn_small` | **SimpleCNN** | **38,122** | **84.83%** | **84.75%** | **0.8468** | **0.8468** | 829.9s |

*Mô hình SimpleCNN đạt hiệu năng cao hơn +2.68% Test Accuracy và +2.64% F1-score dù số lượng tham số nhỏ hơn 5.3 lần so với Baseline MLP.*
