# 01 — Project Overview

## Mục tiêu và Phạm vi

Phân loại ảnh thời trang 28×28 pixel (grayscale) thành 10 lớp sử dụng mạng nơ-ron sâu (PyTorch).  
Phạm vi: tải dữ liệu → EDA → preprocessing → training → evaluation → báo cáo.  
Không mở rộng thành platform phức tạp; chỉ phục vụ bài tập học sâu.

## Bài toán và Phương pháp

| Yếu tố | Giá trị |
|---|---|
| Input | Ảnh grayscale 1×28×28, pixel ∈ [0, 1] |
| Output | 10 logits (lớp thời trang) |
| Paradigm | Supervised learning — multiclass classification |
| Loss | CrossEntropyLoss |
| Metric chính | Accuracy trên validation set |

**Tại sao mạng nơ-ron?**  
FashionMNIST có cấu trúc không gian (texture, shape) mà CNN có thể học tự động tốt hơn mọi phương pháp trích đặc trưng thủ công.

## Tổng quan Kiến trúc Hệ thống

```
scripts/download_and_eda.py → [EDA artifacts]
scripts/train.py            → [checkpoints, history logs]
scripts/evaluate.py         → [metrics JSON, figures]
scripts/predict.py          → [prediction grid]
```

**Modules chính:**
- `fashionmnist/config.py` — cấu hình tập trung
- `fashionmnist/data/` — tải, EDA, transform, chia tập
- `fashionmnist/models/network.py` — BaselineMLP, SimpleCNN
- `fashionmnist/training/` — trainer, checkpoint
- `fashionmnist/evaluation/` — metrics, error analysis
- `fashionmnist/visualization/plots.py` — biểu đồ

## Sản phẩm Bàn giao

- Toàn bộ source code có thể chạy lại
- Artifacts: EDA figures, training curves, confusion matrix, checkpoint
- 8 tài liệu trong `docs/`
- Báo cáo kết quả trong `artifacts/reports/`
