# 07 — Reproducibility và Limitations

## Seed và Môi trường

| Yếu tố | Giá trị / Ghi chú |
|---|---|
| Global seed | 42 (Python, NumPy, PyTorch) |
| Stratified split | Dùng `np.random.default_rng(seed=42)` |
| `cudnn.deterministic` | True |
| `cudnn.benchmark` | False |

## Khả năng Tái Lập

**CPU:** Hoàn toàn tái lập bit-for-bit khi cùng seed, cùng phiên bản thư viện, cùng OS.

**GPU:** Không đảm bảo tái lập hoàn toàn do một số CUDA kernel non-deterministic (e.g., atomics trong backward pass). Thiết lập `cudnn.deterministic=True` giảm thiểu nhưng không loại bỏ hoàn toàn.

**Môi trường được test:**
```
Python: 3.14
PyTorch: 2.14.1+cpu
torchvision: 0.29.1+cpu
NumPy: 2.5.2
scikit-learn: 1.9.1
```

## Giới hạn về Tài nguyên

- Tất cả thí nghiệm được thiết kế chạy được trên CPU (không bắt buộc GPU)
- Thời gian huấn luyện trên CPU: ước tính 5–15 phút/experiment tùy phần cứng
- Không chạy K-fold cross-validation trên toàn bộ dataset (chi phí lớn, dataset đủ lớn để train/val split đáng tin cậy)

## Giới hạn về Phương pháp

- Augmentation chỉ gồm flip và crop cơ bản — không dùng Cutout, Mixup
- Không fine-tune pretrained model (ResNet, ViT) — không cần thiết cho 28×28
- Không thực hiện kiểm định thống kê (t-test) giữa các experiment — tập test đủ lớn để so sánh accuracy trực tiếp
- Không dùng GradCAM hay LIME để giải thích nội tại mạng

## Các Vấn đề Chưa Giải quyết

- Optimal số lớp conv và kích thước filter chưa được tìm kiếm có hệ thống (grid search)
- Normalization constants (0.2860, 0.3530) được tính trước và hardcode vào config; nên tính lại nếu dùng dataset khác
