# 04 — Kiến trúc Mô hình

## Hai Kiến trúc

### 1. BaselineMLP (Baseline)

```
Input (1×28×28) → Flatten(784) → Linear(784→256) → ReLU → Dropout → Linear(256→10)
```

- Không khai thác cấu trúc không gian
- Dùng làm mức tham chiếu tối thiểu
- Tham số: 203 530

### 2. SimpleCNN (Mô hình chính)

```
Conv Block 1: Conv(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  [28→14]
Conv Block 2: Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) [14→7]
GAP: AdaptiveAvgPool(1×1) → Flatten → Linear(64→256) → ReLU → Dropout(0.3) → Linear(256→10)
```

| Thành phần | Vai trò |
|---|---|
| Conv + BN | Học đặc trưng cục bộ, ổn định training |
| MaxPool | Giảm chiều, tăng invariance |
| Global Avg Pool | Giảm tham số, giảm overfitting |
| Dropout(0.3) | Regularization |

- **Tham số:** 38 122 (nhỏ gọn, giảm đáng kể tham số nhờ Global Average Pooling)

## Input/Output

- **Input:** Tensor `(B, 1, 28, 28)`, đã normalize
- **Output:** Tensor `(B, 10)` — logits thô (không Softmax)

## Loss Function và Optimizer

| Yếu tố | Lựa chọn | Lý do |
|---|---|---|
| Loss | CrossEntropyLoss | Tiêu chuẩn cho multiclass; kết hợp log-softmax + NLLLoss |
| Optimizer | Adam | Hội tụ nhanh, phù hợp dataset vừa |
| LR Scheduler | CosineAnnealingLR | Giảm LR mượt, tránh oscillation cuối training |

**Không** đặt Softmax ở cuối model vì CrossEntropyLoss đã tích hợp log-softmax.
