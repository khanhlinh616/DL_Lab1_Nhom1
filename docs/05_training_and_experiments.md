# 05 — Training và Experiments

## Quy trình Huấn luyện

1. `model.train()` → forward → `CrossEntropyLoss` → `loss.backward()` → `optimizer.step()`
2. `model.eval()` + `torch.no_grad()` → validation loss và accuracy
3. LR scheduler step sau mỗi epoch
4. Lưu `best_model.pt` khi `val_accuracy` cải thiện
5. Early stopping nếu không cải thiện sau `patience` epochs

## Cấu hình Huấn luyện Chính (SimpleCNN)

| Siêu tham số | Giá trị |
|---|---|
| Epochs | 15 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| LR scheduler | CosineAnnealingLR |
| Early stopping | patience=5 |
| Dropout | 0.3 |
| Seed | 42 |

## Chiến lược Checkpoint

- `best_model.pt` — lưu mỗi khi `val_accuracy` cải thiện
- `checkpoint_epoch_{N}.pt` — lưu mỗi 5 epoch
- Mỗi checkpoint chứa: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_metric`, `cfg_model`, `cfg_training`

## Các Thí nghiệm

| Tên | Kiến trúc | LR | Batch | Epochs |
|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP | 0.01 (SGD) | 128 | 10 |
| `exp_cnn_small` | SimpleCNN | 0.001 (Adam) | 128 | 15 |
| `exp_lr_01` | SimpleCNN | 0.01 (Adam) | 128 | 15 |
| `exp_batch_256` | SimpleCNN | 0.001 (Adam) | 256 | 15 |

> Kết quả thực tế của các thí nghiệm được ghi trong `artifacts/experiments/` và cập nhật trong `08_results_and_conclusion.md` sau khi chạy.

## Bias–Variance và No Free Lunch

- **Underfitting:** train loss cao, val loss cao → tăng model capacity hoặc tăng epochs
- **Overfitting:** train loss thấp nhưng val loss tăng → tăng dropout, weight decay, augmentation
- **No Free Lunch:** SimpleCNN phù hợp với ảnh có cấu trúc không gian. Với dữ liệu bảng, mô hình tuyến tính có thể tốt hơn. Không có kiến trúc nào tối ưu universally.
