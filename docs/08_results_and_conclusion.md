# 08 — Results và Conclusion

## Bảng So sánh Mô hình

| Experiment | Architecture | Val Acc | Test Acc | F1 (weighted) | Params | Time (s) |
|---|---|---|---|---|---|---|
| baseline_mlp | BaselineMLP | 0.8216 | 0.8207 | 0.8204 | 203,530 | 192.5 |
| exp_cnn_small | SimpleCNN | 0.8483 | 0.8475 | 0.8468 | 38,122 | 829.9 |

## Kết quả Mô hình Cuối cùng

Mô hình tốt nhất được chọn là **SimpleCNN** (`exp_cnn_small`) dựa trên **validation accuracy** (84.83% vs 82.16% của BaselineMLP):
- **Test Accuracy:** 84.75% (+2.68% so với Baseline MLP)
- **Macro / Weighted F1:** 0.8468 (+2.64% so với Baseline MLP)
- **Số lượng tham số:** 38,122 (nhỏ hơn 5.3 lần so với BaselineMLP với 203,530 tham số)
- **Số lượng mẫu phân loại sai:** Giảm từ 1,793 xuống 1,525 (-268 lỗi)

Chi tiết được lưu trữ tại:
- `artifacts/metrics/test_metrics_exp_cnn_small.json`
- `artifacts/metrics/classification_report_exp_cnn_small.txt`
- `artifacts/figures/confusion_matrix_exp_cnn_small_normalized.png`
- `artifacts/figures/per_class_f1_exp_cnn_small.png`

## Nhận xét Thực tế

1. **Hiệu quả cấu trúc tích chập (CNN):**
   - SimpleCNN cho kết quả cao hơn BaselineMLP (+2.68% Test Acc, +2.64% F1) dù số tham số ít hơn 5.3 lần. Các lớp tích chập giữ được quan hệ không gian 2D cục bộ thay vì dàn phẳng thành vector như MLP.
   - Lớp **Pullover** tăng F1-score từ **68.44% → 79.20% (+10.76%)**.
   - Lớp **Sneaker** tăng từ **87.81% → 93.04%**, **Sandal** từ **90.56% → 94.42%**.

2. **Lớp dễ nhận diện nhất (F1 > 94%):**
   - Trouser (F1: 97.05%), Bag (F1: 96.04%), Sandal (F1: 94.42%), Ankle boot (F1: 94.16%).

3. **Cặp lớp nhầm lẫn nhiều nhất (Error Analysis):**
   - **Shirt ↔ T-shirt/top (176 lỗi)**, **Coat ↔ Shirt (124 lỗi)**, **Pullover ↔ Shirt (106 lỗi)**.
   - Nguyên nhân: Các trang phục thân trên có hình bóng (silhouette) và cấu trúc cổ áo, tay áo rất tương đồng ở độ phân giải 28×28 grayscale.

## Hướng Cải tiến (có căn cứ)

1. **Tăng depth:** thêm conv block thứ 3 nếu val accuracy còn chỗ cải thiện
2. **Label smoothing:** giảm confidence quá cao trên các lớp dễ nhầm
3. **Learning rate warm-up:** ổn định đầu training với Adam
4. **Mixup/CutMix augmentation:** cải thiện robustness cho các cặp dễ nhầm
5. **Transfer learning:** Fine-tune ResNet-18 với input upscaled 32×32 nếu cần accuracy cao hơn 93%
