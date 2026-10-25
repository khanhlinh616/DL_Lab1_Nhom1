# 06 — Evaluation và Error Analysis

## Metrics Sử dụng

| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Precision (macro) | Trung bình không trọng số của precision mỗi lớp |
| Recall (macro) | Trung bình không trọng số của recall mỗi lớp |
| F1 (macro) | Trung bình không trọng số F1 mỗi lớp |
| F1 (weighted) | Trung bình có trọng số theo số mẫu |
| Confusion matrix | Ma trận nhầm lẫn 10×10 |

**Cân bằng lớp:** FashionMNIST có phân bố lớp cân bằng → macro = weighted trong hầu hết trường hợp.

**Test set** chỉ được đánh giá **một lần duy nhất** sau khi chọn xong mô hình tốt nhất theo validation accuracy.

## Kết quả Đánh giá

Kết quả thực nghiệm trên tập test được lưu tại:
- `artifacts/metrics/test_metrics_{experiment}.json`
- `artifacts/metrics/classification_report_{experiment}.txt`

## Confusion Matrix

Hai phiên bản:
1. **Normalized by row** (Recall): cho thấy từng lớp bị nhầm sang lớp nào nhiều nhất
2. **Counts**: số mẫu tuyệt đối

Lưu tại: `artifacts/figures/confusion_matrix_{experiment}_{normalized|counts}.png`

## Error Analysis

Phân tích tập trung vào:
- Top cặp lớp nhầm lẫn nhiều nhất (thường: Shirt ↔ T-shirt/top, Coat ↔ Pullover)
- Per-class error rate
- Grid 25 ảnh dự đoán sai: hiển thị nhãn thật, nhãn dự đoán, confidence

**Giới hạn:** Phân tích lỗi được thực hiện trên test set — không dùng để điều chỉnh mô hình.

## Model Interpretability

Giải thích kết quả qua:
- Confusion matrix: thấy rõ pattern nhầm lẫn có hệ thống
- Ảnh sai + confidence thấp: gợi ý về độ khó của từng lớp
- Per-class F1: lớp nào model học tốt, lớp nào còn yếu

*Không dùng phương pháp giải thích nội tại (GradCAM, LIME) trong phạm vi này.*
