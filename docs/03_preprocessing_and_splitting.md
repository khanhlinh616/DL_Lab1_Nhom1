# 03 — Preprocessing và Splitting

## Tensor Conversion và Normalization

**ToTensor:** Chuyển PIL Image (uint8 [0,255]) → float32 Tensor [0.0, 1.0]  
**Normalize:** Áp dụng `(pixel - mean) / std` theo channel đơn (grayscale)

| Tham số | Giá trị |
|---|---|
| mean | 0.2860 |
| std | 0.3530 |

Các giá trị này được tính từ **tập train** (60 000 ảnh), lưu trong `configs/default.yaml`.  
Không dùng giá trị tính từ validation hoặc test — tránh data leakage.

**Tại sao normalize?** Đưa pixel về phân phối xấp xỉ chuẩn (mean≈0, std≈1) giúp gradient ổn định hơn và tăng tốc hội tụ.

## Augmentation (chỉ áp dụng trên train)

| Transform | Tham số | Mục đích |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Tăng đa dạng về hướng |
| RandomCrop(28, padding=4) | padding=4 | Tăng robustness về vị trí |

Validation và test **không** dùng augmentation — chỉ ToTensor + Normalize.

## Chiến lược Chia Tập

| Tập | Nguồn | Kích thước | Mục đích |
|---|---|---|---|
| Train | 83.33% của official train | ~50 000 | Cập nhật gradient |
| Validation | 16.67% của official train | ~10 000 | Chọn mô hình, early stopping |
| Test | Official test set | 10 000 | Đánh giá cuối cùng |

**Phương pháp:** Stratified split — mỗi lớp được chia theo đúng tỷ lệ.  
**Random seed:** 42 (cố định, ghi trong config).

## Ngăn Data Leakage

- Test set chỉ được đọc trong `scripts/evaluate.py` sau khi chọn xong mô hình
- Normalization constants lấy từ train set, không tính lại từ val/test
- Validation **không** tham gia cập nhật gradient
- `verify_no_overlap()` kiểm tra không trùng index giữa train và val
