# Báo cáo Tổng kết Đánh giá Kỹ thuật (Technical Evaluation Report)

**Dự án:** Phân loại Ảnh Thời trang FashionMNIST bằng PyTorch  
**Bài thực hành:** Practice 1 – Deep Learning Fundamentals  
**Ngày hoàn thành:** 2026-10-25  
**Trạng thái kiểm định:** Đạt chuẩn 100% (46/46 unit tests passed)

---

## 1. Tóm tắt Điều hành (Executive Summary)

Dự án đã triển khai hoàn chỉnh một hệ thống phân loại hình ảnh thời trang từ đầu đến cuối (end-to-end) trên tập dữ liệu chuẩn **FashionMNIST** sử dụng framework **PyTorch**, tuân thủ nghiêm ngặt các nguyên lý kỹ nghệ học sâu:
- Quy trình chia tập phân tầng (**Stratified Split**: 50.000 train / 10.000 validation / 10.000 test độc lập) loại trừ hoàn toàn rò rỉ dữ liệu (data leakage).
- Khảo sát và đối chứng hai kiến trúc đại diện: **BaselineMLP** (Fully Connected Network) và **SimpleCNN** (Deep Convolutional Neural Network with Global Average Pooling).
- Toàn bộ tham số thống kê chuẩn hóa ($mean = 0.2860$, $std = 0.3530$) được tính độc lập trên tập huấn luyện trước khi áp dụng cho mô hình.

### Kết quả Cốt lõi
1. **SimpleCNN vượt trội hoàn toàn về hiệu năng phân loại:** Đạt **84.75% Test Accuracy** và **0.8468 Macro F1**, cao hơn BaselineMLP (+2.68% Accuracy, +2.64% F1).
2. **Tối ưu hóa dung lượng và tham số:** SimpleCNN chỉ sử dụng **38.122 tham số**, giảm **5.3 lần** so với BaselineMLP (203.530 tham số) nhờ việc ứng dụng Global Average Pooling (GAP) thay cho lớp Linear kích thước lớn sau tầng Conv.
3. **Giảm thiểu lỗi phân loại:** Giảm từ 1.793 lỗi xuống còn 1.525 lỗi trên tập test độc lập (-268 mẫu sai).

---

## 2. Bảng Đối chứng Hiệu năng Thực nghiệm

| Chỉ số / Thuộc tính | BaselineMLP | SimpleCNN (`exp_cnn_small`) | Độ lệch ($\Delta$) | Ý nghĩa Kỹ thuật |
|---|---|---|---|---|
| **Số lượng tham số (Params)** | 203.530 | **38.122** | **-81.27% (-5.3x)** | GAP giảm thiểu kích thước mô hình, chống overfitting |
| **Validation Accuracy (Tốt nhất)** | 82.16% | **84.83%** | **+2.67%** | Tiêu chí chọn checkpoint tốt nhất (`best_model.pt`) |
| **Test Accuracy** | 82.07% | **84.75%** | **+2.68%** | Đánh giá 1 lần duy nhất trên tập Test độc lập |
| **Macro F1-score** | 0.8204 | **0.8468** | **+0.0264** | Trọng số cân bằng giữa 10 lớp |
| **Weighted F1-score** | 0.8204 | **0.8468** | **+0.0264** | Đồng nhất với Macro do test set cân bằng tuyệt đối |
| **Macro Precision** | 0.8222 | **0.8475** | **+0.0253** | Độ chính xác tổng thể các lớp |
| **Macro Recall** | 0.8207 | **0.8475** | **+0.0268** | Khả năng bao phủ các mẫu thực tế |
| **Tổng số mẫu dự đoán sai (Errors)** | 1.793 / 10.000 | **1.525 / 10.000** | **-268 lỗi** | Khả năng tổng quát hóa cao hơn |
| **Thời gian huấn luyện (CPU)** | ~192.5 s | ~829.9 s | +637.4 s | Chi phí tính toán tích chập 2D trên CPU |

---

## 3. Phân tích Chi tiết Hiệu năng Từng Lớp (Per-Class Breakdown)

Tập test bao gồm đúng 1.000 mẫu cho mỗi lớp trong số 10 lớp:

| Lớp (Class ID) | BaselineMLP F1 | SimpleCNN F1 | Biến động ($\Delta$ F1) | Nhận định Kỹ thuật |
|---|---|---|---|---|
| **0: T-shirt/top** | 78.86% | **79.12%** | +0.26% | Khó phân biệt với Shirt |
| **1: Trouser** | 96.53% | **97.05%** | +0.52% | Biên dạng ống dài thẳng, đặc trưng rất riêng biệt |
| **2: Pullover** | 68.44% | **79.20%** | **+10.76%** | **Cải thiện lớn nhất:** CNN học được họa tiết sợi/cổ áo dày |
| **3: Dress** | 84.79% | 82.02% | -2.77% | Nhầm lẫn một phần sang T-shirt/top do thân váy ngắn |
| **4: Coat** | 74.09% | 73.67% | -0.42% | Khó phân biệt giữa áo khoác dài và áo len dài tay |
| **5: Sandal** | 90.56% | **94.42%** | **+3.86%** | Cấu trúc quai hở được trích xuất tốt bởi bộ lọc Conv |
| **6: Shirt** | 54.43% | **58.10%** | **+3.67%** | Lớp khó nhất: cấu trúc cổ áo bị mờ ở kích thước 28x28 |
| **7: Sneaker** | 87.81% | **93.04%** | **+5.23%** | CNN nhận diện tốt dáng đế giày thể thao |
| **8: Bag** | 93.75% | **96.04%** | **+2.29%** | Quai xách và khối vuông đặc thù |
| **9: Ankle boot** | 91.15% | **94.16%** | **+3.01%** | Cổ cao và đế ủng nhận diện ổn định |

---

## 4. Phân tích Sai số & Các Cặp Lớp Nhầm Lẫn (Error Analysis)

Dựa trên ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) của mô hình `SimpleCNN`, 5 cặp nhầm lẫn phổ biến nhất gồm:

1. **`Shirt` → `T-shirt/top` (176 mẫu nhầm):**  
   *Nguyên nhân:* Ở độ phân giải cực thấp ($28 \times 28$ grayscale), các chi tiết như hàng cúc, túi ngực nhỏ hoặc cổ áo sơ mi dễ bị triệt tiêu sau phép thu nhỏ và làm mờ (resampling), khiến ảnh giống hệt áo phông cộc tay.
2. **`Coat` → `Shirt` (124 mẫu nhầm):**  
   *Nguyên nhân:* Vạt áo khoác mở tạo nếp gấp dọc ngực tương tự cổ chữ V hoặc hàng khuy sơ mi.
3. **`Pullover` → `Shirt` (106 mẫu nhầm):**  
   *Nguyên nhân:* Áo len chui đầu dài tay có tỷ lệ khung hình và chiều dài ống tay tương đồng với áo sơ mi dài tay.
4. **`Shirt` → `Coat` (105 mẫu nhầm) & `Coat` → `Pullover` (105 mẫu nhầm):**  
   *Nguyên nhân:* Dữ liệu ảnh đơn sắc (grayscale 1 kênh) làm mất hoàn toàn thông tin về chất liệu vải (vải dù/da của áo khoác vs len dệt của áo chui đầu vs cotton của sơ mi).

---

## 5. Đánh giá Tuân thủ Toàn diện Yêu cầu Đề bài Practice 1

| Yêu cầu Đề bài | Trạng thái | Minh chứng Thực thi trong Mã nguồn |
|---|---|---|
| **1. Tensors** | ĐẠT | Sử dụng PyTorch Tensor đa chiều `(B, 1, 28, 28)`, ép kiểu `float32`, chuyển đổi thiết bị linh hoạt (CPU/CUDA). |
| **2. Datasets/Loaders** | ĐẠT | Kế thừa `torchvision.datasets.FashionMNIST`, `DataLoader` với `batch_size=128`, `shuffle=True` (train) và `shuffle=False` (val/test). |
| **3. Transforms** | ĐẠT | Áp dụng `T.ToTensor()`, chuẩn hóa Z-score với mean/std thực nghiệm, augment `RandomHorizontalFlip` và `RandomCrop(padding=4)` chỉ trên tập train. |
| **4. Model Building** | ĐẠT | Triển khai kế thừa `nn.Module` cho cả `BaselineMLP` và `SimpleCNN`. Kiến trúc rõ ràng, không cài Softmax thừa ở ngõ ra. |
| **5. Autograd** | ĐẠT | `loss.backward()` tự động tính đạo hàm trong pha huấn luyện; bao bọc `torch.no_grad()` chặt chẽ trong pha thẩm định và đánh giá. |
| **6. Optimization** | ĐẠT | Sử dụng `torch.optim.Adam` (weight decay $10^{-4}$) kết hợp bộ điều chỉnh tốc độ học `CosineAnnealingLR` và cơ chế Early Stopping (patience=5). |
| **7. Model Saving/Loading** | ĐẠT | Module `checkpoint.py` lưu và phục hồi đầy đủ `model_state_dict`, `optimizer_state_dict`, epoch, và metrics tương ứng vào file `.pt`. |
| **8. Vòng lặp Huấn luyện** | ĐẠT | Chu trình 4 bước chuẩn xác: **Forward** $\rightarrow$ **Loss** (`CrossEntropyLoss`) $\rightarrow$ **Backward** $\rightarrow$ **Optimize** (`optimizer.step()`). |

---

## 6. Kết luận & Hướng Phát triển Khả thi

### Kết luận
Dự án đã chứng minh thực nghiệm một nguyên lý cốt lõi của thị giác máy tính: **Mạng nơ-ron tích chập (CNN) với tính chất chia sẻ trọng số (weight sharing) và bất biến tịnh tiến (translation invariance) vượt trội hơn hẳn mạng truyền thẳng (MLP)** trên dữ liệu dạng lưới 2D, vừa đạt độ chính xác cao hơn (+2.68%), vừa tiết kiệm tài nguyên bộ nhớ (-81.27% số tham số).

### Hướng cải tiến có căn cứ
1. **Label Smoothing Loss:** Giúp giảm thiểu sự tự tin thái quá (over-confidence) của mô hình trên các lớp dễ nhầm như Shirt, Pullover và Coat.
2. **Kỹ thuật Augmentation Nâng cao (Mixup / CutMix):** Huấn luyện mô hình phân biệt ranh giới mờ giữa các kiểu dáng áo thân trên.
3. **Mở rộng Kiến trúc Residual (ResNet-18):** Thêm các khối Residual Connections nếu muốn đẩy độ chính xác vượt mốc 92-93%.
