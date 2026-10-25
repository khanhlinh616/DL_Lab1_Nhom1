# 02 — Dataset và EDA

## Nguồn Dữ liệu

- **Dataset:** FashionMNIST (Zalando Research)
- **Repository:** https://github.com/zalandoresearch/fashion-mnist
- **Tải qua:** `torchvision.datasets.FashionMNIST(download=True)`
- **Lưu trữ:** `data/raw/FashionMNIST/`

## Đặc điểm Dataset

| Thuộc tính | Giá trị |
|---|---|
| Train samples | 60 000 |
| Test samples | 10 000 |
| Image shape | 1 × 28 × 28 (C × H × W) |
| Dtype raw | uint8 [0, 255] |
| Dtype tensor | float32 [0.0, 1.0] |
| Số lớp | 10 |
| Phân bố | Cân bằng: 6 000 train / 1 000 test mỗi lớp |

**10 lớp (theo thứ tự nhãn 0–9):**  
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Phương pháp EDA

EDA được thực hiện **ngay sau khi tải dữ liệu, trước khi chia tập**.  
Module: `src/fashionmnist/data/eda.py`  
Script: `scripts/download_and_eda.py`

**Trình tự:**
1. Tải raw dataset (ToTensor, không normalize)
2. Kiểm tra integrity (kích thước, nhãn, giá trị pixel)
3. Tính thống kê pixel toàn bộ train set
4. Trực quan hóa phân bố lớp, ảnh mẫu, histogram pixel, ảnh trung bình mỗi lớp
5. Lưu kết quả vào `artifacts/eda/` và `artifacts/figures/`

## Kết quả EDA (sau khi chạy `download_and_eda.py`)

- **Pixel mean (train):** ~0.2860 — ảnh chủ yếu tối (background đen)
- **Pixel std (train):** ~0.3530 — độ tương phản cao
- **Phân bố lớp:** hoàn toàn cân bằng, không cần xử lý class imbalance
- **Invalid labels:** 0 (cả train và test)
- **Nhận xét:** Shirt và T-shirt/top có ngoại hình tương tự -> dễ nhầm lẫn; Trouser và Bag rất đặc trưng -> dự kiến F1 cao

**Artifacts tạo ra:**
- `artifacts/eda/eda_statistics.json`
- `artifacts/figures/eda_raw_class_distribution.png`
- `artifacts/figures/dataset_split_distribution.png`
- `artifacts/figures/eda_class_distribution.png`
- `artifacts/figures/eda_sample_grid.png`
- `artifacts/figures/eda_pixel_histogram.png`
- `artifacts/figures/eda_class_mean_images.png`
