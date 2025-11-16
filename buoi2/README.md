📊 Adult Income Classification & CIFAR-10 Data Augmentation
1. Mục tiêu bài tập
Adult Income Classification

Xây dựng các mô hình học máy để phân loại thu nhập người dân dựa trên tập dữ liệu Adult Income (UCI Machine Learning Repository).

Mục tiêu phân loại:

>50K 💰 – Thu nhập cao

<=50K 💵 – Thu nhập thấp

CIFAR-10 Image Data Augmentation

Sử dụng 5000 ảnh từ CIFAR-10 để huấn luyện CNN, áp dụng Data Augmentation nhằm cải thiện generalization.

2. Tiền xử lý dữ liệu (Adult Income)

Làm sạch dữ liệu: loại bỏ giá trị thiếu (?) hoặc điền giá trị thiếu

Biến phân loại: điền bằng mode

Biến số: điền bằng median

Mã hóa biến phân loại:

One-Hot Encoding 🎨

Target: LabelEncoder

Chuẩn hóa biến số:

StandardScaler ⚖️ hoặc MinMaxScaler 📏

3. Mô hình học máy
Các mô hình cơ bản:

Logistic Regression

Decision Tree 🌳

Random Forest 🌲

Mô hình nâng cao:

Gradient Boosting ⚡

AdaBoost 🚀

Chia dữ liệu:

80% train, 20% test

Stratify theo nhãn

Đánh giá mô hình:

Accuracy ✅

Precision 🎯

Recall 🔄

F1-score ⚖️

Confusion Matrix 🧩

So sánh:

Không tiền xử lý ❌

Có tiền xử lý ✅

Lưu kết quả:

CSV: results_no_preprocess.csv & results_preprocess.csv

Ma trận nhầm lẫn: *_cm_baseline.csv & *_cm_processed.csv

Biểu đồ F1-score: f1_comparison.png

Lưu mô hình:

Không tiền xử lý: models/no_preprocess

Có tiền xử lý: models/preprocess

4. Nhận xét mô hình

Tất cả mô hình cải thiện Accuracy & F1-score sau tiền xử lý

Logistic Regression tăng đáng kể F1-score 📈

Gradient Boosting ⚡ thường là mô hình tốt nhất

Ensemble models (Random Forest, AdaBoost) tăng nhẹ nhưng ổn định

5. CIFAR-10 Data Augmentation

Mục tiêu:

Huấn luyện CNN trên dữ liệu gốc & dữ liệu tăng cường

Quan sát hiệu năng (accuracy, loss, tốc độ hội tụ)

Data Augmentation áp dụng:

Lật, xoay, dịch chuyển, zoom, crop ngẫu nhiên

Điều chỉnh độ sáng/độ tương phản

Cấu trúc thư mục & file:

buoi2/
├─ adult.data
├─ adult.test
├─ adult.names
├─ bai1.1.py       # Không tiền xử lý
├─ bai1.py         # Có tiền xử lý
├─ sosanh.py       # So sánh (tùy chọn)
├─ outputs/
│  ├─ results_no_preprocess.csv
│  ├─ results_preprocess.csv
│  ├─ *_cm_baseline.csv
│  ├─ *_cm_processed.csv
│  └─ f1_comparison.png
├─ models/
│  ├─ no_preprocess/
│  └─ preprocess/
├─ bai2.1.py           # Dữ liệu gốc
├─ train_augmented_fixed.py  # Dữ liệu tăng cường đã fix lỗi
├─ old.adult.names
├─ original_accuracy_curve.png
├─ original_loss_curve.png
├─ original_images_examples.png
├─ result_original.csv/xlsx
├─ results_augmented_fixed.csv/xlsx
└─ README.md

6. Hướng dẫn chạy
Adult Income
# Không tiền xử lý
conda activate NlP
python bai1.1.py

# Có tiền xử lý
python bai1.py

# So sánh kết quả
python sosanh.py

CIFAR-10
# Dữ liệu gốc
python bai2.1.py

# Dữ liệu tăng cường đã fix
python train_augmented_fixed.py


Lưu ý:

Cài đặt trước: pip install tensorflow numpy matplotlib pandas openpyxl

TensorFlow ≥ 2.10 để tránh lỗi ImageDataGenerator

7. Kết quả mô hình Adult Income
Model	Accuracy ❌	F1 ❌	Accuracy ✅	F1 ✅
Logistic Regression	0.7873	0.4168	0.8560	0.6745
Decision Tree 🌳	0.8050	0.5991	0.8130	0.6210
Random Forest 🌲	0.8558	0.6765	0.8572	0.6830
Gradient Boosting ⚡	0.8704	0.7020	0.8727	0.7088
AdaBoost 🚀	0.8560	0.6695	0.8589	0.6738

Nhận xét:

Tiền xử lý cải thiện toàn diện Accuracy & F1

Logistic Regression cải thiện mạnh nhất

Gradient Boosting vẫn tốt nhất

Ensemble models ổn định, cải thiện nhẹ