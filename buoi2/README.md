📊 Adult Income Classification
🎯 1. Mục tiêu bài tập

Bài tập này yêu cầu xây dựng các mô hình học máy để phân loại thu nhập của người dân dựa trên tập dữ liệu Adult Income (UCI Machine Learning Repository).

Mục tiêu: phân loại Income thành hai nhãn:

>50K 💰 (thu nhập cao)

<=50K 💵 (thu nhập thấp)

📋 2. Yêu cầu
🧹 Tiền xử lý dữ liệu

Làm sạch dữ liệu, xử lý giá trị thiếu và loại bỏ nhiễu.

Mã hóa các biến phân loại:

Label Encoding 🔤 hoặc One-Hot Encoding 🎨

Chuẩn hóa các biến số:

StandardScaler ⚖️ hoặc MinMaxScaler 📏

🤖 Xây dựng ít nhất 3 mô hình học máy

Logistic Regression

Decision Tree 🌳

Random Forest 🌲

Có thể thêm mô hình nâng cao:

Gradient Boosting ⚡

AdaBoost 🚀

📊 Đánh giá mô hình

Trong hai trường hợp:

Không tiền xử lý dữ liệu (baseline) ❌

Đã tiền xử lý dữ liệu (clean + encode + scale) ✅

Các thước đo đánh giá:

Accuracy ✅

Precision 🎯

Recall 🔄

F1-score ⚖️

Confusion Matrix 🧩

📝 Nhận xét và phân tích

Ảnh hưởng của tiền xử lý dữ liệu đến hiệu năng mô hình.

Mô hình nào cho kết quả tốt nhất và lý do.

🛠️ 3. Quy trình thực hiện
3.1. Chuẩn bị dữ liệu

Tải các file: adult.data, adult.test, adult.names từ UCI ML Repository

Đặt tất cả file cùng thư mục với script Python.

3.2. Tiền xử lý dữ liệu (chỉ áp dụng trong file bai1.py)

Làm sạch dữ liệu:

Loại bỏ giá trị missing (?) hoặc điền giá trị thiếu:

Biến phân loại: điền bằng mode 🔤

Biến số: điền bằng median 📏

Mã hóa các biến:

Categorical: One-Hot Encoding 🎨

Target income: LabelEncoder

Chuẩn hóa các biến số (StandardScaler ⚖️)

3.3. Xây dựng mô hình

Sử dụng 5 mô hình:

Logistic Regression

Decision Tree 🌳

Random Forest 🌲

Gradient Boosting ⚡

AdaBoost 🚀

Chia dữ liệu train/test:

80% train, 20% test

stratify theo nhãn

3.4. Đánh giá mô hình

Thước đo: accuracy_score ✅, precision_score 🎯, recall_score 🔄, f1_score ⚖️

Ma trận nhầm lẫn: confusion_matrix 🧩

So sánh trước và sau tiền xử lý.

Lưu kết quả:

CSV tổng hợp: results_no_preprocess.csv & results_preprocess.csv

Ma trận nhầm lẫn: *_cm_baseline.csv & *_cm_processed.csv

Biểu đồ F1-score: f1_comparison.png 📊

3.5. Lưu mô hình

Mỗi mô hình được lưu trong thư mục:

Không tiền xử lý: models/no_preprocess

Có tiền xử lý: models/preprocess

Định dạng: joblib 💾

3.6. Nhận xét chung

Tiền xử lý dữ liệu giúp:

Logistic Regression cải thiện rõ rệt, F1-score tăng mạnh 📈

Các mô hình ensemble (Random Forest, Gradient Boosting) tăng nhẹ Accuracy và F1

Gradient Boosting ⚡ thường là mô hình tốt nhất sau tiền xử lý

Không tiền xử lý:

Mô hình tuyến tính bị giảm hiệu năng do dữ liệu categorical chưa encode đúng ❌

Accuracy thấp hơn và F1-score giảm, đặc biệt với lớp >50K 💸

▶️ 4. Hướng dẫn chạy
4.1. Mô hình không tiền xử lý
conda activate NlP
python bai1.1.py

4.2. Mô hình có tiền xử lý
conda activate NlP
python bai1.py

4.3. So sánh kết quả

Mở CSV:

outputs/results_no_preprocess.csv

outputs/results_preprocess.csv

📂 5. Thư mục kết quả
buoi2/
│
├─ adult.data
├─ adult.test
├─ adult.names
├─ bai1.1.py             # Không tiền xử lý
├─ bai1.py               # Có tiền xử lý
├─ sosanh.py             # So sánh (tùy chọn)
│
├─ outputs/
│   ├─ results_no_preprocess.csv
│   ├─ results_preprocess.csv
│   ├─ *_cm_baseline.csv
│   ├─ *_cm_processed.csv
│   └─ f1_comparison.png 📊
│
└─ models/
    ├─ no_preprocess/
    └─ preprocess/


📊 6. So sánh kết quả mô hình (Baseline vs Preprocessed)
Model	Accuracy ❌ NoPreprocess	F1 ❌ NoPreprocess	Accuracy ✅ Preprocess	F1 ✅ Preprocess
Logistic Regression	0.7873	0.4168	0.8560	0.6745 📈
Decision Tree 🌳	0.8050	0.5991	0.8130	0.6210 📈
Random Forest 🌲	0.8558	0.6765	0.8572	0.6830 📈
Gradient Boosting ⚡	0.8704	0.7020	0.8727	0.7088 📈
AdaBoost 🚀	0.8560	0.6695	0.8589	0.6738 📈

Nhận xét :

🔹 Tất cả mô hình đều cải thiện Accuracy và F1-score sau khi tiền xử lý.

🔹 Logistic Regression được hưởng lợi nhiều nhất, F1-score tăng đáng kể.

🔹 Gradient Boosting ⚡ vẫn là mô hình tốt nhất về cả Accuracy và F1-score sau tiền xử lý.

🔹 Các mô hình ensemble (Random Forest 🌲, AdaBoost 🚀) cải thiện nhẹ, nhưng ổn định.



📌 Bài 2 – Tăng cường dữ liệu ảnh (Image Data Augmentation) trong huấn luyện mô hình học sâu
🎯 Mục tiêu

Sử dụng 5000 ảnh từ tập huấn luyện CIFAR-10 (5 lớp × 1000 ảnh mỗi lớp) để huấn luyện mô hình phân loại.

Thực hiện Data Augmentation: lật ảnh, xoay, dịch chuyển, zoom, cắt ngẫu nhiên, thay đổi độ sáng/độ tương phản.

Huấn luyện mô hình hai trường hợp: dữ liệu gốc ✅ và dữ liệu đã tăng cường 📈.

So sánh hiệu năng (accuracy, loss, tốc độ hội tụ).

Mỗi cấu hình mô hình chạy 3 lần → lấy kết quả trung bình.

🗂️ Cấu trúc dự án
├── bai2.1.py                  # Chạy mô hình trên dữ liệu gốc (Original)
├── bai2.py                     # Phiên bản tăng cường ban đầu (có thể lỗi ảnh đen)
├── train_augmented_fixed.py    # Phiên bản tăng cường dữ liệu đã fix lỗi ảnh đen
├── old.adult.names             # File tham khảo
├── original_accuracy_curve.png # Biểu đồ accuracy mô hình dữ liệu gốc
├── original_loss_curve.png     # Biểu đồ loss mô hình dữ liệu gốc
├── original_images_examples.png# Ví dụ ảnh gốc
├── result_original.csv          # Kết quả huấn luyện dữ liệu gốc (CSV)
├── result_original.xlsx         # Kết quả huấn luyện dữ liệu gốc (Excel)
├── results_augmented_fixed.csv  # Kết quả huấn luyện dữ liệu tăng cường (CSV)
├── results_augmented_fixed.xlsx # Kết quả huấn luyện dữ liệu tăng cường (Excel)
└── README.md                   # Hướng dẫn

🏃‍♂️ Hướng dẫn chạy
1️⃣ bai2.1.py – Dữ liệu gốc

Mục đích: Huấn luyện CNN trên dữ liệu gốc 5000 ảnh.

Hoạt động:

Chọn 5 lớp × 1000 ảnh.

Chuẩn hóa ảnh (0-1).

Xây dựng CNN.

Huấn luyện 3 lần.

Lưu kết quả trung bình → result_original.csv / result_original.xlsx.

Vẽ và lưu biểu đồ accuracy/loss → original_accuracy_curve.png, original_loss_curve.png.

Hiển thị một số ảnh gốc → original_images_examples.png.

2️⃣ train_augmented_fixed.py – Dữ liệu tăng cường

Mục đích: Huấn luyện CNN trên dữ liệu tăng cường, không còn lỗi ảnh đen.

Hoạt động:

Chọn 5 lớp × 1000 ảnh.

Chuẩn hóa ảnh (0-1).

Áp dụng ImageDataGenerator: rotation, flip, shift, zoom, shear.

Điều chỉnh brightness/contrast an toàn bằng numpy.

Hiển thị một số ảnh trước và sau khi tăng cường 🖼️.

Tạo dataset tăng cường đầy đủ (có thể nhân đôi).

Xây dựng CNN giống dữ liệu gốc.

Huấn luyện 3 lần, lưu history.

Lưu kết quả → results_augmented_fixed.csv / results_augmented_fixed.xlsx.

Vẽ biểu đồ accuracy/loss từng run 📊.

3️⃣ bai2.py

Phiên bản cũ, có thể lỗi ảnh đen do brightness/contrast trực tiếp.

Được thay thế bởi train_augmented_fixed.py. ⚠️

💾 Output / Kết quả
📄 File kết quả	Nội dung
result_original.csv/xlsx	Kết quả huấn luyện dữ liệu gốc (3 lần + trung bình)
results_augmented_fixed.csv/xlsx	Kết quả huấn luyện dữ liệu tăng cường (3 lần + trung bình)
original_images_examples.png	Một số ảnh gốc trước huấn luyện 🖼️
augmented_examples.png	Một số ảnh sau tăng cường 🖼️
original_accuracy_curve.png	Accuracy train/val dữ liệu gốc 📊
original_loss_curve.png	Loss train/val dữ liệu gốc 📊
aug_train_accuracy_allruns.png	Accuracy train trên dữ liệu tăng cường 📊
aug_val_accuracy_allruns.png	Accuracy validation trên dữ liệu tăng cường 📊
augmented_loss_curve.png	Loss trên dữ liệu tăng cường 📊
📊 So sánh hiệu năng

Mở CSV/Excel → so sánh val_accuracy và val_loss giữa dữ liệu gốc ✅ và tăng cường 📈

Quan sát tốc độ hội tụ và độ chính xác cuối cùng.

Thường thấy dữ liệu tăng cường cải thiện generalization.

⚠️ Lưu ý

Mỗi file .py chạy độc lập trên VSCode / PyCharm.

Cài đặt trước khi chạy:

pip install tensorflow numpy matplotlib pandas openpyxl


Phiên bản TensorFlow ≥ 2.10 để tránh lỗi với ImageDataGenerator.

📝 Hướng dẫn sử dụng

Chạy bai2.1.py → dữ liệu gốc

Chạy train_augmented_fixed.py → dữ liệu tăng cường

Mở CSV/Excel → so sánh accuracy/loss trung bình

Xem ảnh minh họa và biểu đồ → đánh giá hiệu quả Data Augmentation