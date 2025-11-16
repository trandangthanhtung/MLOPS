<<<<<<< HEAD
# MLOPS
=======
# Bài Thực Hành Buổi 1 — Phân loại hoa Iris bằng Logistic Regression và FastAPI

## Thông tin chung
- **Sinh viên thực hiện:** Trần Thanh Tùng - K16  
- **Giảng viên hướng dẫn:** ThS. Phạm Xuân Trí  
- **Môn học:**  MLOps (Bài thực hành Buổi 1)  
- **Mục tiêu:**  
  - Ôn tập quy trình thực hiện một bài toán Machine Learning.  
  - Làm quen với FastAPI để triển khai mô hình dưới dạng API.

---

## Mô tả bài tập (Yêu cầu)
1. **Xây dựng và huấn luyện một mô hình máy học**, thể hiện đầy đủ các bước:
   - a. Chuẩn bị dữ liệu  
   - b. Xử lý và chọn đặc trưng (tùy chọn)  
   - c. Xây dựng và huấn luyện mô hình  
   - d. Đánh giá và tối ưu mô hình  
   - e. Xây dựng chương trình demo

2. **Triển khai mô hình bằng FastAPI**:
   - a. Cài đặt và cấu hình FastAPI, Uvicorn  
   - b. Ứng dụng FastAPI gồm:
     - Endpoint `/` : hiển thị thông tin mô tả về mô hình.
     - Endpoint `/predict` : nhận dữ liệu đầu vào (JSON) và trả kết quả dự đoán.
     - Định nghĩa class `BaseModel` (từ `pydantic`) để mô tả input dữ liệu.


3. **cấu trúc thư mục**: 
     📁 MLOPS/
│
├── 📄 buoi1.py                # Huấn luyện & lưu model + scaler
├── 📄 api.py                  # FastAPI app dùng model đã lưu
├── 📄 iris_model.pkl          # Model Logistic Regression đã lưu
├── 📄 iris_scaler.pkl         # Scaler để chuẩn hóa dữ liệu khi dự đoán
├── 📄 requirements.txt        # Danh sách thư viện cần cài đặt

└── 📄 README.md               # Mô tả bài thực hành, quy trình, hướng dẫn chạy

4. **🚀 HƯỚNG DẪN CHẠY MODEL “Phân loại hoa Iris bằng FastAPI”**:

💡 Yêu cầu trước khi chạy

Bạn cần có môi trường Python (>=3.8) và cài các thư viện sau.
Nếu dùng Conda:

conda activate NLP

🧩 Bước 1 — Cài đặt thư viện cần thiết

Trong thư mục dự án (MLOPS), tạo file requirements.txt với nội dung:

fastapi
uvicorn
scikit-learn
joblib
pydantic


Rồi chạy:

pip install -r requirements.txt

⚙️ Bước 2 — Huấn luyện mô hình và lưu lại

Chạy file buoi1.py để huấn luyện Logistic Regression và lưu model + scaler.

python buoi1.py


Kết quả (ví dụ):

Train acc: 0.97
Test acc : 1.00
Mô hình LogisticRegression đã lưu thành công vào iris_model.pkl


Sau khi chạy xong, thư mục của bạn sẽ có thêm:

iris_model.pkl
iris_scaler.pkl

🌐 Bước 3 — Chạy FastAPI

Chạy lệnh:

uvicorn api:app --reload


Nếu hiện dòng như sau là thành công ✅:

INFO:     Uvicorn running on http://127.0.0.1:8000

📄 Bước 4 — Kiểm tra API
🔹 Truy cập trình duyệt:

👉 http://127.0.0.1:8000/

→ sẽ thấy JSON mô tả mô hình:

{
  "project": "Phân loại hoa Iris",
  "author": "Trần Thanh Tùng - K16",
  "lecturer": "ThS. Phạm Xuân Trí",
  "model": "Logistic Regression",
  "description": "Ứng dụng FastAPI dự đoán loại hoa Iris dựa trên 4 đặc trưng hình thái.",
  "usage": "Gửi dữ liệu JSON tới /predict để nhận kết quả dự đoán."
}

🔹 Bước 5 — Gửi dữ liệu để dự đoán

Vào trang tương tác API của FastAPI:
👉 http://127.0.0.1:8000/docs

Chọn POST /predict

Bấm “Try it out”

Nhập ví dụ:

{
  "sepal_length": 5.9,
  "sepal_width": 3.0,
  "petal_length": 5.1,
  "petal_width": 1.8
}


Bấm Execute

✅ Kết quả trả về (ví dụ):
{
  "predicted_class": "virginica",
  "probabilities": {
    "setosa": 0.002,
    "versicolor": 0.067,
    "virginica": 0.931
  }
}

Buổi 2 – Tiền xử lý dữ liệu & Xây dựng mô hình phân loại thu nhập (Adult Income Classification)
🎯 1. Mục tiêu bài tập

Bài tập yêu cầu xây dựng các mô hình học máy để phân loại thu nhập của người dân dựa trên tập dữ liệu Adult Income – UCI Machine Learning Repository.

Mục tiêu: dự đoán nhãn thu nhập:

>50K → Thu nhập cao 💰

<=50K → Thu nhập thấp 👤

🧩 2. Yêu cầu
✅ Tiền xử lý dữ liệu

Làm sạch dữ liệu, xử lý giá trị thiếu và nhiễu.

Mã hóa biến phân loại:

Label Encoding hoặc One-Hot Encoding.

Chuẩn hóa các biến số:

StandardScaler hoặc MinMaxScaler.

🤖 Xây dựng tối thiểu 3 mô hình học máy

Logistic Regression

Decision Tree

Random Forest
(Có thể thêm: Gradient Boosting, AdaBoost)

🔬 Đánh giá mô hình trong 2 trường hợp

Không tiền xử lý dữ liệu – Baseline ❌

Có tiền xử lý đầy đủ – Preprocessed ✅

📏 Các thước đo đánh giá

Accuracy

Precision

Recall

F1-score

Confusion Matrix

📝 Nhận xét

Ảnh hưởng của tiền xử lý đến hiệu năng mô hình.

Mô hình tốt nhất và giải thích lý do.

📚 3. Quy trình thực hiện
🔎 3.1. Chuẩn bị dữ liệu

Tải file:

adult.data

adult.test

adult.names

Đặt chung thư mục với script Python.

🧼 3.2. Tiền xử lý dữ liệu (áp dụng trong file: bai1.py)
✔ Làm sạch dữ liệu

Thay ? thành NaN

Điền giá trị thiếu:

Biến phân loại → mode

Biến số → median

✔ Mã hóa

Categorical → One-Hot Encoding

Target → LabelEncoder

✔ Chuẩn hóa

StandardScaler cho các biến số

🏗 3.3. Xây dựng mô hình

Sử dụng 5 mô hình:

Logistic Regression

Decision Tree 🌳

Random Forest 🌲

Gradient Boosting ⚡

AdaBoost 🚀

Chia dữ liệu:

train 80% – test 20%, stratify theo nhãn

📊 3.4. Đánh giá mô hình

Dùng các metric:

accuracy_score

precision_score

recall_score

f1_score

confusion_matrix

classification_report

So sánh:

results_no_preprocess.csv

results_preprocess.csv

Lưu ma trận nhầm lẫn & biểu đồ F1.

💾 3.5. Lưu mô hình

Không tiền xử lý → models/no_preprocess/

Có tiền xử lý → models/preprocess/

Định dạng: .joblib

📊 4. So sánh kết quả mô hình (Baseline ❌ vs Preprocessed ✅)
Model	Accuracy ❌ NoPre	F1 ❌ NoPre	Accuracy ✅ Pre	F1 ✅ Pre
Logistic Regression	0.7873	0.4168	0.8560	0.6745 📈
Decision Tree 🌳	0.8050	0.5991	0.8130	0.6210 📈
Random Forest 🌲	0.8558	0.6765	0.8572	0.6830 📈
Gradient Boosting ⚡	0.8704	0.7020	0.8727	0.7088 📈
AdaBoost 🚀	0.8560	0.6695	0.8589	0.6738 📈
📝 Nhận xét nhanh

Tiền xử lý giúp tăng Accuracy & F1 cho tất cả mô hình.

Logistic Regression tăng mạnh nhờ dữ liệu đã được chuẩn hóa + one-hot.

Gradient Boosting ⚡ là mô hình hoạt động tốt nhất.

Các mô hình ensemble tăng nhẹ nhưng ổn định.

🖥 5. Hướng dẫn chạy
🔹 Chạy mô hình Không tiền xử lý
conda activate NlP
python bai1.1.py

🔹 Chạy mô hình Có tiền xử lý
conda activate NlP
python bai1.py

🔹 So sánh kết quả
python sosanh.py

📁 6. Cấu trúc thư mục
buoi2/
│
├─ adult.data
├─ adult.test
├─ adult.names
├─ bai1.1.py              # Không tiền xử lý
├─ bai1.py                # Có tiền xử lý
├─ sosanh.py              # So sánh mô hình
│
├─ outputs/
│   ├─ results_no_preprocess.csv
│   ├─ results_preprocess.csv
│   ├─ *_cm_baseline.csv
│   ├─ *_cm_processed.csv
│   └─ f1_comparison.png
│
└─ models/
    ├─ no_preprocess/
    └─ preprocess/ 


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