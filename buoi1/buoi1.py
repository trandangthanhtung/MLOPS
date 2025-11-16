# ============================================
# Bài thực hành Buổi 1 - Machine Learning & FastAPI
# Đề tài: Phân loại hoa Iris bằng Logistic Regression
# Sinh viên: Trần Thanh Tùng - K16
# Giảng viên hướng dẫn: ThS. Phạm Xuân Trí
# ============================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

# --- 1. Chuẩn bị dữ liệu ---
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(" Dữ liệu Iris có", X.shape[0], "mẫu và", X.shape[1], "đặc trưng.")
print(" Các lớp:", target_names)

# --- 2. Chia dữ liệu Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Chuẩn hóa dữ liệu ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Xây dựng và huấn luyện mô hình ---
model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto')
model.fit(X_train_scaled, y_train)

# --- 5. Đánh giá mô hình ---
y_pred = model.predict(X_test_scaled)

train_acc = model.score(X_train_scaled, y_train)
test_acc = accuracy_score(y_test, y_pred)

print("\n=== Kết quả đánh giá ===")
print(f"Độ chính xác Train: {train_acc:.3f}")
print(f"Độ chính xác Test : {test_acc:.3f}")
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, target_names=target_names))
print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))

# --- 6. Lưu mô hình và scaler ---
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")

print("\n Đã lưu mô hình vào 'iris_model.pkl' và scaler vào 'iris_scaler.pkl'")
