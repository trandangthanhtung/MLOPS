from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --- Tải mô hình và scaler ---
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
target_names = ["setosa", "versicolor", "virginica"]

# --- Khởi tạo FastAPI ---
app = FastAPI(
    title="Dự đoán Loài Hoa Iris ",
    description="Ứng dụng FastAPI sử dụng mô hình Logistic Regression để dự đoán loại hoa Iris dựa trên 4 đặc trưng hình thái.",
    version="1.0.0",
)

# --- Lớp mô tả dữ liệu đầu vào ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- Endpoint 1: Trang mô tả thông tin ---
@app.get("/")
def home():
    """
    Trả về thông tin tổng quan về dự án và hướng dẫn sử dụng API.
    """
    return {
        "project": "Phân loại hoa Iris bằng Logistic Regression",
        "author": "Trần Thanh Tùng - K16",
        "supervisor": "ThS. Phạm Xuân Trí",
        "description": (
            "Ứng dụng FastAPI cho phép dự đoán loài hoa Iris dựa trên 4 đặc trưng hình thái: "
            "chiều dài và chiều rộng của đài hoa (sepal), cánh hoa (petal)."
        ),
        "model": "Logistic Regression (Scikit-learn)",
        "features": [
            "sepal_length (chiều dài đài hoa)",
            "sepal_width (chiều rộng đài hoa)",
            "petal_length (chiều dài cánh hoa)",
            "petal_width (chiều rộng cánh hoa)"
        ],
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "example_input": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            "note": "Truy cập /docs để thử dự đoán trực tiếp bằng giao diện Swagger UI."
        }
    }


# --- Endpoint 2: Dự đoán ---
@app.post("/predict")
def predict(data: IrisInput):
    """
    Dự đoán loài hoa Iris từ 4 đặc trưng hình thái.
    """
    # Chuyển dữ liệu đầu vào thành mảng numpy
    X_input = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    # Chuẩn hóa dữ liệu (theo scaler đã lưu)
    X_scaled = scaler.transform(X_input)

    # Dự đoán kết quả
    pred_idx = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    # Ghi log vào file
    with open("predict_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Input: {data.dict()} → Predicted: {target_names[pred_idx]}\n")

    # Trả kết quả
    return {
        "input_data": data.dict(),
        "prediction": {
            "predicted_class": target_names[pred_idx],
            "probabilities": {
                target_names[i]: round(float(probs[i]), 4)
                for i in range(len(target_names))
            }
        },
        "message": "Dự đoán thành công "
    }
