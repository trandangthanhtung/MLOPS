from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Air Quality Prediction API")

MODEL_PATH = "/app/models/best_model.pkl"
# Biến toàn cục để giữ model và thời gian chỉnh sửa file cuối cùng
current_model = None
last_model_time = 0

def load_model_if_updated():
    global current_model, last_model_time
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        # Nếu file mới hơn bản trong RAM hoặc chưa load lần nào
        if mtime > last_model_time:
            try:
                current_model = joblib.load(MODEL_PATH)
                last_model_time = mtime
                print(f"🔄 Đã cập nhật model mới nhất (Cập nhật lúc: {mtime})")
            except Exception as e:
                print(f"❌ Lỗi khi load model: {e}")
    return current_model

# ======================
# Schema & Logic Đánh Giá
# ======================
class AirQualityInput(BaseModel):
    PT08_S1_CO: float
    NMHC_GT: float
    C6H6_GT: float
    PT08_S2_NMHC: float
    NOx_GT: float
    PT08_S3_NOx: float
    NO2_GT: float
    PT08_S4_NO2: float
    PT08_S5_O3: float
    T: float
    RH: float
    AH: float

def assess_air_quality(co_value: float):
    if co_value < 1.5: return "Tốt", "An toàn"
    elif co_value < 3.0: return "Bình thường", "An toàn"
    elif co_value < 5.0: return "Ô nhiễm nhẹ", "Cảnh báo"
    elif co_value < 10.0: return "Ô nhiễm nặng", "Nguy hiểm"
    else: return "Rất nguy hiểm", "Khẩn cấp"

# ======================
# API Endpoints
# ======================
@app.get("/")
def root():
    model_status = load_model_if_updated()
    return {
        "message": "Air Quality Prediction API",
        "model_loaded": model_status is not None,
        "last_update": last_model_time
    }

@app.post("/predict")
def predict(data: AirQualityInput):
    # Kiểm tra và load model mới nhất trước khi dự đoán
    model_to_use = load_model_if_updated()
    
    if model_to_use is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Hãy chạy training trước!")

    X = np.array([[
        data.PT08_S1_CO, data.NMHC_GT, data.C6H6_GT, data.PT08_S2_NMHC,
        data.NOx_GT, data.PT08_S3_NOx, data.NO2_GT, data.PT08_S4_NO2,
        data.PT08_S5_O3, data.T, data.RH, data.AH
    ]])

    try:
        co_pred = float(model_to_use.predict(X)[0])
        chat_luong, canh_bao = assess_air_quality(co_pred)

        return {
            "prediction": {
                "co_val": round(co_pred, 2),
                "label": chat_luong,
                "alert": canh_bao
            },
            "model_version_time": last_model_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")