import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ===============================
# CONFIG
# ===============================
DATA_PATH = "/opt/airflow/data/processed/clean.csv"
MODEL_DIR = "/opt/airflow/models"
TARGET_COL = "CO(GT)"
FEATURE_COLS = [
    "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
    "PT08.S5(O3)", "T", "RH", "AH",
]

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# MLFLOW SETUP (Cấu hình chống lỗi 403)
# ===============================
# Lấy URI từ biến môi trường, mặc định là service name trong docker-compose
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "AIR_QUALITY_MLOPS"

# Hàm bổ trợ để set experiment an toàn
def setup_mlflow(name):
    try:
        # Kiểm tra xem experiment đã tồn tại chưa
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            print(f"Creating new experiment: {name}")
            mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to MLflow at {tracking_uri}")
        print(f"Error details: {e}")
        # Nếu lỗi 403 tiếp tục, code vẫn chạy nhưng không log vào MLflow
        return False
    return True

# Thực hiện setup
is_mlflow_ready = setup_mlflow(experiment_name)

# ===============================
# LOAD & SPLIT DATA
# ===============================
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAINING
# ===============================
print("🚀 Training XGBoost model...")

# Khởi tạo run (chỉ khi MLflow sẵn sàng)
run = mlflow.start_run(run_name="XGBoost") if is_mlflow_ready else None

try:
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    if is_mlflow_ready:
        mlflow.log_params({
            "model": "XGBoost",
            "n_estimators": 100,
            "max_depth": 5
        })
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.xgboost.log_model(model, "model")

    print(f"✅ Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

finally:
    if run:
        mlflow.end_run()

# ===============================
# SAVE MODEL LOCALLY
# ===============================
# ===============================
# SAVE MODEL TO ROOT (Dành cho FastAPI)
# ===============================
# Đường dẫn file model cuối cùng
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FLAG_FILE_PATH = os.path.join(MODEL_DIR, "MODEL_READY")

# Lưu model thực tế
joblib.dump(model, FINAL_MODEL_PATH)

# Lưu timestamp vào file READY để FastAPI nhận diện có model mới
with open(FLAG_FILE_PATH, "w") as f:
    f.write(str(time.time()))

print(f"💾 Model saved to: {FINAL_MODEL_PATH}")
print("🎉 Pipeline finished!")

print("🎉 Training completed!")