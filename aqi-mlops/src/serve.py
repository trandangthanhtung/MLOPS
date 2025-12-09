cat << 'EOF' > aqi-mlops/src/serve.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "AQI Prediction API"}

@app.post("/predict")
def predict(features: list):
    x = np.array(features).reshape(1, -1)
    y = model.predict(x)[0]
    return {"AQI": float(y)}
EOF