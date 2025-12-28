# Air Quality MLOps Project

Dự án MLOps dự đoán chất lượng không khí sử dụng MLflow, Airflow, và FastAPI.

## 🏗️ Kiến trúc

- **MLflow**: Tracking experiments và lưu trữ models (port 5000)
- **Airflow**: Orchestration pipeline training (port 8081)
- **FastAPI**: API serving predictions (port 8000)

## 📋 Yêu cầu

- Docker Desktop
- Docker Compose

## 🚀 Cách chạy

### 1. Build tất cả services

```powershell
docker compose build
```

### 2. Khởi động services

```powershell
docker compose up -d
```

### 3. Kiểm tra services

- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8081 (username: `admin`, password: `admin`)
- **FastAPI docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

### 4. Chạy pipeline training

Truy cập Airflow UI và trigger DAG `air_quality_pipeline`:

1. Vào http://localhost:8081
2. Login với admin/admin
3. Tìm DAG `air_quality_pipeline`
4. Click nút "Play" để chạy

Pipeline sẽ thực hiện:
- **preprocess_data**: Xử lý dữ liệu thô
- **drift_check**: Kiểm tra data drift
- **train_models**: Train XGBoost model và log vào MLflow

### 5. Test API prediction

Sau khi training xong, test API:

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "PT08_S1_CO": 1300,
    "NMHC_GT": 150,
    "C6H6_GT": 11.9,
    "PT08_S2_NMHC": 1046,
    "NOx_GT": 166,
    "PT08_S3_NOx": 1056,
    "NO2_GT": 113,
    "PT08_S4_NO2": 1692,
    "PT08_S5_O3": 1268,
    "T": 13.6,
    "RH": 48.9,
    "AH": 0.7578
  }'
```

## 📁 Cấu trúc project

```
.
├── airflow/              # Airflow configs và DAGs
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       └── aqi_pipeline.py
├── app/                  # FastAPI application
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── mlflow/               # MLflow server
│   ├── Dockerfile
│   └── mlruns/          # Experiments storage
├── src/                  # Source code
│   ├── preprocessing/   # Data preprocessing
│   ├── training/        # Model training
│   ├── monitoring/      # Drift detection
│   └── features/        # Feature engineering
├── data/                 # Raw và processed data
├── models/               # Saved models
├── config/               # Configuration files
└── docker-compose.yml
```

## 🛠️ Troubleshooting

### Services không start

```powershell
docker compose logs <service_name>
```

### Reset toàn bộ

```powershell
docker compose down -v
docker compose up -d --build
```

### Xem logs real-time

```powershell
docker compose logs -f
```

## 📊 Sử dụng

### Training manual

Vào container airflow:

```powershell
docker exec -it airflow bash
cd /opt/airflow
python src/training/train_all_models.py
```

### Xem experiments trong MLflow

Truy cập http://localhost:5000 để xem:
- Metrics (RMSE, MAE, R2)
- Parameters
- Models đã train
- Artifacts

## 🔧 Development

### Sửa code và rebuild

```powershell
docker compose up -d --build <service_name>
```

### Chỉ rebuild một service cụ thể

```powershell
docker compose build airflow
docker compose up -d airflow
```

## 📝 API Endpoints

- `GET /`: Thông tin API
- `GET /health`: Health check
- `POST /predict`: Dự đoán chất lượng không khí

## 🎯 Features

- ✅ Automated data preprocessing
- ✅ XGBoost regression model
- ✅ MLflow experiment tracking
- ✅ Airflow pipeline orchestration
- ✅ FastAPI REST API
- ✅ Docker containerization
- ✅ Data drift detection

## 📄 License

MIT
