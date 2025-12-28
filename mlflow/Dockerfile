FROM python:3.10-slim

WORKDIR /mlflow

# Cài đặt curl để phục vụ Healthcheck (giúp Airflow biết khi nào MLflow thực sự sẵn sàng)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow

# Tạo thư mục chứa dữ liệu
RUN mkdir -p /mlflow/mlruns /mlflow/models

# Expose port mặc định
EXPOSE 5000

# Lệnh khởi chạy: Sử dụng --host 0.0.0.0 để lắng nghe mọi kết nối trong Docker Network
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns"]