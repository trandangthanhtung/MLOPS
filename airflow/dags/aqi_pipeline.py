from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import sys

# Thêm src vào sys.path nếu cần
sys.path.append("/opt/airflow")

default_args = {"start_date": datetime(2024, 1, 1)}

with DAG(
    "air_quality_pipeline",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False
) as dag:

    # BashOperator chạy feature_engineering.py
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/airflow && PYTHONPATH=/opt/airflow python src/preprocessing/feature_engineering.py"
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command="cd /opt/airflow && PYTHONPATH=/opt/airflow python src/monitoring/drift_check.py"
    )
    train = BashOperator(
        task_id="train_models",
        bash_command='cd /opt/airflow && PYTHONPATH=/opt/airflow/src python src/training/train_all_models.py',
    )
    preprocess >> drift_check >> train
