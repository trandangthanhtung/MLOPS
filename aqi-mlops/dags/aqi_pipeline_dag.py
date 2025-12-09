cat << 'EOF' > aqi-mlops/dags/aqi_pipeline_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("aqi_pipeline",
         start_date=datetime(2024,1,1),
         schedule_interval="@daily",
         catchup=False):

    ingest = BashOperator(
        task_id="ingest",
        bash_command="python /project/src/ingest.py"
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /project/src/preprocess.py"
    )

    train = BashOperator(
        task_id="train",
        bash_command="python /project/src/train.py"
    )

    ingest >> preprocess >> train
EOF