import yaml
import joblib
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def get_best_model(experiment_name, metric="rmse"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    return runs[0] if runs else None

import yaml

def read_config(path="/opt/airflow/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)