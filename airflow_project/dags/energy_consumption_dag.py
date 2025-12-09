from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset
from datetime import datetime
import os
import pickle
import json

DATA_CSV = "/opt/airflow/data/energy_consumption.csv"
RAW_OUT = "/opt/airflow/data/energy_raw.csv"
PROCESSED = "/opt/airflow/data/energy_processed.csv"
MODEL_PATH = "/opt/airflow/models/energy_model.pkl"
ENCODERS_PATH = "/opt/airflow/models/energy_encoders.pkl"
METRICS_PATH = "/opt/airflow/data/energy_metrics.json"

dataset_energy = Dataset(DATA_CSV)
dataset_energy_model = Dataset(MODEL_PATH)

def ingest_energy():
    import pandas as pd
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(DATA_CSV + " not found")
    df = pd.read_csv(DATA_CSV)
    os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print("Ingested energy:", RAW_OUT, "shape", df.shape)

def preprocess_energy():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(RAW_OUT)
    encoders = {}
    # encode categorical cols
    for c in ["country", "energy_source"]:
        if c in df.columns:
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    # keep numeric features
    keep_cols = [c for c in df.columns if c not in ("country", "energy_source")]
    # add encoded columns
    for k in encoders.keys():
        keep_cols.append(k + "_enc")
    df[keep_cols].to_csv(PROCESSED, index=False)
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print("Preprocessed energy saved:", PROCESSED)

def train_energy():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    df = pd.read_csv(PROCESSED)
    target = "co2_emission_mt" if "co2_emission_mt" in df.columns else "consumption_TWh"
    if target not in df.columns:
        raise KeyError("Target not present in processed data")
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=["number"])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    metrics = {"mse": mse, "n_train": int(X_train.shape[0])}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    print("Energy model saved:", MODEL_PATH, "MSE:", mse)

def evaluate_energy():
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    df = pd.read_csv(PROCESSED)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    target = "co2_emission_mt" if "co2_emission_mt" in df.columns else "consumption_TWh"
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=["number"])
    preds = model.predict(X)
    mse = float(mean_squared_error(df[target], preds))
    print("Full eval MSE:", mse)
    try:
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
    except Exception:
        m = {}
    m["mse_full"] = mse
    with open(METRICS_PATH, "w") as f:
        json.dump(m, f)

with DAG(
    dag_id="energy_consumption_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    t_ing = PythonOperator(task_id="ingest_energy", python_callable=ingest_energy, outlets=[dataset_energy])
    t_prep = PythonOperator(task_id="preprocess_energy", python_callable=preprocess_energy)
    t_train = PythonOperator(task_id="train_energy_model", python_callable=train_energy, outlets=[dataset_energy_model])
    t_eval = PythonOperator(task_id="evaluate_energy_model", python_callable=evaluate_energy)

    t_ing >> t_prep >> t_train >> t_eval