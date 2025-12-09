from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset
from datetime import datetime
import os
import pickle
import json

DATA_CSV = "/opt/airflow/data/inflation.csv"
RAW_OUT = "/opt/airflow/data/inflation_raw.csv"
PROCESSED = "/opt/airflow/data/inflation_processed.csv"
MODEL_PATH = "/opt/airflow/models/inflation_model.pkl"
ENCODERS_PATH = "/opt/airflow/models/inflation_encoders.pkl"
METRICS_PATH = "/opt/airflow/data/inflation_metrics.json"

# Dataset objects (appear in UI)
dataset_inflation = Dataset(DATA_CSV)
dataset_model = Dataset(MODEL_PATH)

def ingest():
    import pandas as pd
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found")
    df = pd.read_csv(DATA_CSV)
    os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print("Ingested:", RAW_OUT, "shape", df.shape)

def preprocess():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(RAW_OUT)
    # simple feature: encode country, create cyclical month features
    encoders = {}
    if "country" in df.columns:
        le = LabelEncoder()
        df["country_enc"] = le.fit_transform(df["country"].astype(str))
        encoders["country"] = le
    # month cyclical
    if "month" in df.columns:
        df["month_sin"] = (2 * 3.14159265 * df["month"] / 12).apply(lambda x: __import__('math').sin(x))
        df["month_cos"] = (2 * 3.14159265 * df["month"] / 12).apply(lambda x: __import__('math').cos(x))
    # keep relevant cols
    keep_cols = [c for c in df.columns if c not in ("country", )]
    df[keep_cols].to_csv(PROCESSED, index=False)
    # save encoders
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print("Preprocessed saved:", PROCESSED, "encoders:", list(encoders.keys()))

def train():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    df = pd.read_csv(PROCESSED)
    if "inflation_rate" not in df.columns:
        raise KeyError("inflation_rate column not found")
    X = df.drop(columns=["inflation_rate"])
    y = df["inflation_rate"]
    # keep numeric only (if any object slipped through)
    X = X.select_dtypes(include=["number"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    # save metrics
    metrics = {"mae": mae, "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    print("Trained model saved:", MODEL_PATH, "MAE:", mae)

def evaluate():
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    df = pd.read_csv(PROCESSED)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X = df.drop(columns=["inflation_rate"])
    X = X.select_dtypes(include=["number"])
    if X.shape[0] == 0:
        print("No numeric features to evaluate")
        return
    preds = model.predict(X)
    mae = float(mean_absolute_error(df["inflation_rate"], preds))
    print("Evaluation MAE (full):", mae)
    # append to metrics file
    try:
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
    except Exception:
        m = {}
    m["mae_full"] = mae
    with open(METRICS_PATH, "w") as f:
        json.dump(m, f)

with DAG(
    dag_id="inflation_forecast_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    t_ingest = PythonOperator(task_id="ingest_inflation", python_callable=ingest, outlets=[dataset_inflation])
    t_prep = PythonOperator(task_id="preprocess_inflation", python_callable=preprocess)
    t_train = PythonOperator(task_id="train_inflation_model", python_callable=train, outlets=[dataset_model])
    t_eval = PythonOperator(task_id="evaluate_inflation_model", python_callable=evaluate)

    t_ingest >> t_prep >> t_train >> t_eval