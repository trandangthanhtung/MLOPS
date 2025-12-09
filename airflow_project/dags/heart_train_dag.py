from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset
from datetime import datetime
import pandas as pd
import os
import pickle

# Dataset logic
heart_data = Dataset("/opt/airflow/data/heart.csv")
trained_model = Dataset("/opt/airflow/data/heart_model.pkl")

DATA_PATH = "/opt/airflow/data/heart.csv"
MODEL_PATH = "/opt/airflow/data/heart_model.pkl"
ENCODER_PATH = "/opt/airflow/data/encoders.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    return len(df)


def train_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(DATA_PATH)

    label_cols = df.select_dtypes(include=['object']).columns
    encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(encoders, open(ENCODER_PATH, "wb"))
    print("Model saved.")


def test_inference():
    df = pd.read_csv(DATA_PATH)
    model = pickle.load(open(MODEL_PATH, "rb"))
    encoders = pickle.load(open(ENCODER_PATH, "rb"))

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col].astype(str))

    X = df.drop("HeartDisease", axis=1)
    sample = X.iloc[0:1]

    pred = model.predict(sample)[0]
    print("Prediction sample:", pred)


with DAG(
    "heart_train_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    task_load = PythonOperator(
        task_id="load_dataset",
        python_callable=load_data,
        outlets=[heart_data],          # 👈 Dataset xuất hiện ở UI
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        outlets=[trained_model],       # 👈 Dataset thứ 2
    )

    task_test = PythonOperator(
        task_id="test_inference",
        python_callable=test_inference,
        outlets=[],
    )

    task_load >> task_train >> task_test