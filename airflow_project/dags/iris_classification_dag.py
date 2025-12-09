from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_DIR = '/opt/airflow/data'  # fix ở đây

def load_data():
    data = load_iris(as_frame=True)
    df = data.frame
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(f'{DATA_DIR}/iris_raw.csv', index=False)
    print("Data saved to", f'{DATA_DIR}/iris_raw.csv')

def preprocess():
    df = pd.read_csv(f'{DATA_DIR}/iris_raw.csv')
    df.to_csv(f'{DATA_DIR}/iris_preprocessed.csv', index=False)
    print("Preprocessed data saved.")

def train_model():
    df = pd.read_csv(f'{DATA_DIR}/iris_preprocessed.csv')
    X = df.drop(columns=['target', 'target_name'], errors='ignore')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(clf, f'{DATA_DIR}/iris_model.pkl')
    X_test.to_csv(f'{DATA_DIR}/X_test.csv', index=False)
    y_test.to_csv(f'{DATA_DIR}/y_test.csv', index=False)
    print("Model trained and saved.")

def evaluate_model(**kwargs):
    df_X = pd.read_csv(f'{DATA_DIR}/X_test.csv')
    df_y = pd.read_csv(f'{DATA_DIR}/y_test.csv')
    clf = joblib.load(f'{DATA_DIR}/iris_model.pkl')
    y_pred = clf.predict(df_X)
    acc = accuracy_score(df_y, y_pred)
    print("Test Accuracy:", acc)

with DAG(
    dag_id='iris_classification',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    t0 = PythonOperator(task_id='load_data', python_callable=load_data)
    t1 = PythonOperator(task_id='preprocess', python_callable=preprocess)
    t2 = PythonOperator(task_id='train_model', python_callable=train_model)
    t3 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)

    t0 >> t1 >> t2 >> t3