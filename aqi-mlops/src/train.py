cat << 'EOF' > aqi-mlops/src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PROCESSED = "data/processed/processed.csv"

def train():
    df = pd.read_csv(PROCESSED)
    X = df.drop("AQI_proxy", axis=1)
    y = df["AQI_proxy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.set_experiment("AQI-Training")

    with mlflow.start_run():
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

        print("✔ Training done — RMSE:", rmse)

if __name__ == "__main__":
    train()
EOF