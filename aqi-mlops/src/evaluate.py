cat << 'EOF' > aqi-mlops/src/evaluate.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL = "models/model.pkl"
DATA = "data/processed/processed.csv"

def evaluate():
    df = pd.read_csv(DATA)
    model = joblib.load(MODEL)

    y_true = df["AQI_proxy"]
    y_pred = model.predict(df.drop("AQI_proxy", axis=1))

    plt.plot(y_true[:200], label="actual")
    plt.plot(y_pred[:200], label="pred")
    plt.legend()
    plt.title("AQI prediction")
    plt.savefig("models/eval_plot.png")

    print("✔ Evaluation plot saved")

if __name__ == "__main__":
    evaluate()
EOF
