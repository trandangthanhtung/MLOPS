cat << 'EOF' > aqi-mlops/src/preprocess.py
import pandas as pd
import numpy as np

RAW = "data/raw/Air Quality.csv"
PROCESSED = "data/processed/processed.csv"

def preprocess():
    df = pd.read_csv(RAW)

    # Basic cleaning
    df = df.replace([-200, "NA"], np.nan)
    df = df.interpolate().ffill().bfill()

    # Create AQI proxy
    df["AQI_proxy"] = df.select_dtypes("number").mean(axis=1)

    df.to_csv(PROCESSED, index=False)
    print("✔ Saved processed data:", PROCESSED)

if __name__ == "__main__":
    preprocess()
EOF