cat << 'EOF' > aqi-mlops/src/ingest.py
import pandas as pd
import os

RAW_PATH = "data/raw/Air Quality.csv"

def load_raw():
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"✔ Loaded: {df.shape} rows")
    return df

if __name__ == "__main__":
    load_raw()
EOF