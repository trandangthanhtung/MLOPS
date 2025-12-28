import pandas as pd
import numpy as np
from src.monitoring.drift import psi

# ================== CONFIG ==================
DATA_PATH = "/opt/airflow/data/processed/clean.csv"

FEATURE_COLS = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

PSI_THRESHOLD = 0.25
FAIL_ON_DRIFT = False   # 🔥 ĐỔI TRUE nếu muốn Airflow FAIL

# ============================================

def check_drift():
    df = pd.read_csv(DATA_PATH)

    # --- Safety check ---
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing columns for drift check: {missing_cols}")

    # --- Split baseline vs recent (time-based) ---
    split_idx = int(len(df) * 0.7)
    baseline = df.iloc[:split_idx]
    recent = df.iloc[split_idx:]

    print("\n🔍 Drift check (PSI):")

    drifted_features = {}

    for col in FEATURE_COLS:
        base_col = baseline[col].to_numpy()
        new_col = recent[col].to_numpy()

        # Remove NaN
        base_col = base_col[~np.isnan(base_col)]
        new_col = new_col[~np.isnan(new_col)]

        # Skip if not enough data
        if len(base_col) < 50 or len(new_col) < 50:
            print(f"  ⚠️ {col:<15} skipped (not enough samples)")
            continue

        value = psi(base_col, new_col)
        print(f"  {col:<15} PSI = {value:.4f}")

        if value > PSI_THRESHOLD:
            drifted_features[col] = value

    # --- Summary ---
    if drifted_features:
        print("\n⚠️ DATA DRIFT DETECTED (soft warning):")
        for f, v in drifted_features.items():
            print(f"   - {f}: PSI = {v:.4f}")

        if FAIL_ON_DRIFT:
            raise ValueError("❌ Pipeline stopped due to data drift")
        else:
            print("➡️ Pipeline continues (FAIL_ON_DRIFT = False)")
    else:
        print("\n✅ No significant data drift detected")

    return drifted_features


if __name__ == "__main__":
    check_drift()
