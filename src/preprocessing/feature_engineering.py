# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = "/opt/airflow/data/AirQualityUCI.csv"
OUT_PATH = "/opt/airflow/data/processed/clean.csv"

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

TARGET_COL = "CO(GT)"

def preprocess():
    # Load raw CSV
    df = pd.read_csv(RAW_PATH, sep=";")

    # Drop cột rác
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Tạo datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )

    df = df.drop(columns=["Date", "Time"])

    # Chuẩn hóa số: , → .
    for col in df.columns:
        if col != "datetime":
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    # Thay -200 → NaN
    df = df.replace(-200, np.nan)

    # ❗ GIỮ DỮ LIỆU DÀI
    # Chỉ drop dòng nếu thiếu quá nhiều cột
    df = df.dropna(thresh=int(len(df.columns) * 0.7))

    # Điền giá trị thiếu theo thời gian
    df = df.sort_values("datetime")
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Giữ cột cần thiết
    df = df[["datetime"] + FEATURE_COLS + [TARGET_COL]]

    # Save
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"✅ clean.csv saved | shape = {df.shape}")

if __name__ == "__main__":
    preprocess()