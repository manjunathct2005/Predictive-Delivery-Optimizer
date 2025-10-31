import pandas as pd

def feature_engineering(df):
    # Convert timestamp if exists
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.weekday
        df["month"] = df["timestamp"].dt.month

    # Fill missing numeric values
    df = df.fillna(df.mean(numeric_only=True))

    return df