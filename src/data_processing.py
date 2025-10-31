import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Label Creation: delayed or on-time
    df["delayed"] = (df["delay_probability"] > 0.5).astype(int)

    # Drop unnecessary or identifier columns
    drop_cols = ["timestamp", "risk_classification"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df

def scale_features(X):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler