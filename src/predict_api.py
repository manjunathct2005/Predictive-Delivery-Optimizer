import joblib
import pandas as pd

model = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/scaler.joblib")

def predict_delay(input_data: dict):
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    pred = model.predict_proba(scaled)[0][1]
    return {"delay_probability": float(pred), "status": "Delayed" if pred > 0.5 else "On Time"}
