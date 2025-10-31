import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="centered")

st.title("🚚 Predictive Delivery Optimizer")
st.markdown("AI model to predict delivery delays using logistics and IoT data.")

# --- Load model safely ---
MODEL_PATH = "models/xgb_model.joblib"
SCALER_PATH = "models/scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model files not found! Please run `python -m src.train_model` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Upload CSV ---
uploaded = st.file_uploader("📂 Upload your logistics dataset (CSV)", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    st.subheader("📄 Data Preview")
    st.write(data.head())

    # Drop unwanted columns so features match training data
    cols_to_drop = ["delayed", "delay_probability", "delivery_time_deviation", "risk_classification", "timestamp"]
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors="ignore")

    # Select only numeric columns
    features = data.select_dtypes(include=np.number)

    # Scale features using the same scaler used during training
    try:
        scaled = scaler.transform(features)
    except Exception as e:
        st.error(f"Feature mismatch detected: {e}")
        st.stop()

    # Predict probabilities
    preds = model.predict_proba(scaled)[:, 1]

    # Add predictions back to DataFrame
    data["Predicted_Probability"] = preds
    data["Predicted_Status"] = np.where(preds > 0.5, "Delayed", "On Time")

    st.subheader("✅ Predictions")
    st.dataframe(data[["Predicted_Probability", "Predicted_Status"]])

    # Download predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="predicted_delays.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin predictions.")
