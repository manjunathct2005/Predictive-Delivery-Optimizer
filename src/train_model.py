# ============================
#  train_model.py
#  Predictive Delivery Optimizer
# ============================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# --- Fix import path dynamically ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_and_clean_data, scale_features
from src.feature_engineering import feature_engineering

# ============================
#  STEP 1 — Load & Preprocess Data
# ============================
print("\n🚀 Starting Model Training...")

DATA_PATH = "data/dynamic_supply_chain_logistics_dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = load_and_clean_data(DATA_PATH)
df = feature_engineering(df)

# ============================
#  STEP 2 — Split Features & Labels
# ============================
if "delayed" not in df.columns:
    df["delayed"] = (df["delay_probability"] > 0.5).astype(int)

X = df.drop(columns=["delayed", "delay_probability", "delivery_time_deviation"], errors="ignore")
y = df["delayed"]

# ============================
#  STEP 3 — Scale Features
# ============================
X_scaled, scaler = scale_features(X)

# ============================
#  STEP 4 — Split Train/Test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================
#  STEP 5 — Build and Train Model
# ============================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

print("🧠 Training XGBoost model...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ============================
#  STEP 6 — Evaluate Model
# ============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Training Completed Successfully!")
print(f"🔹 Accuracy: {acc:.4f}")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# ============================
#  STEP 7 — Save Artifacts
# ============================
os.makedirs("models", exist_ok=True)
dump(model, "models/xgb_model.joblib")
dump(scaler, "models/scaler.joblib")

print("\n💾 Model and Scaler saved successfully in 'models/' folder.")
print("🎯 You can now run evaluate_model.py or launch Streamlit app.\n")
