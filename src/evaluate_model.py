import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("data/dynamic_supply_chain_logistics_dataset.csv")
df["delayed"] = (df["delay_probability"] > 0.5).astype(int)

X = df.drop(columns=["delayed", "delay_probability", "risk_classification", "delivery_time_deviation", "timestamp"], errors="ignore")
y = df["delayed"]

model = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/scaler.joblib")

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("ROC-AUC:", roc_auc_score(y, y_pred))
print(classification_report(y, y_pred))