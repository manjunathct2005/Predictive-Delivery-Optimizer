import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/scaler.joblib")

df = pd.read_csv("data/dynamic_supply_chain_logistics_dataset.csv")
df["delayed"] = (df["delay_probability"] > 0.5).astype(int)

X = df.drop(columns=["delayed", "delay_probability", "risk_classification", "delivery_time_deviation", "timestamp"], errors="ignore")
X_scaled = scaler.transform(X)

explainer = shap.Explainer(model)
shap_values = explainer(X_scaled)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("reports/shap_summary_plot.png")
print("✅ SHAP summary plot saved in reports/")
