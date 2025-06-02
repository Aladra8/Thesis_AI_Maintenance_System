# # src/evaluate_model.py

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     confusion_matrix, classification_report,
#     roc_auc_score, roc_curve
# )
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import load_model

# # ------------------------------
# # Setup paths
# # ------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# X_PATH = os.path.join(BASE_DIR, "data", "processed", "X_processed.csv")
# Y_PATH = os.path.join(BASE_DIR, "data", "processed", "y_labels.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
# EVAL_DIR = os.path.join(BASE_DIR, "outputs", "evaluation")

# # Ensure output directory exists
# os.makedirs(EVAL_DIR, exist_ok=True)

# # ------------------------------
# # Load data and model
# # ------------------------------
# print("📥 Loading processed data and model...")
# X = pd.read_csv(X_PATH)
# y = pd.read_csv(Y_PATH).squeeze()  # Convert DataFrame to Series
# model = load_model(MODEL_PATH)

# # ------------------------------
# # Train/test split
# # ------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ------------------------------
# # Evaluate model
# # ------------------------------
# print("🔍 Evaluating model...")
# y_pred_probs = model.predict(X_test).flatten()
# y_pred = (y_pred_probs >= 0.5).astype(int)

# # ------------------------------
# # Classification report
# # ------------------------------
# report = classification_report(y_test, y_pred, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv(os.path.join(EVAL_DIR, "classification_report.csv"))
# print("✅ Classification report saved.")

# # ------------------------------
# # Confusion Matrix
# # ------------------------------
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
# plt.close()
# print("✅ Confusion matrix saved.")

# # ------------------------------
# # ROC Curve
# # ------------------------------
# roc_auc = roc_auc_score(y_test, y_pred_probs)
# fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.title("ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(EVAL_DIR, "roc_curve.png"))
# plt.close()
# print(f"✅ ROC curve saved (AUC: {roc_auc:.2f})")

# # ------------------------------
# # Done
# # ------------------------------
# print("🎉 Evaluation complete. Results saved to outputs/evaluation/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
import os

# 📂 Ensure output directories exist
os.makedirs("outputs/evaluation", exist_ok=True)

# 📦 Load the test data
print("📦 Loading dataset and model...")
df = pd.read_csv("data/raw/predictive-maintenance-dataset.csv")

# Rename sensor columns (for consistency)
df.rename(columns={
    "x1": "sensor_1",
    "x2": "sensor_2",
    "x3": "sensor_3",
    "x4": "sensor_4",
    "x5": "sensor_5"
}, inplace=True)

# Fill missing vibration values
df['vibration'] = df['vibration'].fillna(df['vibration'].median())

# Recreate 'fault' labels (top 5% vibration = fault)
thresh = df['vibration'].quantile(0.95)
df['fault'] = df['vibration'] > thresh

# Extract features and labels
X = df.drop(columns=['ID', 'fault'])
y = df['fault'].astype(int)

# Normalize with saved scaler
scaler = joblib.load("outputs/model/scaler.pkl")
X_scaled = scaler.transform(X)

# Load trained model
model = load_model("outputs/model/model.h5")

# Evaluate the model
print("🔍 Evaluating model...")
y_pred_probs = model.predict(X_scaled).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# Print classification report
report = classification_report(y, y_pred, digits=4)
print("\n📊 Classification Report:\n", report)

# Save report to file
with open("outputs/evaluation/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Fault"], yticklabels=["Normal", "Fault"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_probs)
roc_auc = roc_auc_score(y, y_pred_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/evaluation/roc_curve.png")
plt.close()

print("✅ Evaluation complete. Results saved in 'outputs/evaluation/'")
