import os
from pathlib import Path

import mlflow
import pandas as pd

from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

# --------------------------------------------------
# Get Latest Registered Model Version
# --------------------------------------------------

client = MlflowClient()

versions = client.search_model_versions("name='IrisClassifier'")

latest_version = max(
    versions,
    key=lambda v: int(v.version)
).version

print(f"Loading Model Version: {latest_version}")

model_uri = f"models:/IrisClassifier/{latest_version}"

model = mlflow.pyfunc.load_model(model_uri)

# --------------------------------------------------
# Load Evaluation Data
# --------------------------------------------------

eval_path = PROJECT_DIR / "data" / "processed" / "eval.csv"

df = pd.read_csv(eval_path)

X = df.drop(columns=["target"])
y_true = df["target"]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
