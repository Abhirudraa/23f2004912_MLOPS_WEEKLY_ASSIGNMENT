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

PROJECT_DIR = "/home/jupyter/23f2004912_MLOPS_WEEKLY_ASSIGNMENT"

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR}/mlflow.db")

# --------------------------------------------------
# Get Latest Registered Model Version
# --------------------------------------------------

client = MlflowClient()

versions = client.search_model_versions("name='IrisClassifier'")

latest_version = max(
    versions,
    key=lambda x: int(x.version)
).version

model_uri = f"models:/IrisClassifier/{latest_version}"
print(f"Loading Model Version: {latest_version}")

model = mlflow.pyfunc.load_model(model_uri)

# --------------------------------------------------
# Load Evaluation Data
# --------------------------------------------------

df = pd.read_csv("data/processed/eval.csv")

X = df.drop(columns=["target"])
y_true = df["target"]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
