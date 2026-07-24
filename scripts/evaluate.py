import pandas as pd

from skops.io import load, get_untrusted_types

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# Load model
# -----------------------------

trusted = get_untrusted_types(file="models/model.skops")

model = load(
    "models/model.skops",
    trusted=trusted
)

# -----------------------------
# Load evaluation data
# -----------------------------

df = pd.read_csv("data/processed/eval.csv")

X = df.drop(columns=["target"])
y_true = df["target"]

# -----------------------------
# Predict
# -----------------------------

y_pred = model.predict(X)

# -----------------------------
# Metrics
# -----------------------------

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
