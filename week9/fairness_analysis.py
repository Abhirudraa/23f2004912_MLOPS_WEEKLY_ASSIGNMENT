import numpy as np
import pandas as pd

from skops.io import load, get_untrusted_types

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

from fairlearn.metrics import MetricFrame


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42

MODEL_PATH = "models/model.skops"
EVAL_PATH = "data/processed/eval.csv"
OUTPUT_PATH = "week9/eval_with_location.csv"


# --------------------------------------------------
# Load model
# --------------------------------------------------

trusted = get_untrusted_types(file=MODEL_PATH)

model = load(
    MODEL_PATH,
    trusted=trusted
)


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

df = pd.read_csv(EVAL_PATH)


# --------------------------------------------------
# Introduce sensitive attribute
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

df["location"] = rng.integers(
    0,
    2,
    size=len(df)
)


# --------------------------------------------------
# Model features
# Location is NOT included.
# --------------------------------------------------

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X = df[features]
y_true = df["target"]

sensitive_features = df["location"]


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X)


# --------------------------------------------------
# Fairness analysis
# --------------------------------------------------

metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,

        "precision": lambda y_true, y_pred:
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "recall": lambda y_true, y_pred:
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nOverall metrics:")
print(metric_frame.overall)

print("\nMetrics by location:")
print(metric_frame.by_group)

print("\nMetric differences between groups:")
print(metric_frame.difference())


# --------------------------------------------------
# Save dataset with sensitive attribute
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"\nSaved dataset to: {OUTPUT_PATH}")
