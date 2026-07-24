import pandas as pd

from skops.io import load, get_untrusted_types
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_data():
    df = pd.read_csv("data/processed/eval.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def test_model_metrics():
    trusted = get_untrusted_types(file="models/model.skops")
    model = load("models/model.skops", trusted=trusted)

    X, y_true = load_data()
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    assert acc >= 0.90, f"Accuracy {acc:.3f} < 0.90"
    assert prec >= 0.85, f"Precision {prec:.3f} < 0.85"
    assert rec >= 0.85, f"Recall {rec:.3f} < 0.85"
    assert f1 >= 0.85, f"F1 {f1:.3f} < 0.85"
