import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    X = pd.read_csv('data/processed/eval.csv').drop('target', axis=1)
    y = pd.read_csv('data/processed/eval.csv')['target']
    return X, y

def test_model_metrics():
    model = joblib.load('models/model.joblib')
    X, y_true = load_data()
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    assert acc >= 0.90, f"Accuracy {acc:.3f} < 0.90"
    assert prec >= 0.85, f"Precision {prec:.3f} < 0.85"
    assert rec >= 0.85, f"Recall {rec:.3f} < 0.85"
    assert f1 >= 0.85, f"F1 {f1:.3f} < 0.85"
