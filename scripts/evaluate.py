import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = pd.read_csv('data/processed/eval.csv').drop('target', axis=1)
y_true = pd.read_csv('data/processed/eval.csv')['target']
model = joblib.load('models/model.joblib')
y_pred = model.predict(X)

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
print(f"F1: {f1_score(y_true, y_pred, average='weighted'):.3f}")
