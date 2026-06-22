from google.cloud import storage
from io import BytesIO

import joblib
import pandas as pd
import json
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()

parser.add_argument("--bucket")
parser.add_argument("--model_path")
parser.add_argument("--data_path")
parser.add_argument("--run_id")

args = parser.parse_args()

client = storage.Client()

bucket = client.bucket(args.bucket)

blob = bucket.blob(args.model_path)

model_bytes = blob.download_as_bytes()

model = joblib.load(
    BytesIO(model_bytes)
)

print("Model fetched from GCS")

df = pd.read_csv(args.data_path)

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

metrics = {
    "run_id": args.run_id,
    "dataset": args.data_path,
    "accuracy": float(acc)
}

metrics_blob = bucket.blob(
    f"runs/{args.run_id}/metrics/metrics.json"
)

metrics_blob.upload_from_string(
    json.dumps(metrics, indent=2),
    content_type="application/json"
)

print("Metrics uploaded to GCS")