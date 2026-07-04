from feast import FeatureStore
import joblib
import pandas as pd

# ----------------------------------------------------
# Connect to Feast
# ----------------------------------------------------
store = FeatureStore(
    repo_path="iris_feature_repo/feature_repo"
)

# ----------------------------------------------------
# Load trained model
# ----------------------------------------------------
model = joblib.load("models/model.joblib")

# ----------------------------------------------------
# Sample IDs for inference
# ----------------------------------------------------
entity_rows = [
    {"iris_id": 0},
    {"iris_id": 8},
    {"iris_id": 10},
    {"iris_id": 16},
    {"iris_id": 23},
]

# ----------------------------------------------------
# Fetch features from ONLINE store
# ----------------------------------------------------
online_features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=entity_rows,
).to_df()

print("\nFeatures fetched from Feast:\n")
print(online_features)

# ----------------------------------------------------
# Predict
# ----------------------------------------------------
X = online_features[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

predictions = model.predict(X)

print("\nPredictions:\n")

for iris_id, pred in zip(
    online_features["iris_id"],
    predictions,
):
    print(f"iris_id={iris_id}  -->  {pred}")

# ----------------------------------------------------
# Compare with actual labels
# ----------------------------------------------------
iris_df = pd.read_parquet(
    "iris_feature_repo/feature_repo/data/iris_features.parquet"
)

actual = iris_df[
    iris_df["iris_id"].isin(
        online_features["iris_id"]
    )
][["iris_id", "species"]]

print("\nActual labels:\n")
print(actual)
