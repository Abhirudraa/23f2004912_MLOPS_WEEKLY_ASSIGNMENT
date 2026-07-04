from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------------------------------------------
# Connect to Feast
# ---------------------------------------------------
store = FeatureStore(
    repo_path="iris_feature_repo/feature_repo"
)

# ---------------------------------------------------
# Read entity dataframe
# (Only entity keys + timestamps)
# ---------------------------------------------------
iris_df = pd.read_parquet(
    "iris_feature_repo/feature_repo/data/iris_features.parquet"
)

entity_df = iris_df[["iris_id", "event_timestamp"]]

# ---------------------------------------------------
# Fetch historical features from Feast
# ---------------------------------------------------
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
).to_df()

# ---------------------------------------------------
# Add target labels
# (Labels are NOT stored in Feast)
# ---------------------------------------------------
training_df = training_df.merge(
    iris_df[["iris_id", "species"]],
    on="iris_id",
)

# ---------------------------------------------------
# Prepare training data
# ---------------------------------------------------
X = training_df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = training_df["species"]

# ---------------------------------------------------
# Train/Test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ---------------------------------------------------
# Train model
# ---------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"Accuracy : {accuracy:.4f}")

# ---------------------------------------------------
# Save model
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/model.joblib",
)

print("Model saved.")
