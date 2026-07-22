import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
mlflow.set_experiment("Iris Classification")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/raw/iris.csv")

le = LabelEncoder()
df["target"] = le.fit_transform(df["species"])

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

pd.concat([X_train, y_train], axis=1).to_csv(
    "data/processed/train.csv",
    index=False,
)

pd.concat([X_test, y_test], axis=1).to_csv(
    "data/processed/eval.csv",
    index=False,
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
]

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for params in param_grid:

    with mlflow.start_run():

        model = RandomForestClassifier(
            random_state=42,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
        )

        # --------------------------
        # Log Parameters
        # --------------------------

        mlflow.log_params(params)

        # --------------------------
        # Log Metrics
        # --------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # --------------------------
        # Log & Register Model
        # --------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name="IrisClassifier",
        )

        print(
            f"Completed Run | "
            f"Trees={params['n_estimators']} | "
            f"Depth={params['max_depth']} | "
            f"Accuracy={accuracy:.4f}"
        )

print("\nAll experiments completed successfully.")
