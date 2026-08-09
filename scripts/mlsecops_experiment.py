import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = "/home/jupyter/23f2004912_MLOPS_WEEKLY_ASSIGNMENT"

TRAINING_DIR = "poisoned_data"
EVAL_FILE = "data/processed/eval.csv"

EXPERIMENT_NAME = "Iris MLSecOps Poisoning"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET = "target"

POISONING_LEVELS = [0, 5, 10, 50]


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_DIR}/mlflow.db"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# Load clean evaluation dataset
# --------------------------------------------------

eval_df = pd.read_csv(EVAL_FILE)

X_eval = eval_df[FEATURES]
y_eval = eval_df[TARGET]


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

for poisoning_level in POISONING_LEVELS:

    print("\n" + "=" * 60)
    print(f"Starting experiment: {poisoning_level}% poisoning")
    print("=" * 60)

    # ----------------------------------------------
    # Load poisoned training dataset
    # ----------------------------------------------

    train_file = os.path.join(
        TRAINING_DIR,
        f"iris_poisoned_{poisoning_level}.csv"
    )

    train_df = pd.read_csv(train_file)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    # ----------------------------------------------
    # Count poisoned samples
    # ----------------------------------------------

    original_train = pd.read_csv(
        "data/processed/train.csv"
    )

    differences = (
        original_train[FEATURES + [TARGET]]
        != train_df[FEATURES + [TARGET]]
    ).any(axis=1)

    poisoned_samples = int(differences.sum())
    clean_samples = len(train_df) - poisoned_samples

    # ----------------------------------------------
    # MLflow run
    # ----------------------------------------------

    with mlflow.start_run(
        run_name=f"poisoning_{poisoning_level}_percent"
    ):

        # ------------------------------------------
        # Train model
        # ------------------------------------------

        model = RandomForestClassifier(
            n_estimators=MODEL_PARAMS["n_estimators"],
            max_depth=MODEL_PARAMS["max_depth"],
            random_state=MODEL_PARAMS["random_state"],
        )

        model.fit(X_train, y_train)

        # ------------------------------------------
        # Evaluate on CLEAN evaluation dataset
        # ------------------------------------------

        predictions = model.predict(X_eval)

        accuracy = accuracy_score(
            y_eval,
            predictions
        )

        precision = precision_score(
            y_eval,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_eval,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_eval,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # ------------------------------------------
        # Log parameters
        # ------------------------------------------

        mlflow.log_param(
            "poisoning_level",
            poisoning_level
        )

        mlflow.log_param(
            "training_samples",
            len(train_df)
        )

        mlflow.log_param(
            "poisoned_samples",
            poisoned_samples
        )

        mlflow.log_param(
            "clean_samples",
            clean_samples
        )

        mlflow.log_param(
            "n_estimators",
            MODEL_PARAMS["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            MODEL_PARAMS["max_depth"]
        )

        mlflow.log_param(
            "random_state",
            MODEL_PARAMS["random_state"]
        )

        # ------------------------------------------
        # Log metrics
        # ------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # ------------------------------------------
        # Log model
        # ------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model"
        )

        # ------------------------------------------
        # Print results
        # ------------------------------------------

        print(
            f"Poisoning Level : {poisoning_level}%"
        )

        print(
            f"Training Samples: {len(train_df)}"
        )

        print(
            f"Poisoned Samples: {poisoned_samples}"
        )

        print(
            f"Accuracy        : {accuracy:.4f}"
        )

        print(
            f"Precision       : {precision:.4f}"
        )

        print(
            f"Recall          : {recall:.4f}"
        )

        print(
            f"F1 Score        : {f1:.4f}"
        )


print("\n" + "=" * 60)
print("All MLSecOps poisoning experiments completed.")
print("=" * 60)
