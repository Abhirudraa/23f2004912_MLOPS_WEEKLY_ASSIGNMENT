# Week 5 - Integrating MLflow into the IRIS Classification Pipeline

## Overview

This project demonstrates a complete Machine Learning Operations (MLOps) workflow using the IRIS dataset. The pipeline integrates **MLflow** for experiment tracking and model management while using **DVC** exclusively for dataset versioning.

The implementation includes:

* Hyperparameter tuning
* MLflow Experiment Tracking
* MLflow Model Registry
* Model versioning
* Evaluation using models fetched from the MLflow Registry
* Data versioning with DVC
* Git-based source code version control

---

# Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       ├── train.csv
│       └── eval.csv
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── inference.py
├── mlflow.db
├── mlruns/
├── models/
├── tests/
├── README.md
├── LICENSE
└── .gitignore
```

---

# Utility of Each File

## `scripts/train.py`

This is the main training pipeline.

Responsibilities:

* Loads the IRIS dataset
* Performs preprocessing
* Splits the dataset into train and evaluation sets
* Performs hyperparameter tuning using multiple Random Forest configurations
* Trains multiple models
* Logs:

  * Hyperparameters
  * Evaluation metrics
  * Model artifacts
* Registers trained models into the MLflow Model Registry
* Saves processed train and evaluation datasets

Technologies Used:

* Scikit-learn
* MLflow

---

## `scripts/evaluate.py`

Evaluation pipeline.

Responsibilities:

* Connects to the MLflow Tracking Server
* Retrieves the latest registered model from the MLflow Model Registry
* Loads the evaluation dataset
* Performs prediction
* Calculates:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

Unlike previous assignments, it **does not load a local Joblib model**.

---

## `inference.py`

Inference pipeline.

Purpose:

* Demonstrates prediction on unseen samples.
* Can be extended to load the production model from the MLflow Model Registry.

---

## `data/raw/iris.csv`

Original IRIS dataset.

This file is version-controlled using **DVC**.

---

## `data/processed/train.csv`

Training dataset generated during preprocessing.

Tracked using DVC.

---

## `data/processed/eval.csv`

Evaluation dataset generated after the train-test split.

Tracked using DVC.

---

## `mlflow.db`

SQLite backend used by MLflow.

Stores:

* Experiments
* Runs
* Metrics
* Parameters
* Model metadata
* Registry information

This file is ignored by Git because it is generated locally.

---

## `mlruns/`

MLflow artifact directory.

Contains:

* Logged model artifacts
* MLmodel metadata
* Environment specifications
* Run artifacts

Ignored from Git.

---

## `models/`

Previously stored local Joblib models.

For this assignment, model tracking has been removed from DVC.

All trained models are now managed exclusively through the MLflow Model Registry.

---

## `tests/`

Contains automated test cases.

### `test_data.py`

Validates:

* Dataset existence
* Dataset integrity
* Required columns

### `test_model.py`

Validates:

* Model predictions
* Expected output format
* Basic inference functionality

---

## `.gitignore`

Excludes generated files from Git.

Examples:

* `mlflow.db`
* `mlruns/`
* Python cache files

---

## `LICENSE`

Project license information.

---

# MLflow Workflow

```text
Dataset
    │
    ▼
Training Pipeline
    │
    ▼
Hyperparameter Tuning
    │
    ▼
MLflow Experiment Tracking
    │
    ├── Parameters
    ├── Metrics
    ├── Artifacts
    ▼
Model Registry
    │
    ▼
Evaluation Pipeline
```

---

# Hyperparameter Tuning

The training pipeline evaluates multiple Random Forest configurations.

Example search space:

```python
[
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5}
]
```

Each configuration generates an independent MLflow run.

---

# MLflow Features Used

* Experiment Tracking
* Multiple Training Runs
* Hyperparameter Logging
* Metric Logging
* Model Artifact Logging
* Model Registry
* Model Versioning
* Tracking UI

---

# DVC Integration

DVC is used only for dataset versioning.

Tracked:

* `data/raw/iris.csv`
* `data/processed/train.csv`
* `data/processed/eval.csv`

Not tracked:

* Trained models

Model management is handled completely by MLflow.

---

# How to Run

## Train

```bash
python scripts/train.py
```

---

## Evaluate

```bash
python scripts/evaluate.py
```

---

## Start MLflow UI

```bash
mlflow ui \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*"
```

Open:

```
http://localhost:5000
```

(or access through the Jupyter notebook proxy when running on GCP.)

---

# Technologies Used

* Python
* Scikit-learn
* MLflow
* DVC
* Git
* Pandas

---

# Assignment Tasks Completed

* Hyperparameter tuning integrated into the training pipeline.
* Experiment tracking using MLflow.
* Multiple experiments compared in the MLflow Tracking UI.
* Models registered and versioned using the MLflow Model Registry.
* Removed model dependency from DVC.
* Evaluation pipeline modified to fetch models directly from the MLflow Model Registry.

---

# Author

**Abhijeet Kumar**

IIT Madras BS Degree Programme – Data Science and Applications

GitHub: **Abhirudraa**

