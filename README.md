# Week 3 - Integrating Feast Feature Store into the IRIS Pipeline
```
- **Course:** MLOps Weekly Assignment 3
- **Name:** Abhijeet Kumar
- **Roll Number:** 23f2004912
```
---

# Overview

In Week 2, the IRIS machine learning pipeline used **DVC** to version datasets and trained models. While DVC solved dataset and model versioning, the training pipeline still loaded features directly from the CSV file.

This creates a common production issue known as **training-serving skew**, where feature engineering logic used during training may differ from the logic used during inference.

To solve this problem, this assignment integrates **Feast Feature Store** into the existing pipeline.

Feast becomes the single source of truth for feature definitions, allowing both training and inference to consume identical features.

---

# Objectives

- Initialize a Feast Feature Repository
- Define Entities, Data Sources and Feature Views
- Register feature definitions with Feast
- Materialize features into the online store
- Retrieve historical features for training
- Retrieve online features for inference
- Demonstrate consistent predictions using Feast

---

# Existing Week 2 Pipeline

```
Raw CSV
   │
   ▼
Training Script
   │
   ▼
Random Forest Model
   │
   ▼
Model Saved
```

Training directly consumed the CSV file.

---

# Week 3 Pipeline

```
                  Raw CSV
                     │
                     ▼
          prepare_data.py
                     │
                     ▼
            Parquet Dataset
                     │
                     ▼
          Feast Feature Store
         ┌───────────────────┐
         │                   │
         │ Offline Store     │
         │ Historical Data   │
         │                   │
         │ Online Store      │
         │ SQLite            │
         └───────────────────┘
            │           │
            ▼           ▼
     Model Training   Inference
```

Both training and inference now obtain features from Feast.

---

# Repository Structure

```
23f2004912_MLOPS_WEEKLY_ASSIGNMENT/

│
├── data/
│
├── models/
│
├── scripts/
│     ├── train.py
│     ├── train_feast.py
│     ├── inference_feast.py
│     └── test_feast.py
│
├── iris_feature_repo/
│     └── feature_repo/
│            ├── feature_store.yaml
│            ├── feature_definitions.py
│            ├── prepare_data.py
│            └── data/
│                  └── iris_features.parquet
│
└── README.md
```

---

# Task 1 : Initialize Feast Repository

A new Feast Feature Repository was initialized.

```
feast init iris_feature_repo
```

The generated repository contains

- feature_store.yaml
- feature_definitions.py
- data/
- registry
- online store configuration

SQLite was used as the online feature store.

---

# Task 2 : Define Entities, Data Sources and Feature Views

## Entity

Each iris flower is uniquely identified using

```
iris_id
```

This acts as the primary key inside Feast.

---

## Data Source

The original CSV dataset was converted into a Parquet file.

Feast reads feature values from

```
iris_features.parquet
```

instead of directly using the CSV.

---

## Event Timestamp

Feast requires every record to contain an event timestamp.

Since the IRIS dataset is static, an artificial timestamp

```
2025-01-01
```

was added to every row.

This satisfies Feast's temporal requirements.

---

## Feature View

The Feature View contains only the model input features.

```
sepal_length

sepal_width

petal_length

petal_width
```

The target label

```
species
```

is intentionally excluded because it is not an input feature.

---

# Task 3 : Apply Definitions and Materialize Features

Feature definitions were registered using

```
feast apply
```

This created

- Entity
- Feature View
- Registry
- SQLite Online Store

After registration, features were materialized.

```
feast materialize \
"2025-01-01T00:00:00" \
"2025-01-02T00:00:00"
```

Materialization copies feature values from the offline store into the online store for low-latency serving.

---

# Task 4 : Offline Feature Retrieval for Training

Instead of reading the CSV directly,

```
train_feast.py
```

retrieves historical features from Feast.

Workflow

```
CSV

↓

Parquet

↓

Feast Offline Store

↓

Historical Features

↓

Random Forest

↓

Model
```

Training was performed using

```
store.get_historical_features(...)
```

This ensures that the exact same feature definitions are used during training.

---

# Task 5 : Online Feature Retrieval for Inference

During inference, feature values are fetched from the Feast Online Store.

Workflow

```
Entity IDs

↓

SQLite Online Store

↓

Feature Vector

↓

Random Forest Model

↓

Prediction
```

Instead of manually creating feature vectors, the inference script retrieves them using

```
store.get_online_features(...)
```

This guarantees consistency between training and serving.

---

# Files Added

## prepare_data.py

Responsibilities

- Read IRIS CSV
- Generate iris_id
- Add event_timestamp
- Save Parquet dataset

---

## feature_definitions.py

Defines

- Entity
- FileSource
- Feature View

---

## train_feast.py

Responsibilities

- Retrieve historical features
- Train Random Forest
- Save trained model

---

## inference_feast.py

Responsibilities

- Retrieve online features
- Perform prediction
- Compare predictions with actual labels

---

## test_feast.py

Utility script used to verify

- Entity registration
- Feature View registration

---

# Feast Concepts Used

## Feature Repository

Stores all Feast configuration files.

---

## Entity

Represents a uniquely identifiable object.

In this assignment

```
iris_id
```

---

## Data Source

Location where Feast reads raw features.

```
iris_features.parquet
```

---

## Feature View

Logical grouping of model features.

```
sepal_length

sepal_width

petal_length

petal_width
```

---

## Offline Store

Used during training.

Provides historical features.

---

## Online Store

SQLite database.

Used during inference.

Provides low-latency feature lookup.

---

## Materialization

Copies feature values from the Offline Store to the Online Store.

---

## Training-Serving Skew

Training-serving skew occurs when

- training features differ from inference features
- duplicated feature engineering logic becomes inconsistent
- preprocessing pipelines diverge

Using Feast eliminates this issue because both training and inference read features from the same feature definitions.

---

# Challenges Faced

## 1. Missing Week 3 Branch

The Week 3 branch was not available in the repository.

Solution:

Created a new branch from the completed Week 2 implementation.

---

## 2. Feast Template Files

The default Feast repository included sample Driver feature definitions.

Solution:

Removed the example configuration and replaced it with IRIS-specific feature definitions.

---

## 3. Feast Timestamp Requirement

The IRIS dataset contains no timestamp column.

Solution:

Generated a synthetic event timestamp for every record.

---

## 4. NumPy and Scikit-Learn Compatibility

The environment produced

```
numpy.dtype size changed
```

Solution:

Reinstalled compatible versions of NumPy, SciPy and Scikit-Learn.

---

## 5. Missing Entity IDs During Online Retrieval

Some entity IDs returned

```
NaN
```

Reason:

The current DVC dataset version contained only 49 rows.

Solution:

Inference was performed using valid entity IDs present in the dataset.

---

# Learning Outcomes

Through this assignment I learned

- the role of Feature Stores in MLOps
- how Feast separates feature engineering from model training
- defining Entities, Feature Views and Data Sources
- retrieving historical features for training
- materializing features into an online store
- retrieving online features for inference
- the difference between Offline Store and Online Store
- how Feature Stores eliminate training-serving skew
- integrating Feast into an existing ML pipeline without changing the underlying model

---

# Technologies Used

- Python
- Feast 0.64
- Scikit-Learn
- Pandas
- SQLite
- Parquet
- DVC
- Git
- Google Cloud Platform

---

# Conclusion

This assignment successfully integrated **Feast Feature Store** into the existing IRIS machine learning pipeline.

Unlike the Week 2 implementation, where the training pipeline directly consumed CSV files, both training and inference now retrieve features through Feast. Historical features are fetched from the offline store during model training, while low-latency feature retrieval during inference is served from the SQLite online store after materialization.

The integration demonstrates how a Feature Store becomes the single source of truth for feature engineering, significantly reducing the possibility of training-serving skew while making the pipeline more production-ready and easier to maintain.
