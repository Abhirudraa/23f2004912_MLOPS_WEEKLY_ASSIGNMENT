# MLOps Weekly Assignment – Week 4: Continuous Integration for ML Pipelines

## Student Information

* **Name:** Abhijeet Kumar
* **Roll Number:** 23F2004912
* **Course:** MLOps
* **Institute:** IIT Madras BS Degree Program in Data Science and Applications

---

# Project Overview

This project implements a complete **Machine Learning Continuous Integration (CI) pipeline** for the Iris Classification problem. The workflow combines **GitHub Actions**, **DVC**, **Google Cloud Storage (GCS)**, **Workload Identity Federation (OIDC)**, **pytest**, and **CML** to automatically validate data, evaluate model performance, and report results on every code change.

The objective is to ensure that every push and pull request is automatically tested before being merged into the main branch.

---

# Project Structure

```text
.
├── data
│   ├── raw
│   │   ├── iris.csv
│   │   └── iris.csv.dvc
│   └── processed
│       ├── train.csv
│       ├── train.csv.dvc
│       ├── eval.csv
│       └── eval.csv.dvc
├── models
│   ├── model.joblib
│   └── model.joblib.dvc
├── scripts
│   ├── train.py
│   └── evaluate.py
├── tests
│   ├── test_data.py
│   └── test_model.py
├── inference.py
├── setup.ipynb
├── README.md
└── LICENSE
```

---

# Technologies Used

* Python
* Scikit-learn
* Pytest
* DVC
* Google Cloud Storage (GCS)
* GitHub Actions
* Google Workload Identity Federation (OIDC)
* Continuous Machine Learning (CML)

---

# Assignment Tasks Completed

## Task 1 – Data Validation Tests

Implemented unit tests to validate:

* Dataset schema
* Missing values
* Feature data types
* Valid feature ranges

---

## Task 2 – Model Evaluation Tests

Implemented tests that:

* Load the trained model
* Run inference on evaluation data
* Verify evaluation metrics satisfy minimum thresholds

Metrics evaluated include:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Task 3 – GitHub Actions with DVC

Configured GitHub Actions to:

* Checkout repository
* Install dependencies
* Authenticate with Google Cloud using Workload Identity Federation
* Pull versioned datasets and models using DVC
* Execute automated tests

---

## Task 4 – Continuous Integration

Configured CI to execute automatically on:

* Every Push
* Every Pull Request

The workflow is enabled for both the **main** branch and all **week_*** branches.

---

## Task 5 – CML Reporting

Integrated Continuous Machine Learning (CML) to automatically publish:

* Model evaluation metrics
* Test execution summary

as part of the Pull Request workflow.

Example report:

```text
Accuracy : 0.952
Precision: 0.958
Recall   : 0.952
F1 Score : 0.952

All tests passed.
```

---

## Task 6 – Pull Request Workflow

Successfully completed the full CI workflow:

* Created Pull Request
* Triggered GitHub Actions
* Executed automated tests
* Generated evaluation report
* Merged `week_4` into `main`

---

# GitHub Actions Workflow

The CI pipeline performs the following steps automatically:

```text
Push / Pull Request
        │
        ▼
GitHub Actions
        │
        ├── Authenticate with Google Cloud (OIDC)
        ├── Configure DVC
        ├── Pull Data & Model
        ├── Run Data Validation Tests
        ├── Run Model Evaluation Tests
        ├── Evaluate Model
        ├── Generate Metrics
        └── Publish CML Report
```

---

# Model Performance

| Metric    | Value |
| --------- | ----: |
| Accuracy  | 0.952 |
| Precision | 0.958 |
| Recall    | 0.952 |
| F1 Score  | 0.952 |

---

# Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd 23f2004912_MLOPS_WEEKLY_ASSIGNMENT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull versioned artifacts:

```bash
dvc pull
```

Run tests:

```bash
pytest
```

Evaluate the model:

```bash
python scripts/evaluate.py
```

---

# Learning Outcomes

Through this assignment, the following concepts were implemented:

* Continuous Integration using GitHub Actions
* Data and model versioning with DVC
* Secure Google Cloud authentication using Workload Identity Federation
* Automated model validation with Pytest
* Continuous Machine Learning reporting with CML
* Automated quality checks before merging code

---

# Author

**Abhijeet Kumar**

Roll Number: **23F2004912**

IIT Madras BS Degree Program in Data Science and Applications
