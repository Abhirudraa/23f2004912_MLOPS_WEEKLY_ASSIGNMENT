# 23f2004912_MLOPS_WEEKLY_ASSIGNMENT

## Week 2 Assignment: Integrating DVC into the IRIS Pipeline

**Author:** Abhirudraa  
**Roll Number:** 23f2004912  
**Course:** MLOps (Week 2 Assignment)  
**Repository:** [https://github.com/Abhirudraa/23f2004912_MLOPS_WEEKLY_ASSIGNMENT](https://github.com/Abhirudraa/23f2004912_MLOPS_WEEKLY_ASSIGNMENT)

---

## 📋 Assignment Overview

This assignment demonstrates the integration of **Data Version Control (DVC)** into a machine learning pipeline using the classic **IRIS dataset**. The objective is to create a reproducible, version-controlled workflow for both data and model artifacts, backed by **Google Cloud Storage (GCS)** .

### Key Objectives
- ✅ Initialize DVC in a Git repository
- ✅ Configure GCS as a DVC remote
- ✅ Version data and models across multiple iterations
- ✅ Switch between data/model versions using DVC checkout
- ✅ Push all relevant files to Git and GCS

---
# Repo structure 
23f2004912_MLOPS_WEEKLY_ASSIGNMENT/
├── data/
│   ├── iris.csv.dvc         ← ✅ Pointer file (small, tracked by Git)
│   └── .gitignore           ← ✅ Prevents actual data from Git
├── models/
│   ├── model.joblib.dvc     ← ✅ Pointer file (small, tracked by Git)
│   └── .gitignore           ← ✅ Prevents actual model from Git
├── inference.py             ← ✅ Code (tracked by Git)
├── metrics.csv              ← ✅ Metrics (small file, tracked by Git)
└── setup.ipynb

## ON GCS
gs://mlops-course-week2-ad35ebae-2568-47b9-83a-unique/iris-pipeline/
└── files/
    └── md5/
        ├── [hash_49ba...]/  ← v1/data.csv (actual data)
        ├── [hash_54d4...]/  ← raw/iris.csv (actual data)
        └── [hash_xxxx...]/  ← v2/data.csv (actual data)

## 📄 File Descriptions

### Core Configuration Files

| File | Description | Purpose |
|------|-------------|---------|
| `.dvc/` | DVC internal directory | Stores DVC configuration, cache, and state |
| `.dvc/.gitignore` | DVC cache ignore | Prevents large cache files from entering Git |
| `.dvc/config` | DVC remote configuration | Stores GCS remote URL and settings |
| `.dvcignore` | DVC ignore file | Specifies files DVC should ignore |
| `.gitignore` | Git ignore file | Prevents large files from being committed to Git |

### Data Files

| File | Description | Details |
|------|-------------|---------|
| `data/raw/iris.csv` | Original IRIS dataset | 150 samples, 5 columns (sepal_length, sepal_width, petal_length, petal_width, species) |
| `data/v1/data.csv` | Augmented dataset v1 | First augmented version with additional samples |
| `data/v2/data.csv` | Augmented dataset v2 | Second augmented version |
| `data/iris.csv` | Active data file | Points to current version being used |
| `data/iris.csv.dvc` | DVC pointer file | Contains MD5 hash pointing to data in GCS |

### Model Files

| File | Description | Details |
|------|-------------|---------|
| `models/model.joblib` | Trained model | Random Forest classifier saved with joblib |
| `models/model.joblib.dvc` | DVC pointer file | Contains MD5 hash pointing to model in GCS |

### Scripts

| File | Description | Purpose |
|------|-------------|---------|
| `scripts/train.py` | Training script | Loads data, trains Random Forest model, saves model and metrics |

### Output Files

| File | Description | Details |
|------|-------------|---------|
| `metrics.csv` | Model accuracy | Contains accuracy score for the current model version |

### Version Tags

| Tag | Description | Data Version | Accuracy |
|-----|-------------|--------------|----------|
| `v1.0` | Model trained on raw IRIS dataset | raw/iris.csv (150 samples) | 1.0000 |
| `v2.0` | Model trained on augmented v1 dataset | v1/data.csv | 0.9048 |
| `v3.0` | Model trained on v2 dataset | v2/data.csv | 1.0000 |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Git
- DVC (Data Version Control)
- Google Cloud SDK (for GCS access)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Abhirudraa/23f2004912_MLOPS_WEEKLY_ASSIGNMENT.git
cd 23f2004912_MLOPS_WEEKLY_ASSIGNMENT
git checkout week_2


# Install Dependencies
pip install -r requirements.txt
pip install "dvc[gs]"

# Configure DVC Remote (if pushing/pulling from GCS)

dvc remote add -d gcs_remote gs://your-bucket-name/iris-pipeline

# Pull Data and Models from GCS
dvc pull

# Train the Model

python scripts/train.py data/iris.csv
```
