# Week 1 - IRIS ML Pipeline on Vertex AI

## Overview

This project implements an end-to-end IRIS classification pipeline using:

- Google Cloud Vertex AI Workbench
- Google Cloud Storage (GCS)
- Scikit-Learn
- Python

## Repository Structure

```text
.
├── setup.ipynb
├── inference.py
|── LICENSE
└── README.md
```
## File Descriptions
# setup.ipynb

Main notebook used for:

- Configuring GCP and Vertex AI environment
- Uploading datasets to Google Cloud Storage
- Loading and preprocessing data
- Training the Decision Tree classifier
- Evaluating model performance
- Saving model artifacts
- Uploading trained model artifacts to GCS
- Running multiple experiment executions
  
# inference.py

Standalone inference script used for:

- Loading a trained model artifact from GCS
- Running predictions on evaluation datasets
- Computing evaluation metrics
- Supporting reproducible inference independent of the training notebook

## Tasks Completed

### Task 1
- Set up GCP Project
- Created Vertex AI Workbench instance
- Enabled required APIs

### Task 2
- Created GCS bucket
- Uploaded IRIS dataset to GCS

### Task 3
- Trained Decision Tree Classifier
- Saved model as `model.joblib`
- Uploaded model artifacts to GCS

### Task 4
- Created separate `inference.py`
- Loaded trained model from GCS
- Ran inference on evaluation dataset

### Task 5
- Executed training and inference multiple times
- Stored outputs in timestamp-based folders

### Task 6 (Optional)
- Evaluated model on:
  - `v1.csv`
  - `v2.csv`
- Compared results across data versions

## Model

- Algorithm: Decision Tree Classifier
- Accuracy: ~98.3%

## Author

Abhijeet Kumar

IIT Madras BS Degree Program

Roll Number: 23f2004912
