# MLOps Weekly Assignment – Week 7
**Stress Testing, Observability & Scaling the IRIS Pipeline**

## Project Overview

This project demonstrates a complete MLOps deployment pipeline for an IRIS classification API on Google Kubernetes Engine (GKE). The pipeline includes Continuous Integration (CI), Continuous Deployment (CD), Docker containerization, Kubernetes deployment, Horizontal Pod Autoscaling (HPA), stress testing using **wrk**, and monitoring through Google Cloud Monitoring and Cloud Logging.

---

# Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── data/
├── k8s/
├── models/
├── scripts/
├── stress/
├── tests/
├── app.py
├── Dockerfile
├── inference.py
├── README.md
├── requirements.txt
└── ...
```

---

# File Descriptions

## Application Files

### `app.py`
Flask application exposing the REST API for IRIS prediction. It loads the trained model and provides the `/predict` endpoint.

### `inference.py`
Utility script used for loading the trained model and performing inference outside the web application.

### `requirements.txt`
Lists all Python dependencies required for running the application and training pipeline.

### `Dockerfile`
Builds the Docker image for the IRIS prediction API.

---

## Machine Learning Files

### `models/model.skops`
Serialized trained machine learning model used by the prediction API.

### `scripts/train.py`
Trains the machine learning model using the IRIS dataset.

### `scripts/evaluate.py`
Evaluates the trained model and reports performance metrics.

---

## Dataset

### `data/raw/`
Contains the original IRIS dataset tracked using DVC.

### `data/processed/`
Contains processed training and evaluation datasets used during model training.

---

## Kubernetes Configuration

### `k8s/deployment.yaml`
Defines the Kubernetes Deployment for the IRIS API including container image, resource requests, limits, and replica configuration.

### `k8s/service.yaml`
Creates a Kubernetes LoadBalancer Service to expose the application publicly.

### `k8s/hpa.yaml`
Configures the Horizontal Pod Autoscaler (HPA) to automatically scale the deployment based on CPU utilization.

---

## Stress Testing

### `stress/post.lua`
Lua script used by **wrk** to send HTTP POST requests with JSON payloads to the `/predict` endpoint during stress testing.

---

## CI/CD

### `.github/workflows/ci.yml`
Implements the Continuous Integration pipeline.

Functions include:

- Checkout repository
- Install dependencies
- Authenticate with Google Cloud
- Pull DVC artifacts
- Execute unit tests
- Evaluate model
- Generate CI report

### `.github/workflows/cd.yml`
Implements the Continuous Deployment pipeline.

Functions include:

- Authenticate with Google Cloud
- Build Docker image
- Push image to Artifact Registry
- Deploy latest image to Google Kubernetes Engine
- Verify deployment
- Install **wrk**
- Execute automated stress testing after deployment

---

## Testing

### `tests/test_data.py`
Unit tests for validating the processed dataset.

### `tests/test_model.py`
Unit tests for validating the trained machine learning model.

---

## Experiment Tracking

### `mlruns/`
Stores MLflow experiment runs and model artifacts.

### `mlflow.db`
SQLite database used by MLflow for experiment tracking.

---

## Documentation

### `README.md`
Project documentation describing the repository structure, setup process, deployment workflow, and utility of each file.

---

# Technologies Used

- Python
- Flask
- scikit-learn
- DVC
- MLflow
- Docker
- Kubernetes (GKE)
- Google Cloud Platform
- GitHub Actions
- Artifact Registry
- Horizontal Pod Autoscaler (HPA)
- Cloud Monitoring
- Cloud Logging
- wrk

---

# Assignment Objectives Achieved

- Dockerized the IRIS prediction API.
- Deployed the application on Google Kubernetes Engine.
- Automated CI/CD using GitHub Actions.
- Configured Horizontal Pod Autoscaler (HPA).
- Performed stress testing using **wrk**.
- Monitored resource utilization through Google Cloud Monitoring.
- Inspected application logs using Google Cloud Logging.
- Compared system performance under scalable and constrained deployment scenarios.

---

# Stress Test Configuration

### Scenario 1

- Threads: 4
- Connections: 1000
- Duration: 30 seconds
- HPA: Minimum 1 replica, Maximum 3 replicas

### Scenario 2

- Threads: 4
- Connections: 2000
- Duration: 30 seconds
- HPA restricted to 1 replica

These experiments demonstrate Kubernetes autoscaling behavior and identify performance bottlenecks under constrained resources.
