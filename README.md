# Week 6 – Continuous Deployment with Kubernetes (GKE)

## Objective

Containerize the IRIS Classification API, push the Docker image to Google Artifact Registry, and automate deployment to Google Kubernetes Engine (GKE) using GitHub Actions.

---

# Project Architecture

```
Developer
    │
    ▼
Git Push (week_6)
    │
    ▼
GitHub Actions (CD)
    │
    ├── Authenticate using Workload Identity Federation
    ├── Build Docker Image
    ├── Push Image to Artifact Registry
    ├── Get GKE Credentials
    └── Rolling Update on Kubernetes
                │
                ▼
        Google Kubernetes Engine
                │
                ▼
         IRIS Prediction API
```

---

# Project Structure

```
.
├── app.py
├── Dockerfile
├── requirements.txt
├── models/
│   └── model.skops
├── data/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── scripts/
└── tests/
```

---

# Technologies Used

- Python
- Flask
- Docker
- Kubernetes
- Google Kubernetes Engine (GKE)
- Google Artifact Registry
- GitHub Actions
- Workload Identity Federation
- DVC
- MLflow
- Google Cloud Platform

---

# Dockerization

The Flask inference API is containerized using Docker.

Build locally

```bash
docker build -t iris-api:v1 .
```

Run locally

```bash
docker run -p 5000:5000 iris-api:v1
```

---

# Kubernetes Deployment

Deployment manifest

```
k8s/deployment.yaml
```

Service manifest

```
k8s/service.yaml
```

Deployment uses

- 1 Replica
- Rolling Update strategy
- Container Port 5000
- ImagePullPolicy Always

---

# Continuous Deployment Workflow

Whenever code is pushed to the **week_6** branch,

GitHub Actions automatically

1. Authenticates to Google Cloud
2. Configures Docker
3. Builds Docker Image
4. Pushes image to Artifact Registry
5. Fetches GKE credentials
6. Updates Kubernetes Deployment
7. Waits for successful rollout

No manual deployment commands are required after the workflow is configured.

---

# GitHub Actions

### CI Workflow

Performs

- Install dependencies
- Authenticate to GCP
- Pull DVC artifacts
- Run tests
- Evaluate model
- Generate metrics
- Create CML report

Workflow file

```
.github/workflows/ci.yml
```

---

### CD Workflow

Performs

- Authenticate to Google Cloud
- Configure Docker
- Build Docker Image
- Push Docker Image
- Fetch GKE Credentials
- Rolling Update on GKE

Workflow file

```
.github/workflows/cd.yml
```

---

# API Endpoints

### Home

```
GET /
```

Response

```json
{
  "message": "IRIS Prediction API is Running!"
}
```

---

### Prediction

```
POST /predict
```

Example

```json
{
    "features":[5.1,3.5,1.4,0.2]
}
```

Response

```json
{
    "prediction":0
}
```

---

# Deployment Verification

Check deployment

```bash
kubectl get deployment
```

Check pods

```bash
kubectl get pods
```

Check services

```bash
kubectl get svc
```

Describe deployment

```bash
kubectl describe deployment iris-api
```

Verify deployed image

```bash
kubectl get deployment iris-api \
-o=jsonpath='{.spec.template.spec.containers[0].image}'
```

---

# Testing the API

Home endpoint

```bash
curl http://<EXTERNAL-IP>/
```

Prediction endpoint

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
-H "Content-Type: application/json" \
-d '{"features":[5.1,3.5,1.4,0.2]}'
```

---

# Authentication

Deployment uses **Workload Identity Federation**, eliminating the need for service account keys.

Authentication flow

GitHub Actions

↓

Workload Identity Provider

↓

GitHub Service Account

↓

Google Cloud Resources

---

# Improvements Implemented

- Feature-name compatible inference using Pandas DataFrame
- Dockerized Flask API
- Artifact Registry integration
- Kubernetes deployment
- External LoadBalancer service
- Rolling updates
- Automated CI pipeline
- Automated CD pipeline
- Workload Identity Federation authentication

---

# Challenges Faced

- Artifact Registry permission issues
- Kubernetes IAM configuration
- GKE authentication plugin installation
- Workload Identity configuration
- GitHub Actions deployment permissions
- MLflow dependency configuration
- Model path compatibility during CI

Each issue was resolved through proper IAM role assignment, GKE authentication setup, dependency management, and workflow refinement.

---

# Outcome

Successfully implemented a complete MLOps deployment pipeline where every push to the **week_6** branch automatically

- Builds a Docker image
- Pushes the image to Artifact Registry
- Deploys the latest version to GKE
- Performs a rolling update without downtime

The deployed API is publicly accessible through a Kubernetes LoadBalancer and supports real-time IRIS species prediction.
