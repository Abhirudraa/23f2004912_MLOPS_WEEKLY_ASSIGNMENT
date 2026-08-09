# MLSecOps — IRIS Data Poisoning Experiment

This project extends the IRIS machine-learning pipeline with basic **MLSecOps** experimentation. It simulates a data-poisoning attack by replacing a percentage of training samples with randomly generated feature values and random class labels.

The experiment evaluates four training-data conditions:

* **0% poisoning** — clean baseline
* **5% poisoning**
* **10% poisoning**
* **50% poisoning**

Each variant is trained using the same Random Forest configuration and evaluated on the same clean evaluation dataset. MLflow is used to track the poisoning level and corresponding model metrics.

## Repository Structure

| File / Directory                     | Utility                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `data/raw/`                          | Contains the original raw IRIS dataset and its DVC metadata.                                                      |
| `data/processed/`                    | Contains the processed training and evaluation datasets used by the ML pipeline.                                  |
| `data/processed/train.csv`           | Clean IRIS training dataset used as the baseline and source for generating poisoned variants.                     |
| `data/processed/eval.csv`            | Clean evaluation dataset used to measure model performance consistently across all experiments.                   |
| `poisoned_data/`                     | Contains the generated poisoned versions of the training dataset.                                                 |
| `poisoned_data/iris_poisoned_0.csv`  | Clean 0% poisoning baseline.                                                                                      |
| `poisoned_data/iris_poisoned_5.csv`  | Training dataset with 5% of samples poisoned.                                                                     |
| `poisoned_data/iris_poisoned_10.csv` | Training dataset with 10% of samples poisoned.                                                                    |
| `poisoned_data/iris_poisoned_50.csv` | Training dataset with 50% of samples poisoned.                                                                    |
| `scripts/poison_data.py`             | Generates the poisoned datasets by replacing all four features and the target label of randomly selected samples. |
| `scripts/mlsecops_experiment.py`     | Trains the Random Forest model on each poisoning level and logs parameters, metrics, and models to MLflow.        |
| `scripts/train.py`                   | Original model-training script from the IRIS ML pipeline.                                                         |
| `scripts/evaluate.py`                | Evaluates the trained IRIS model using the evaluation dataset.                                                    |
| `models/model.skops`                 | Serialized model artifact from the main IRIS pipeline.                                                            |
| `mlflow.db`                          | SQLite backend database containing MLflow experiment tracking information.                                        |
| `setup.ipynb`                        | Notebook used for project setup and experimentation.                                                              |
| `app.py`                             | Application/API component used to serve the IRIS model.                                                           |
| `inference.py`                       | Inference component used to load the model and perform predictions.                                               |
| `tests/`                             | Automated tests for validating the data and model pipeline.                                                       |
| `Dockerfile`                         | Defines the container image used to package the application.                                                      |
| `k8s/`                               | Kubernetes deployment configuration for the IRIS inference service.                                               |
| `k8s/deployment.yaml`                | Defines the Kubernetes deployment for the IRIS API.                                                               |
| `k8s/service.yaml`                   | Exposes the IRIS API through a Kubernetes service.                                                                |
| `k8s/hpa.yaml`                       | Configures horizontal pod autoscaling for the deployed service.                                                   |
| `stress/`                            | Contains stress-testing resources for the deployed inference API.                                                 |
| `stress/post.lua`                    | Lua script used with `wrk` for HTTP POST load testing.                                                            |
| `.github/workflows/ci.yml`           | GitHub Actions workflow for continuous integration and automated testing.                                         |
| `.github/workflows/cd.yml`           | GitHub Actions workflow for continuous deployment.                                                                |
| `.dvc/`                              | DVC configuration and metadata used for data versioning.                                                          |
| `.dvcignore`                         | Specifies files/directories ignored by DVC.                                                                       |
| `requirements.txt`                   | Python dependencies required to run the project.                                                                  |
| `LICENSE`                            | Project license.                                                                                                  |

## MLSecOps Experiment

### Data Poisoning

`scripts/poison_data.py` selects a percentage of samples from the clean training dataset and replaces:

* `sepal_length`
* `sepal_width`
* `petal_length`
* `petal_width`

with randomly generated values between 0 and 8.

The original target label is also replaced with a randomly selected class.

Using the clean training dataset of 80 samples, the experiment generates:

| Poisoning Level | Poisoned Samples | Clean Samples |
| --------------: | ---------------: | ------------: |
|              0% |                0 |            80 |
|              5% |                4 |            76 |
|             10% |                8 |            72 |
|             50% |               40 |            40 |

### MLflow Experiment

`scripts/mlsecops_experiment.py` trains the same Random Forest configuration on all four datasets.

For every run, MLflow records:

**Parameters**

* `poisoning_level`
* `training_samples`
* `poisoned_samples`
* `clean_samples`
* `n_estimators`
* `max_depth`
* `random_state`

**Metrics**

* Accuracy
* Precision
* Recall
* F1 score

All models are evaluated against the same clean evaluation dataset so that the effect of poisoning can be compared consistently.

## Security Threats Covered

The Week 8 exercise considers four major ML security threat vectors:

1. **Data Poisoning** — manipulation of training data.
2. **Adversarial Examples** — manipulated inference inputs designed to cause incorrect predictions.
3. **Model Extraction** — repeated querying of a model to reconstruct its behavior.
4. **Prompt Injection** — malicious instructions targeting LLM-based systems.

The practical implementation in this repository focuses specifically on **data poisoning**.

## Mitigation Considerations

Potential production mitigations include:

* Schema enforcement
* Feature-range validation
* Statistical distribution checks
* Anomaly detection
* Data provenance tracking
* Automated data-quality gates
* Quarantining suspicious samples before training

The experiment also demonstrates that simply collecting more data does not necessarily solve a data-quality problem. If newly collected data is also poisoned, the additional quantity can increase the amount of contaminated information. Maintaining a sufficient quantity of **clean, trustworthy data** is therefore important for reliable model training.
