# Week 9 – Explainability, Fairness, and Drift in the IRIS Pipeline

## Overview

Week 9 extends the existing IRIS machine learning pipeline with responsible ML capabilities covering:

- Fairness assessment using Fairlearn
- Model explainability using SHAP
- Data drift detection using statistical tests
- Model governance through a model card

The analysis is performed on the existing IRIS Random Forest classifier. The sensitive `location` attribute is used only for fairness auditing and is not included as a model feature.

---

## Week 9 Directory Structure

```text
week9/
├── fairness_analysis.py
├── shap_analysis.py
├── drift_analysis.py
├── model_card.md
├── eval_with_location.csv
├── production.csv
└── plots/
    ├── shap_setosa.png
    ├── shap_versicolor.png
    ├── shap_virginica.png
    ├── drift_sepal_length.png
    ├── drift_sepal_width.png
    ├── drift_petal_length.png
    └── drift_petal_width.png
