# IRIS Classifier — Model Card

## 1. Model Overview

The IRIS classifier is a multi-class machine learning model that predicts
the species of an iris flower from four botanical measurements.

The classifier is implemented using a Random Forest model.

The three prediction classes are:

- Setosa
- Versicolor
- Virginica

## 2. Intended Use

The model is intended for educational and demonstration purposes to
illustrate a production machine learning pipeline involving:

- Model training
- Evaluation
- Explainability
- Fairness auditing
- Data drift monitoring
- ML governance

It should not be used for high-stakes real-world decisions.

## 3. Training Data

The model is trained using the IRIS dataset.

The four input features are:

- sepal_length
- sepal_width
- petal_length
- petal_width

The target variable represents the iris species and is encoded as:

- 0 — Setosa
- 1 — Versicolor
- 2 — Virginica

A stratified train/test split with a test size of 20% and random seed
42 was used.

## 4. Model

The classifier is a RandomForestClassifier.

The existing training pipeline evaluates multiple combinations of:

- Number of trees: 50 and 100
- Maximum depth: 3 and 5

The trained model is stored using the `skops` format.

## 5. Overall Evaluation Performance

On the evaluation dataset used for the Week 9 fairness analysis:

| Metric | Score |
|---|---:|
| Accuracy | 0.9524 |
| Macro Precision | 0.9583 |
| Macro Recall | 0.9444 |

## 6. Fairness Evaluation

A randomly assigned `location` attribute with values 0 and 1 was
introduced for fairness auditing.

The location attribute was deliberately excluded from model training
and prediction.

Fairlearn's `MetricFrame` was used to calculate metrics separately
for the two location groups.

| Metric | Location 0 | Location 1 |
|---|---:|---:|
| Accuracy | 1.0000 | 0.9231 |
| Precision | 1.0000 | 0.9333 |
| Recall | 1.0000 | 0.9167 |

The observed differences were:

| Metric | Difference |
|---|---:|
| Accuracy | 0.0769 |
| Precision | 0.0667 |
| Recall | 0.0833 |

The sensitive attribute was randomly assigned and was not used as a
model feature. Therefore, these differences should not be interpreted
as evidence that location causes discriminatory behavior. The
evaluation dataset is small, and random group assignment can produce
different subgroup compositions and performance.

The fairness analysis demonstrates why subgroup performance should be
monitored independently of overall model accuracy.

## 7. Explainability

SHAP was used to generate full-dataset feature attribution summaries
for all three prediction classes:

- Setosa
- Versicolor
- Virginica

The SHAP summary plots show the contribution of each feature to the
model's prediction.

For the Virginica class:

- Positive SHAP values push the prediction toward Virginica.
- Negative SHAP values push the prediction away from Virginica.
- Red points represent high feature values.
- Blue points represent low feature values.
- Features are ordered according to their overall contribution.

The SHAP analysis provides post-hoc explanations of the Random Forest
model's predictions.

## 8. Drift Monitoring

Production data was simulated by shifting the `petal_length`
distribution by +1.0.

The Kolmogorov-Smirnov two-sample test was used to compare the training
and simulated production distributions.

| Feature | KS Statistic | p-value | Drift |
|---|---:|---:|---|
| sepal_length | 0.00 | 1.000000 | No |
| sepal_width | 0.00 | 1.000000 | No |
| petal_length | 0.35 | 0.000096 | Yes |
| petal_width | 0.00 | 1.000000 | No |

The `petal_length` distribution shows statistically significant drift
because its p-value is below the 0.05 significance threshold.

This experiment demonstrates data drift rather than concept drift.
The input distribution was changed, but the relationship between the
features and target was not independently changed.

## 9. Limitations

- The IRIS dataset is small and may not represent real production data.
- The fairness evaluation contains a small number of samples per
  location group.
- The location attribute was randomly generated and is therefore only
  a demonstration sensitive attribute.
- The drift experiment uses synthetic distribution shifts.
- Detecting data drift does not necessarily mean that model accuracy
  has degraded.
- Concept drift cannot be established without monitoring the
  relationship between inputs and ground-truth outcomes over time.
- SHAP provides post-hoc explanations and does not prove causal
  relationships between features and predictions.

## 10. Governance Considerations

For a production deployment, the following should be monitored:

1. Overall model performance.
2. Performance across sensitive groups.
3. Feature distribution drift.
4. Prediction distribution drift.
5. Ground-truth performance when labels become available.
6. Model and data versions.
7. Model limitations and intended use.

Significant fairness gaps or data drift should trigger investigation
before automatically promoting or continuing to rely on a model.
