# Case Study 1 — Hospital Readmission Prediction

## Objective
Predict whether a patient will be readmitted within 30 days.

## Why logistic regression + L2?
Logistic regression produces a probability that can be mapped to a risk threshold. L2 regularization penalizes large coefficients and helps reduce overfitting when predictors are correlated or numerous.

## Features
Diagnosis category, age, heart rate, systolic blood pressure, glucose, oxygen saturation, length of stay, number of diagnoses, and prior outpatient/emergency/inpatient visits.

## Evaluation
ROC-AUC = **0.693**. A threshold was selected around 80% recall: threshold = **0.4164**, recall = **0.800**, precision = **0.305**.

## Clinical cost discussion
A false negative means a high-risk patient is classified as low risk and may miss extra follow-up. A false positive means a lower-risk patient receives additional screening/care coordination. The appropriate threshold therefore depends on the relative clinical and operational costs; it should be chosen with clinicians and validated prospectively, not from this educational dataset.
