# Two ML Case Studies — Understanding Report

## 1. Hospital Readmission Prediction

### Problem
Predict whether a patient will be readmitted within 30 days after discharge.

### Model
**Logistic Regression with L2 regularization.** Logistic regression converts a weighted combination of patient features into a probability. L2 adds a penalty on large coefficients, reducing overfitting and helping when features are correlated.

### Features
Age, heart rate, systolic blood pressure, glucose, oxygen saturation, length of stay, number of diagnoses, diagnosis category, insulin use, and prior inpatient/emergency/outpatient visits.

### ROC-AUC
The trained run produced ROC-AUC **0.693** on the held-out test set. A threshold near 80% recall was selected to demonstrate a clinically sensitive operating point.

### False negatives vs false positives
A **false negative** is a patient who is actually at risk but predicted low risk. In a real hospital workflow this could mean missing an opportunity for discharge planning or follow-up. A **false positive** is a patient predicted high risk who would not be readmitted; this can consume care-coordination resources and create unnecessary interventions. The right threshold is therefore a cost-sensitive clinical decision, not simply 0.5.

> This implementation is an educational simulation. It is not clinically validated and must not be used for patient care.

## 2. Credit Card Fraud Detection

### Problem
Fraud is rare, so accuracy can be misleading. The project demonstrates a heavily imbalanced binary classification workflow.

### Model
**XGBoost + SMOTE.** SMOTE creates synthetic minority-class training examples. It must be applied only to the training partition; applying it before the split can leak information into the test set.

### Threshold tuning
The model outputs a fraud probability. The classification threshold was tuned instead of blindly using 0.5. The selected threshold maximized F1 on the held-out predictions for this educational experiment.

### Metrics
The trained run produced **ROC-AUC 0.997** and **PR-AUC 0.930**, with precision **0.900**, recall **0.840**, and F1 **0.869** at the selected threshold.

PR-AUC is particularly informative in rare-event detection because it focuses on the precision/recall trade-off for the minority class.

### Feature importance
XGBoost feature importance is saved in `results/feature_importance.csv`. The dataset is synthetic and deliberately injects signal into selected anonymized V-features, so these importances demonstrate the technique rather than making claims about real fraud drivers.

## Dataset note
The repositories generated for this assignment do **not** redistribute raw public datasets. Instead they contain reproducible synthetic datasets with the same general structure. This keeps the repositories lightweight and runnable without credentials.

## How to run

```bash
pip install -r requirements.txt
python src/train.py
```

Each project saves a trained `.joblib` model and evaluation artifacts under `models/` and `results/`.
