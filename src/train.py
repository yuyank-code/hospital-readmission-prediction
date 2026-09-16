from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Project paths and configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for directory in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES = 30_000
TEST_SIZE = 0.20


# -----------------------------------------------------------------------------
# 1. Create a reproducible synthetic patient dataset
# -----------------------------------------------------------------------------
def create_dataset(n_samples=N_SAMPLES):
    """Create synthetic patient records for the educational case study."""

    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 91, n_samples)
    heart_rate = rng.normal(78, 13, n_samples).clip(45, 150)
    systolic_bp = rng.normal(125, 18, n_samples).clip(75, 210)
    glucose = rng.normal(125, 45, n_samples).clip(50, 450)
    spo2 = rng.normal(96.5, 2, n_samples).clip(80, 100)

    prior_inpatient = rng.poisson(1.2, n_samples)
    prior_emergency = rng.poisson(1.7, n_samples)
    prior_outpatient = rng.poisson(2.5, n_samples)

    length_of_stay = rng.gamma(2.2, 2.2, n_samples).clip(1, 30)
    number_of_diagnoses = rng.poisson(5, n_samples).clip(1, 16)

    diagnosis = rng.choice(
        ["circulatory", "diabetes", "respiratory", "renal", "digestive", "other"],
        n_samples,
        p=[0.24, 0.20, 0.18, 0.10, 0.10, 0.18],
    )

    gender = rng.choice(["Female", "Male"], n_samples)
    insulin = rng.choice(["No", "Yes"], n_samples, p=[0.55, 0.45])

    # Synthetic risk-generating process used only to create the target variable.
    logit = (
        -4.1
        + 0.018 * age
        + 0.55 * (diagnosis == "circulatory")
        + 0.45 * (diagnosis == "renal")
        + 0.35 * (diagnosis == "respiratory")
        + 0.10 * number_of_diagnoses
        + 0.18 * prior_inpatient
        + 0.11 * prior_emergency
        + 0.04 * prior_outpatient
        + 0.07 * length_of_stay
        + 0.004 * (glucose - 120)
        - 0.045 * (spo2 - 95)
        + 0.006 * (heart_rate - 75)
        + 0.002 * (systolic_bp - 120)
        + 0.55 * (insulin == "Yes")
    )

    probability = 1 / (1 + np.exp(-logit))
    readmitted = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "spo2": spo2,
            "prior_inpatient_visits": prior_inpatient,
            "prior_emergency_visits": prior_emergency,
            "prior_outpatient_visits": prior_outpatient,
            "length_of_stay": length_of_stay,
            "number_of_diagnoses": number_of_diagnoses,
            "diagnosis_category": diagnosis,
            "gender": gender,
            "insulin": insulin,
            "readmitted_30d": readmitted,
        }
    )


# -----------------------------------------------------------------------------
# 2. Build the preprocessing + Logistic Regression pipeline
# -----------------------------------------------------------------------------
def build_model(numeric_features, categorical_features):
    """Build preprocessing and L2-regularized logistic regression."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


# -----------------------------------------------------------------------------
# 3. Train, evaluate, and save the model
# -----------------------------------------------------------------------------
def main():
    df = create_dataset()
    df.to_csv(DATA_DIR / "synthetic_patient_records.csv", index=False)

    target = "readmitted_30d"
    X = df.drop(columns=target)
    y = df[target]

    categorical_features = ["diagnosis_category", "gender", "insulin"]
    numeric_features = [
        column for column in X.columns if column not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Select a threshold that gives approximately 80% recall.
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    threshold_index = np.argmin(np.abs(tpr - 0.80))
    threshold = float(thresholds[threshold_index])

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "n_total": int(len(df)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold_for_~80pct_recall": threshold,
        "precision_at_threshold": float(precision_score(y_test, predictions)),
        "recall_at_threshold": float(recall_score(y_test, predictions)),
        "f1_at_threshold": float(f1_score(y_test, predictions)),
        "confusion_matrix": matrix.tolist(),
    }

    joblib.dump(model, MODEL_DIR / "logistic_l2_readmission.joblib")

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save ROC curve.
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("30-Day Readmission ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # Save coefficient ranking for model interpretation.
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    coefficient_df.sort_values(
        "abs_coefficient", ascending=False
    ).to_csv(RESULTS_DIR / "top_coefficients.csv", index=False)

    print("Training complete.")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Recall:   {metrics['recall_at_threshold']:.3f}")
    print(f"Precision:{metrics['precision_at_threshold']:.3f}")


if __name__ == "__main__":
    main()
