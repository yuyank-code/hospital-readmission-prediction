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

# ============================================================
# 1. DATA LOAD
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("1. DATA LOAD")
print("=" * 70)

print(f"Loading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("2. DATA CLEANING / PRE-PROCESSING")
print("=" * 70)

# '?' represents missing values in the original dataset.
df = df.replace("?", np.nan)

# Create the 30-day readmission target.
# <30 = 1
# NO and >30 = 0
df["readmitted_30"] = (
    df["readmitted"] == "<30"
).astype(int)

print("\nTarget distribution:")
print(df["readmitted_30"].value_counts())

print("\nTarget percentage:")
print(
    (df["readmitted_30"].value_counts(normalize=True) * 100)
    .round(2)
)


# Remove original target column.
df = df.drop(columns=["readmitted"])


# Remove identifiers.
# These identify encounters/patients rather than representing
# useful clinical predictors.
df = df.drop(
    columns=[
        "encounter_id",
        "patient_nbr",
    ],
    errors="ignore",
)


# Remove columns with very high missingness.
df = df.drop(
    columns=[
        "weight",
        "payer_code",
        "medical_specialty",
    ],
    errors="ignore",
)


X = df.drop(columns=["readmitted_30"])
y = df["readmitted_30"]

print("\nFeatures after cleaning:", X.shape[1])
print("Samples:", X.shape[0])


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

print("\nTraining target distribution:")
print(y_train.value_counts())


# ============================================================
# 4. PRE-PROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("4. PRE-PROCESSING")
print("=" * 70)

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

print("Pre-processing pipeline created.")


# ============================================================
# 5. LOGISTIC REGRESSION - WITHOUT L2
# ============================================================

print("\n" + "=" * 70)
print("5. TRAINING - LOGISTIC REGRESSION WITHOUT L2")
print("=" * 70)

model_without_l2 = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=2000,
            ),
        ),
    ]
)

print("Training WITHOUT L2...")

model_without_l2.fit(
    X_train,
    y_train,
)

prob_without_l2 = model_without_l2.predict_proba(
    X_test
)[:, 1]

pred_without_l2 = (
    prob_without_l2 >= 0.50
).astype(int)

print("WITHOUT L2 training complete.")


# ============================================================
# 6. LOGISTIC REGRESSION - WITH L2
# ============================================================

print("\n" + "=" * 70)
print("6. TRAINING - LOGISTIC REGRESSION WITH L2")
print("=" * 70)

model_with_l2 = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
            ),
        ),
    ]
)

print("Training WITH L2...")

model_with_l2.fit(
    X_train,
    y_train,
)

prob_with_l2 = model_with_l2.predict_proba(
    X_test
)[:, 1]

pred_with_l2 = (
    prob_with_l2 >= 0.50
).astype(int)

print("WITH L2 training complete.")


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    probabilities,
    predictions,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"True Negatives:      {tn:,}")
    print(f"False Positives:     {fp:,}")
    print(f"False Negatives:     {fn:,}")
    print(f"True Positives:      {tp:,}")

    print(
        f"\nFalse Negative Rate:  "
        f"{false_negative_rate:.4f}"
    )

    return {
        "model": name,
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_negative_rate": float(
            false_negative_rate
        ),
    }


# ============================================================
# 8. EVALUATE BOTH MODELS
# ============================================================

print("\n" + "=" * 70)
print("7. MODEL EVALUATION")
print("=" * 70)

results_without_l2 = evaluate_model(
    "LOGISTIC REGRESSION WITHOUT L2",
    y_test,
    prob_without_l2,
    pred_without_l2,
)

results_with_l2 = evaluate_model(
    "LOGISTIC REGRESSION WITH L2",
    y_test,
    prob_with_l2,
    pred_with_l2,
)


# ============================================================
# 9. ROC-AUC CURVE
# ============================================================

print("\n" + "=" * 70)
print("8. ROC-AUC")
print("=" * 70)

fpr_without, tpr_without, _ = roc_curve(
    y_test,
    prob_without_l2,
)

fpr_with, tpr_with, _ = roc_curve(
    y_test,
    prob_with_l2,
)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr_without,
    tpr_without,
    label=(
        f"Without L2 "
        f"(AUC = {results_without_l2['roc_auc']:.4f})"
    ),
)

plt.plot(
    fpr_with,
    tpr_with,
    label=(
        f"With L2 "
        f"(AUC = {results_with_l2['roc_auc']:.4f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - 30-Day Hospital Readmission"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = (
    RESULTS_DIR /
    "roc_curve_comparison.png"
)

plt.savefig(
    roc_path,
    dpi=200,
)

plt.close()

print(
    f"ROC curve saved: {roc_path}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

comparison = pd.DataFrame(
    [
        results_without_l2,
        results_with_l2,
    ]
)

comparison.to_csv(
    RESULTS_DIR /
    "model_comparison.csv",
    index=False,
)


fn_analysis = pd.DataFrame(
    [
        {
            "model": "Without L2",
            "false_negatives":
                results_without_l2[
                    "false_negative"
                ],
            "false_negative_rate":
                results_without_l2[
                    "false_negative_rate"
                ],
        },
        {
            "model": "With L2",
            "false_negatives":
                results_with_l2[
                    "false_negative"
                ],
            "false_negative_rate":
                results_with_l2[
                    "false_negative_rate"
                ],
        },
    ]
)

fn_analysis.to_csv(
    RESULTS_DIR /
    "false_negative_analysis.csv",
    index=False,
)


# ============================================================
# 11. SAVE METRICS JSON
# ============================================================

metrics = {
    "dataset": {
        "name":
            "Diabetes 130-US Hospitals",
        "rows":
            int(len(df)),
        "target":
            "30-day readmission",
        "positive_class":
            "<30 days",
    },
    "models": {
        "without_l2":
            results_without_l2,
        "with_l2":
            results_with_l2,
    },
}

with open(
    RESULTS_DIR / "metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# 12. SAVE MODELS
# ============================================================

print("\n" + "=" * 70)
print("9. SAVING MODELS")
print("=" * 70)

joblib.dump(
    model_without_l2,
    MODELS_DIR /
    "logistic_without_l2.joblib",
)

joblib.dump(
    model_with_l2,
    MODELS_DIR /
    "logistic_l2_readmission.joblib",
)

print("Models saved.")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    RESULTS_DIR /
    "model_comparison.csv"
)

print(
    RESULTS_DIR /
    "false_negative_analysis.csv"
)

print(
    RESULTS_DIR /
    "metrics.json"
)

print(
    RESULTS_DIR /
    "roc_curve_comparison.png"
)

print(
    MODELS_DIR /
    "logistic_without_l2.joblib"
)

print(
    MODELS_DIR /
    "logistic_l2_readmission.joblib"
)

print("\nDone.")



# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]

threshold_rows = []

for threshold in thresholds:

    for model_name, probabilities in [
        ("LOGISTIC REGRESSION WITHOUT L2", prob_without_l2),
        ("LOGISTIC REGRESSION WITH L2", prob_with_l2),
    ]:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        threshold_rows.append({
            "model": model_name,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "false_negative_rate": fn / (fn + tp)
        })

threshold_df = pd.DataFrame(threshold_rows)

threshold_df.to_csv(
    RESULTS_DIR / "threshold_analysis.csv",
    index=False
)

print(threshold_df.to_string(index=False))

print("\nThreshold analysis saved to:")
print(RESULTS_DIR / "threshold_analysis.csv")
