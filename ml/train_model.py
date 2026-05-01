"""
MediSense AI v2 — Random Forest Training Pipeline
===================================================
Improvements over v1:
  - Proper 80/20 stratified train/test split
  - 5-fold cross-validation with mean ± std reporting
  - Full metrics: accuracy, precision, recall, F1 (macro + weighted)
  - Confusion matrix saved as JSON
  - Overfitting check: compare train vs test accuracy
  - Tuned hyperparameters to reduce overfitting
  - Full sklearn Pipeline (scaler + model) saved with joblib
  - Feature importance report
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MDL_DIR  = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

print("=" * 60)
print("  MediSense AI v2 — ML Training Pipeline")
print("=" * 60)

# ── 1. Load raw data ────────────────────────────────────────────────────────────
print("\n[1/7] Loading datasets...")
dataset      = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
descriptions = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
precautions  = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
severity_df  = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

print(f"  Raw dataset: {dataset.shape[0]} rows × {dataset.shape[1]} cols")
print(f"  Unique diseases: {dataset['Disease'].nunique()}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
print("\n[2/7] Preprocessing...")

def clean(s):
    """Normalise symptom string: strip, lowercase, underscores."""
    if pd.isna(s) or str(s).strip() == "":
        return ""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

# Normalise severity map
severity_df["Symptom"] = severity_df["Symptom"].apply(clean)
SEVERITY_MAP = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Collect all unique symptoms (sorted for stable column order)
symptom_cols = [c for c in dataset.columns if c.startswith("Symptom_")]
all_symptoms = sorted({
    clean(v)
    for col in symptom_cols
    for v in dataset[col].dropna()
    if clean(v)
})
print(f"  Unique symptoms found: {len(all_symptoms)}")

# Handle missing values: NaN symptom slots → empty string → weight 0
def encode_row(row):
    """Convert a dataset row into a severity-weighted feature vector."""
    vec = {s: 0 for s in all_symptoms}
    for col in symptom_cols:
        s = clean(row.get(col, ""))
        if s and s in vec:
            vec[s] = SEVERITY_MAP.get(s, 1)   # default weight=1 for unknowns
    return vec

print("  Building feature matrix (this takes ~10 seconds)...")
X_rows = [encode_row(row) for _, row in dataset.iterrows()]
X = pd.DataFrame(X_rows, columns=all_symptoms)
y_raw = dataset["Disease"].str.strip()

# Encode target labels
le = LabelEncoder()
y  = le.fit_transform(y_raw)
print(f"  Feature matrix: {X.shape}  |  Classes: {len(le.classes_)}")

# ── 3. Train / test split ─────────────────────────────────────────────────────
print("\n[3/7] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

# ── 4. Build pipeline ─────────────────────────────────────────────────────────
print("\n[4/7] Building sklearn Pipeline...")
# StandardScaler on severity weights; RF with tuned params to reduce overfitting
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=150,      # reduced from 200 — less memorisation
        max_depth=20,          # cap depth to prevent pure overfitting
        min_samples_split=4,   # require ≥4 samples to split a node
        min_samples_leaf=2,    # require ≥2 samples at each leaf
        max_features="sqrt",   # random feature subset at each split
        class_weight="balanced",  # handles any class imbalance
        random_state=42,
        n_jobs=-1,
    )),
])

# ── 5. Cross-validation ───────────────────────────────────────────────────────
print("\n[5/7] 5-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X_train, y_train, cv=cv,
    scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
    return_train_score=True,
)

cv_acc_test  = cv_results["test_accuracy"]
cv_acc_train = cv_results["train_accuracy"]
cv_f1        = cv_results["test_f1_weighted"]

print(f"  CV Train Accuracy : {cv_acc_train.mean():.4f} ± {cv_acc_train.std():.4f}")
print(f"  CV Val Accuracy   : {cv_acc_test.mean():.4f}  ± {cv_acc_test.std():.4f}")
print(f"  CV F1 (weighted)  : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

overfit_gap = cv_acc_train.mean() - cv_acc_test.mean()
if overfit_gap > 0.05:
    print(f"  ⚠️  Overfitting gap: {overfit_gap:.4f} — consider further regularisation")
else:
    print(f"  ✅ Overfitting gap: {overfit_gap:.4f} — acceptable")

# ── 6. Final training & evaluation ───────────────────────────────────────────
print("\n[6/7] Training final model on full training set...")
pipeline.fit(X_train, y_train)

y_pred_train = pipeline.predict(X_train)
y_pred_test  = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)
precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
recall    = recall_score(y_test,    y_pred_test, average="weighted", zero_division=0)
f1        = f1_score(y_test,        y_pred_test, average="weighted", zero_division=0)
f1_macro  = f1_score(y_test,        y_pred_test, average="macro",    zero_division=0)

print(f"\n  ── Final Evaluation (held-out test set) ──")
print(f"  Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"  Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"  Precision (W)  : {precision:.4f}")
print(f"  Recall    (W)  : {recall:.4f}")
print(f"  F1        (W)  : {f1:.4f}")
print(f"  F1        (M)  : {f1_macro:.4f}")

# Per-class classification report
report_str  = classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0)
report_dict = classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0, output_dict=True)
print(f"\n  Classification Report (first 10 classes):\n")
for line in report_str.split("\n")[:15]:
    print(" ", line)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Feature importance (top 20)
rf_model = pipeline.named_steps["rf"]
importance = pd.Series(rf_model.feature_importances_, index=all_symptoms)
top_features = importance.nlargest(20).to_dict()
print(f"\n  Top 5 most important symptoms:")
for sym, imp in list(top_features.items())[:5]:
    print(f"    {sym:<40} {imp:.4f}")

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
print("\n[7/7] Saving artefacts...")

# Save full pipeline (scaler + rf)
joblib.dump(pipeline,      os.path.join(MDL_DIR, "pipeline.pkl"))
joblib.dump(le,            os.path.join(MDL_DIR, "label_encoder.pkl"))
joblib.dump(all_symptoms,  os.path.join(MDL_DIR, "symptoms_list.pkl"))

# Build lookup maps
desc_map = {
    r["Disease"].strip(): str(r["Description"]).strip()
    for _, r in descriptions.iterrows()
}

prec_map = {}
for _, r in precautions.iterrows():
    ps = [
        str(r[f"Precaution_{i}"]).strip()
        for i in range(1, 5)
        if pd.notna(r.get(f"Precaution_{i}")) and str(r.get(f"Precaution_{i}", "")).strip()
    ]
    prec_map[r["Disease"].strip()] = ps

SPEC_MAP = {
    "Heart attack":                            "Cardiologist",
    "Hypertension ":                           "Cardiologist",
    "Diabetes ":                               "Endocrinologist",
    "Hypothyroidism":                          "Endocrinologist",
    "Hyperthyroidism":                         "Endocrinologist",
    "Hypoglycemia":                            "Endocrinologist",
    "Tuberculosis":                            "Pulmonologist",
    "Pneumonia":                               "Pulmonologist",
    "Bronchial Asthma":                        "Pulmonologist",
    "Common Cold":                             "General Physician",
    "Allergy":                                 "Allergist",
    "Migraine":                                "Neurologist",
    "(vertigo) Paroymsal  Positional Vertigo": "Neurologist",
    "Paralysis (brain hemorrhage)":            "Neurologist",
    "Cervical spondylosis":                    "Orthopedist",
    "Arthritis":                               "Rheumatologist",
    "Dengue":                                  "Infectious Disease Specialist",
    "Malaria":                                 "Infectious Disease Specialist",
    "Typhoid":                                 "Infectious Disease Specialist",
    "AIDS":                                    "Infectious Disease Specialist",
    "Hepatitis A":                             "Gastroenterologist",
    "Hepatitis B":                             "Gastroenterologist",
    "Hepatitis C":                             "Gastroenterologist",
    "Hepatitis D":                             "Gastroenterologist",
    "Hepatitis E":                             "Gastroenterologist",
    "Chronic cholestasis":                     "Gastroenterologist",
    "GERD":                                    "Gastroenterologist",
    "Gastroenteritis":                         "Gastroenterologist",
    "Peptic ulcer diseae":                     "Gastroenterologist",
    "Jaundice":                                "Gastroenterologist",
    "Alcoholic hepatitis":                     "Gastroenterologist",
    "Dimorphic hemmorhoids(piles)":            "Proctologist",
    "Urinary tract infection":                 "Urologist",
    "Drug Reaction":                           "Dermatologist",
    "Acne":                                    "Dermatologist",
    "Fungal infection":                        "Dermatologist",
    "Psoriasis":                               "Dermatologist",
    "Impetigo":                                "Dermatologist",
    "Varicose veins":                          "Vascular Surgeon",
    "Chicken pox":                             "General Physician",
}

metadata = {
    "version":       "2.0",
    "diseases":      list(le.classes_),
    "total_symptoms": len(all_symptoms),
    "descriptions":  desc_map,
    "precautions":   prec_map,
    "specializations": SPEC_MAP,
    "metrics": {
        "train_accuracy":   round(train_acc, 4),
        "test_accuracy":    round(test_acc,  4),
        "precision_weighted": round(precision, 4),
        "recall_weighted":  round(recall,   4),
        "f1_weighted":      round(f1,       4),
        "f1_macro":         round(f1_macro, 4),
        "cv_val_accuracy_mean": round(float(cv_acc_test.mean()), 4),
        "cv_val_accuracy_std":  round(float(cv_acc_test.std()),  4),
        "cv_f1_weighted_mean":  round(float(cv_f1.mean()), 4),
        "overfit_gap":      round(float(overfit_gap), 4),
    },
    "feature_importance_top20": {k: round(v, 6) for k, v in top_features.items()},
    "per_class_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "class_names": list(le.classes_),
    "severity_map": SEVERITY_MAP,
}

with open(os.path.join(MDL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  pipeline.pkl       → {os.path.join(MDL_DIR, 'pipeline.pkl')}")
print(f"  label_encoder.pkl  → {os.path.join(MDL_DIR, 'label_encoder.pkl')}")
print(f"  symptoms_list.pkl  → {os.path.join(MDL_DIR, 'symptoms_list.pkl')}")
print(f"  metadata.json      → {os.path.join(MDL_DIR, 'metadata.json')}")

print("\n" + "=" * 60)
print(f"  ✅  Training complete!")
print(f"  Test Accuracy  : {test_acc*100:.1f}%")
print(f"  F1 (weighted)  : {f1*100:.1f}%")
print(f"  Overfit gap    : {overfit_gap:.4f}")
print("=" * 60)
