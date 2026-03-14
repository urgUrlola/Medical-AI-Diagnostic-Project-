# model2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ------------- 1. Load + filter ---------------------------------------------
PATH = "Autoimmune_Disorder_10k_with_All_Disorders.csv"

df = pd.read_csv(PATH)

digestive = [
    "Celiac disease",
    "Crohn’s disease",
    "Crohn's disease",
    "Ulcerative colitis",
    "Autoimmune hepatitis",
    "Primary biliary cholangitis",
    "Primary sclerosing cholangitis",
    "Autoimmune pancreatitis",
    "Autoimmune enteropathy"
]

df = df[df["Diagnosis"].isin(digestive)].copy()
df["Diagnosis"] = df["Diagnosis"].replace({"Crohn’s disease": "Crohn's disease"})

# ------------- 2. Define X, y ------------------------------------------------
y = df["Diagnosis"].astype(str).copy()
X = df.drop(columns=["Diagnosis"])

# Drop or keep Patient_ID depending on if it's informative (usually not)
if "Patient_ID" in X.columns:
    X = X.drop(columns=["Patient_ID"])

# ------------- 3. Separate numeric vs categorical --------------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"Numeric cols: {len(numeric_cols)}; Categorical cols: {len(categorical_cols)}")

# ------------- 4. Preprocessing pipelines ------------------------------------
# helps to work with numerical and categorical values separately
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
], remainder="drop")  # drop any other columns

# ------------- 5. Model pipeline + "hyperparameter" grid -----------------------
clf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", clf)
])

param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 10, 30],
    "clf__min_samples_split": [2, 5],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)

# ------------- 6. Train / Test split ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid.fit(X_train, y_train)

# able to show us which parameters are the most relevant/important
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# ------------- 7. Evaluate on test set --------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------- 8. Feature importances (mapping back to original names) -------
# Extract feature names after preprocessing
# For numeric: names are numeric_cols
# For categorical: we must fetch the onehot encoder feature names
preproc = best_model.named_steps["preproc"]

# transformed numeric names
trans_names = []
if len(numeric_cols) > 0:
    trans_names.extend(numeric_cols)

# categorical
if len(categorical_cols) > 0:
    # Get the fitted onehot encoder
    ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
    # Build categorical feature names
    try:
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        # fallback if older sklearn
        cat_feature_names = []
        for i, col in enumerate(categorical_cols):
            cats = ohe.categories_[i]
            cat_feature_names += [f"{col}__{str(c)}" for c in cats]
    trans_names.extend(cat_feature_names)


importances = best_model.named_steps["clf"].feature_importances_
feat_imp = pd.Series(importances, index=trans_names).sort_values(ascending=False)

print("\nTop 30 feature importances:")
print(feat_imp.head(30))

# Plot top 20
plt.figure(figsize=(10,6))
feat_imp.head(20).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# ------------- 9. Save model -------------------------------------------------
joblib.dump(best_model, "digestive_diagnosis_rf_model.joblib")
print("Model saved to digestive_diagnosis_rf_model.joblib")

