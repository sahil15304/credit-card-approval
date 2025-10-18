#!/usr/bin/env python3
"""
Train a Credit Card Approval classifier (Logistic Regression + hyperparameter tuning).

Usage:
    python train_credit_approval.py --csv path/to/credit_card_approval.csv

Supports either:
- Original UCI headers A1..A16 with '+'/'-' labels, or
- Cleaned CSV with proxy names and numeric 'approved' label.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

UCI_COLS = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]
PROXY_CATS = ["cat_status_like","cat_job_like","cat_education_like","cat_marital_like","cat_residence_like","cat_property_like","cat_bankacct_like","cat_prior_default_like","cat_phone_like"]
PROXY_NUMS = ["num_metric_1","num_metric_2","num_metric_3","num_metric_4","num_metric_5","num_metric_6"]

def load_csv(csv_path: str) -> pd.DataFrame:
    # Try headered CSV first
    try:
        df = pd.read_csv(csv_path)
        if "approved" in df.columns or set(UCI_COLS).issubset(df.columns):
            return df
    except Exception:
        pass
    # Fallback: original UCI with no header
    df = pd.read_csv(csv_path, header=None, names=UCI_COLS, na_values="?")
    return df

def main(args):
    df = load_csv(args.csv)

    # Determine label column
    label_col = "approved" if "approved" in df.columns else "A16"
    X = df.drop(columns=[label_col])
    y_raw = df[label_col].copy()

    # Convert labels
    if y_raw.dtype.kind in "iu":
        y = y_raw.astype(int)
    else:
        y = y_raw.map({"+":1, "-":0}).astype(int)

    # Select feature groups
    if "A1" in X.columns:
        categorical_cols = ["A1","A4","A5","A6","A7","A9","A10","A12","A13"]
        numeric_cols = ["A2","A3","A8","A11","A14","A15"]
    else:
        categorical_cols = PROXY_CATS
        numeric_cols = PROXY_NUMS

    # Preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols),
    ])

    # Model
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", log_reg)])

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 3.0, 10.0], "clf__penalty": ["l1", "l2"]}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluate
    y_proba = best_model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Denied(0)","Approved(1)"])
    print("Best params:", grid.best_params_)
    print(f"Test ROC-AUC: {auc:.3f}")
    print(report)

    # Optional plots
    try:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve")
        plt.savefig("artifacts/roc_curve.png", bbox_inches="tight")
        plt.close()

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix")
        plt.savefig("artifacts/confusion_matrix.png", bbox_inches="tight")
        plt.close()
        print("Saved plots to artifacts/")
    except Exception as e:
        print("Plotting skipped:", e)

    # Save model + schema
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / "model.pkl"
    import json
    schema = {
        "categorical": categorical_cols,
        "numeric": numeric_cols,
        "label": label_col
    }
    (out_path / "schema.json").write_text(json.dumps(schema, indent=2))
    joblib.dump(best_model, model_file)
    print(f"Saved model to: {model_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV (proxy-named or UCI-style)")
    parser.add_argument("--out", default="artifacts", help="Directory to write outputs")
    args = parser.parse_args()
    main(args)
