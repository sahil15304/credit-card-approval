#!/usr/bin/env python3
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
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

UCI_COLS = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]
PROXY_CATS = ["cat_status_like","cat_job_like","cat_education_like","cat_marital_like","cat_residence_like","cat_property_like","cat_bankacct_like","cat_prior_default_like","cat_phone_like"]
PROXY_NUMS = ["num_metric_1","num_metric_2","num_metric_3","num_metric_4","num_metric_5","num_metric_6"]

def load_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        if "approved" in df.columns or set(UCI_COLS).issubset(df.columns):
            return df
    except Exception:
        pass
    return pd.read_csv(csv_path, header=None, names=UCI_COLS, na_values="?")

def main(args):
    df = load_csv(args.csv)
    label_col = "approved" if "approved" in df.columns else "A16"
    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    y = y_raw.astype(int) if y_raw.dtype.kind in "iu" else y_raw.map({"+":1, "-":0}).astype(int)

    if "A1" in X.columns:
        categorical_cols = ["A1","A4","A5","A6","A7","A9","A10","A12","A13"]
        numeric_cols = ["A2","A3","A8","A11","A14","A15"]
    else:
        categorical_cols = PROXY_CATS
        numeric_cols = PROXY_NUMS

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_cols),
    ])

    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", log_reg)])

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 3.0, 10.0], "clf__penalty": ["l1", "l2"]}

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)

    print("Best params:", grid.best_params_)
    print("Test ROC-AUC:", round(auc, 3))
    print(classification_report(y_test, y_pred))

    try:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.savefig(ARTIFACTS_DIR / "roc_curve.png")
        plt.close()

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png")
        plt.close()
    except:
        pass

    schema = {"categorical": categorical_cols, "numeric": numeric_cols, "label": label_col}
    (ARTIFACTS_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    joblib.dump(best_model, ARTIFACTS_DIR / "model.pkl")
    print(f"Saved model to {ARTIFACTS_DIR / 'model.pkl'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    main(args)
