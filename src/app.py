from flask import Flask, render_template, request
import pandas as pd, joblib, json
from pathlib import Path

# -------------------------
# PATHS for src/ layout
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = PROJECT_ROOT / "templates"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"

# Initialize Flask app with correct templates folder
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Load model + schema
model = joblib.load(MODEL_PATH)
schema = json.loads(SCHEMA_PATH.read_text())
categorical_cols = schema["categorical"]
numeric_cols = schema["numeric"]

# -------------------------
# FRIENDLY LABELS
# -------------------------
FRIENDLY = {
    # categorical
    "cat_status_like": "Applicant Category",
    "cat_job_like": "Employment Type",
    "cat_education_like": "Education Level",
    "cat_marital_like": "Marital Category",
    "cat_residence_like": "Residence Category",
    "cat_property_like": "Property / Mortgage Type",
    "cat_bankacct_like": "Bank Relationship",
    "cat_prior_default_like": "Has Past Default",
    "cat_phone_like": "Has Verified Phone",
    # numeric
    "num_metric_1": "Age (approx)",
    "num_metric_2": "Debt Ratio (approx)",
    "num_metric_3": "Monthly Income (approx)",
    "num_metric_4": "Employment Duration (approx)",
    "num_metric_5": "Loan Amount Requested (approx)",
    "num_metric_6": "Credit History Length (approx)",
}

# Binary yes/no
BINARY_TF = [("t", "Yes"), ("f", "No")]

# Human-friendly labels (values map to anonymized codes model expects)
CATEGORICAL_OPTIONS = {
    "cat_prior_default_like": BINARY_TF,
    "cat_phone_like": BINARY_TF,
    "cat_status_like": [
        ("a", "New-to-credit"),
        ("b", "Established customer"),
        ("w", "Low documentation"),
        ("y", "Full documentation"),
        ("u", "Unknown"),
    ],
    "cat_job_like": [
        ("a", "Salaried"),
        ("b", "Self-employed"),
        ("c", "Government"),
        ("d", "Student"),
        ("e", "Retired"),
        ("w", "Other / Contract"),
    ],
    "cat_education_like": [
        ("a", "High school"),
        ("b", "Undergraduate"),
        ("c", "Graduate"),
        ("d", "Postgraduate"),
        ("e", "Doctorate"),
        ("u", "Unknown"),
    ],
    "cat_marital_like": [
        ("a", "Single"),
        ("b", "Married"),
        ("c", "Divorced"),
        ("d", "Widowed"),
        ("u", "Prefer not to say"),
    ],
    "cat_residence_like": [
        ("a", "Owned home"),
        ("b", "Rented home"),
        ("c", "Company-provided"),
        ("d", "Family/Shared"),
        ("u", "Other/Unknown"),
    ],
    "cat_property_like": [
        ("a", "No property"),
        ("b", "Home mortgage"),
        ("c", "Other real estate"),
        ("d", "Vehicle asset"),
        ("u", "Other/Unknown"),
    ],
    "cat_bankacct_like": [
        ("a", "No bank account"),
        ("b", "Savings account"),
        ("c", "Current/Checking"),
        ("d", "Savings + Checking"),
        ("u", "Unknown"),
    ],
}

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "form.html",
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        FRIENDLY=FRIENDLY,
        CATEGORICAL_OPTIONS=CATEGORICAL_OPTIONS
    )

@app.route("/predict", methods=["POST"])
def predict():
    cat_values = [request.form.get(col, "") for col in categorical_cols]
    num_values = []
    for col in numeric_cols:
        val = request.form.get(col, "")
        try:
            num_values.append(float(val))
        except:
            num_values.append(None)

    row = dict(zip(categorical_cols + numeric_cols, cat_values + num_values))
    X = pd.DataFrame([row])

    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    label = "Approved ✅" if pred == 1 else "Denied ❌"

    return render_template(
        "form.html",
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        FRIENDLY=FRIENDLY,
        CATEGORICAL_OPTIONS=CATEGORICAL_OPTIONS,
        result=label,
        probability=f"{proba:.2%}"
    )

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
