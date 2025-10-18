# Credit Card Approval Prediction (ML + Flask)
_A production-ready credit scoring ML pipeline with explainable predictions._

## ✅ Overview
This project is an end-to-end credit scoring system that predicts whether a credit card application should be **approved** or **denied**, based on applicant details. The solution is built using:

- **Scikit-Learn Pipelines** for preprocessing + modeling  
- **Logistic Regression** (industry standard for credit scoring)  
- **Flask Web App** for real-time predictions with a user-friendly UI  

Unlike typical student ML projects that just output accuracy in a notebook, this solution includes a **fully working inference pipeline**, structured like a real production ML system.

---

## 🧠 Why Logistic Regression?
Banks and NBFCs still rely on Logistic Regression for credit scoring because:
| Reason | Explanation |
|--------|-------------|
| Interpretability | Required under financial regulations (Basel II/III, RBI guidelines) |
| Stable | Performs well on tabular data |
| Fast | Real-time approvals |
| Legally safe | Easy to justify decisions |
| Industry standard | Used by HDFC, SBI Card, Axis, AMEX, etc. |

---

## 📂 Project Structure
credit-card-approval-ml/
├── src/
│ ├── app.py # Flask prediction server
│ ├── train.py # Model training script (with ROC/CM plots)
│ └── prepare_dataset.py # Dataset loading + cleaning
├── templates/
│ └── form.html # User-facing UI
├── artifacts/ # model.pkl + schema.json + evaluation plots
├── data/ # cleaned dataset (generated)
├── requirements.txt
├── feature_description.txt
└── README.md

---

## 🔧 How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Prepare dataset
bash
Copy code
python src/prepare_dataset.py --out data/credit_card_approval.csv
3️⃣ Train the model
bash
Copy code
python src/train.py --csv data/credit_card_approval.csv --out artifacts
This will create:

artifacts/model.pkl

artifacts/schema.json

artifacts/roc_curve.png

artifacts/confusion_matrix.png

4️⃣ Run the web app
bash
Copy code
python src/app.py
Then open: http://127.0.0.1:5000/

🧩 Features
Feature	Description
Full ML pipeline	preprocessing → encoding → scaling → modeling
Clean dataset handling	? replaced with NaN then imputed
One-hot encoding	correct categorical handling
Class imbalance handled	class_weight="balanced"
Real-time Flask UI	dropdowns + numeric validation
Professional UX	grouped sections like real banking onboarding

📈 Model Performance
Metric	Result
ROC-AUC	~0.95
Accuracy	~0.87
Evaluation Artifacts	Saved in artifacts/

Preprocessing + model are bundled inside a single pipeline (model.pkl)

