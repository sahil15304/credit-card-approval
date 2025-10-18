# Credit Card Approval Mini-Project

## 1) Prepare dataset (human‑readable proxy names)
```bash
pip install -r requirements.txt
python prepare_dataset.py --out credit_card_approval.csv
```
If download fails (firewall), put `crx.data` in this folder and rerun.

## 2) Train model (Logistic Regression + tuning)
```bash
python train_credit_approval.py --csv credit_card_approval.csv --out artifacts
```
Outputs:
- `artifacts/model.pkl`
- `artifacts/schema.json`
- `artifacts/roc_curve.png`
- `artifacts/confusion_matrix.png`

## 3) Run the web app
```bash
python app.py
# open http://127.0.0.1:5000/
```

## Notes
- CSV can be either proxy headers (above) or original UCI headers; training script detects both.
- Missing values handled; categoricals one‑hot encoded; numerics scaled; class imbalance balanced.
- Threshold = 0.5 by default (adjust in app if needed).
