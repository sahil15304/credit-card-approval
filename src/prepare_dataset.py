#!/usr/bin/env python3
"""
Prepare the UCI Credit Approval (CRX) dataset into a clean CSV with proxy names.

Usage:
    python src/prepare_dataset.py --out data/credit_card_approval.csv
"""

import argparse
import io
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

ORIG_COLS = [
    "A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"
]

PROXY_COLS = [
    "cat_status_like",
    "num_metric_1",
    "num_metric_2",
    "cat_job_like",
    "cat_education_like",
    "cat_marital_like",
    "cat_residence_like",
    "num_metric_3",
    "cat_property_like",
    "cat_bankacct_like",
    "num_metric_4",
    "cat_prior_default_like",
    "cat_phone_like",
    "num_metric_5",
    "num_metric_6",
    "approved",
]

def main(out_path: Path):
    # Try to download; if blocked, allow local fallback
    data_bytes = None
    try:
        import requests
        r = requests.get(UCI_URL, timeout=20)
        r.raise_for_status()
        data_bytes = r.content
        print("Downloaded dataset from UCI.")
    except Exception as e:
        print("Download failed or blocked:", e)
        local_path = PROJECT_ROOT / "crx.data"
        if not local_path.exists():
            raise SystemExit(
                "No internet and no local 'crx.data' found.\n"
                f"Place 'crx.data' at: {local_path} and rerun."
            )
        data_bytes = local_path.read_bytes()
        print("Loaded local crx.data")

    # Read and clean
    df = pd.read_csv(io.BytesIO(data_bytes), header=None, names=ORIG_COLS, na_values="?")
    df.columns = PROXY_COLS
    df["approved"] = df["approved"].map({"+": 1, "-": 0}).astype("Int64")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned CSV to: {out_path.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "credit_card_approval.csv"),
                    help="Output CSV path (default: data/credit_card_approval.csv)")
    args = ap.parse_args()
    main(Path(args.out))
