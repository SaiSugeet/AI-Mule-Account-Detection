# diagnose_model_sample.py
import pandas as pd, numpy as np, json
from pprint import pprint
import joblib
from pathlib import Path

MODEL_DIR = Path("model")
FI = MODEL_DIR / "feature_importances.csv"
FEATS_CSV = Path("data/features_accounts.csv")

print("Loading model & artifacts...")
m = joblib.load(MODEL_DIR / "model.pkl")
fi = pd.read_csv(FI) if FI.exists() else None
print("\nTop feature importances:")
if fi is not None:
    print(fi.sort_values("importance", ascending=False).head(10).to_string(index=False))
else:
    print("feature_importances.csv not found")

# load features table to compute percentiles
if FEATS_CSV.exists():
    df = pd.read_csv(FEATS_CSV)
    sample_id = "A000002"
    sample_row = df[df["account_id"]==sample_id]
    if sample_row.empty:
        print(f"\nSample {sample_id} not found in features CSV — using Scoring computed values instead.")
    else:
        sample_row = sample_row.iloc[0]
        print(f"\nSample {sample_id} feature values (from CSV):")
        print(sample_row.to_string())
        # For top features, show percentile
        if fi is not None:
            topf = fi.sort_values("importance", ascending=False).head(10)["feature"].tolist()
            print("\nPercentiles for top features (sample vs train):")
            for f in topf:
                val = sample_row[f]
                pct = (df[f] <= val).mean()
                print(f"{f}: value={val}  percentile={pct:.3f}")
else:
    print("\nfeatures_accounts.csv not found — cannot compute percentiles.")
