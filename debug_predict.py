# debug_predict.py
# Run this to show the X_row (aligned), model classes, and full predict_proba output.

import json
from Scoring import _ensure_model_loaded, _load_transactions, _compute_account_features
from pprint import pprint
import pandas as pd

# ensure model loaded
_ensure_model_loaded()
from Scoring import _model  # after loader this should be set

recv = "A000002"
tx = _load_transactions()
feat = _compute_account_features(tx, recv, as_of_time=None)

print("\n--- computed features for receiver:", recv, "---")
pprint(feat)

# Build X_row aligned to model.feature_names_in_ if available, else fallback to CSV header
expected = getattr(_model, "feature_names_in_", None)
if expected is None:
    print("\nModel does NOT expose feature_names_in_. Falling back to features CSV header.")
    try:
        header = list(pd.read_csv("data/features_accounts.csv", nrows=0).columns)
        expected = [c for c in header if c not in ("account_id","is_mule")]
        print("Using header from CSV with columns:", expected)
    except Exception as e:
        print("Could not read CSV for header:", e)
        expected = list(feat.keys())
        expected.remove("account_id")
        print("Fallback expected columns:", expected)
else:
    print("\nModel expects columns (feature_names_in_):")
    print(expected)

# build X_row
X_row = pd.DataFrame([{c: feat.get(c, 0) for c in expected}])
print("\n--- X_row being passed to the model (columns & values) ---")
print(X_row.to_string(index=False))

# show model classes and predict_proba
print("\n--- model metadata & outputs ---")
try:
    classes = getattr(_model, "classes_", None)
    print("model.classes_:", classes)
    # get raw predict_proba
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X_row)
        print("predict_proba shape:", proba.shape)
        print("predict_proba array:", proba.tolist())
    else:
        pred = _model.predict(X_row)
        print("model.predict output:", pred)
except Exception as e:
    print("Error calling predict/predict_proba:", repr(e))

print("\n--- End debug ---\n")
