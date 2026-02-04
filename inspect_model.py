import joblib
import traceback
import json
import os
from sklearn.pipeline import Pipeline

print("\n========== MODEL INSPECTION ==========")

model_path = "model/model.pkl"
print("Looking for model at:", os.path.abspath(model_path))

try:
    m = joblib.load(model_path)
    print("SUCCESS: Model loaded.")
    print("Model type:", type(m))

    # Check top-level feature names
    print("\nfeature_names_in_:", getattr(m, "feature_names_in_", None))
    print("n_features_in_:", getattr(m, "n_features_in_", None))

    # If model is pipeline, inspect final estimator
    if isinstance(m, Pipeline):
        print("\nPipeline detected.")
        last = m.steps[-1][1]
        print("Final estimator type:", type(last))
        print("Final estimator feature_names_in_:", getattr(last, "feature_names_in_", None))

except Exception as e:
    print("\nERROR loading model:")
    traceback.print_exc()

print("======================================\n")
