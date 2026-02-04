# diagnostic_model.py
# Run this file to print the model type and the feature names the model expects.
# Save next to Scoring.py and run: python diagnostic_model.py

from Scoring import _ensure_model_loaded, _model
import inspect

# ensure model and policy are loaded
try:
    _ensure_model_loaded()
except Exception as e:
    print("Error while loading model/policy:", e)
    raise

m = _model
print("\n========== MODEL DIAGNOSTIC ==========")
print("Model object type:", type(m))

# Does the model expose feature_names_in_ ?
fn = getattr(m, "feature_names_in_", None)
print("\nfeature_names_in_ (on top-level model) =")
print(fn)

# If the model is a sklearn Pipeline, inspect the final estimator too
try:
    from sklearn.pipeline import Pipeline
    if isinstance(m, Pipeline):
        print("\nPipeline DETECTED.")
        print("Pipeline steps:", m.steps)
        last = m.steps[-1][1]
        print("Final estimator type:", type(last))
        print("Final estimator feature_names_in_ =")
        print(getattr(last, "feature_names_in_", None))
except Exception as e:
    print("\nPipeline check raised exception:", e)

# If the model has an attribute 'n_features_in_', print it
nfi = getattr(m, "n_features_in_", None)
print("\nn_features_in_ =", nfi)

# If model has an internal attribute that suggests training columns, print any attribute names that include 'feature' or 'columns'
attrs = [a for a in dir(m) if "feature" in a.lower() or "column" in a.lower()]
print("\nModel attributes containing 'feature' or 'column':", attrs)

print("======================================\n")
