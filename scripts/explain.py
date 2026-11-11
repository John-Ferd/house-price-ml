"""SHAP explanation of the saved model (needs `pip install -e ".[explain]"`).

Recreates the same 80/20 split as scripts/train.py and explains the saved
pipeline on 2,000 test districts with shap's TreeExplainer, which computes
exact Shapley values for tree ensembles. Each value is how much one feature
moved one prediction away from the average prediction; the mean absolute
value over districts is a global importance.

If the saved model was trained on log(price), the SHAP values are in log
units: a value of +0.10 means that feature raised the prediction by about 10%.

Writes artifacts/shap_summary.png and artifacts/shap_importance.json.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split

from house_price.data import CATEGORICAL, NUMERIC, TARGET, load_housing
from house_price.env import environment
from house_price.features import feature_names

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=str(ROOT / "data" / "housing.csv"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()

df = load_housing(args.data)
X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
_, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=args.seed)
X_te = X_te.sample(min(args.n, len(X_te)), random_state=args.seed)

pipe = joblib.load(ROOT / "models" / "model.joblib")
model = pipe.named_steps["model"]
log_target = isinstance(model, TransformedTargetRegressor)
tree_model = model.regressor_ if log_target else model
Z = pipe.named_steps["prep"].transform(X_te)
names = feature_names()

explainer = shap.TreeExplainer(tree_model)
values = explainer.shap_values(Z)
mean_abs = np.abs(values).mean(axis=0)
order = np.argsort(-mean_abs)
unit = "log(price)" if log_target else "dollars"
importance = {names[i]: round(float(mean_abs[i]), 4 if log_target else 0) for i in order}
(ROOT / "artifacts" / "shap_importance.json").write_text(json.dumps(
    {"model": type(tree_model).__name__, "shap_units": unit, "n_explained": len(X_te),
     "mean_abs_shap": importance, "environment": environment()}, indent=2))
for k, v in list(importance.items())[:8]:
    print(f"{k:28s} {v}")

shap.summary_plot(values, Z, feature_names=names, max_display=12, show=False, plot_size=(7, 5))
plt.title(f"SHAP values ({unit}), {len(X_te):,} test districts")
plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "shap_summary.png", dpi=130)
plt.close()
