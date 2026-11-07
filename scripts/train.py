"""Compare five models with 5-fold cross-validation, then train and save the best.

Steps
1. Hold out 20% of the data as a test set and do not touch it until the end.
2. On the other 80%, score every model with 5-fold cross-validation.
3. Refit the model with the lowest CV RMSE on the full 80% and score it once
   on the test set.

Writes models/model.joblib, models/metadata.json, artifacts/metrics.json and plots.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from house_price.data import CAP, CATEGORICAL, NUMERIC, TARGET, load_housing, synthetic_housing
from house_price.models import candidates, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=str(ROOT / "data" / "housing.csv"))
ap.add_argument("--synthetic", action="store_true", help="tiny synthetic data (CI / no download)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

df = synthetic_housing() if args.synthetic else load_housing(args.data)
X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=args.seed)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def cv(model):
    folds = KFold(5, shuffle=True, random_state=args.seed)
    r = cross_validate(make_pipeline(model), X_tr, y_tr, cv=folds, scoring=scoring)
    return {"rmse": round(float(-r["test_rmse"].mean())), "rmse_sd": round(float(r["test_rmse"].std())),
            "mae": round(float(-r["test_mae"].mean())), "r2": round(float(r["test_r2"].mean()), 4)}


results = {}
for name, model in candidates(args.seed).items():
    results[name] = cv(model)
    r = results[name]
    print(f"{name:18s} CV RMSE ${r['rmse']:>7,} (sd {r['rmse_sd']:,})  MAE ${r['mae']:>7,}  R2 {r['r2']:.3f}")

best = min((k for k in results if k != "baseline_mean"), key=lambda k: results[k]["rmse"])
final = make_pipeline(candidates(args.seed)[best]).fit(X_tr, y_tr)
pred = final.predict(X_te)
uncapped = y_te < CAP
test = {
    "model": best,
    "n_train": len(y_tr), "n_test": len(y_te),
    "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred)))),
    "mae": round(float(mean_absolute_error(y_te, pred))),
    "r2": round(float(r2_score(y_te, pred)), 4),
    "median_abs_pct_error": round(float(np.median(np.abs(pred - y_te) / y_te)), 4),
    "rmse_uncapped_rows": round(float(np.sqrt(mean_squared_error(y_te[uncapped], pred[uncapped])))),
    "frac_test_at_cap": round(float((~uncapped).mean()), 4),
}
test["rmse_reduction_vs_mean_baseline_cv"] = round(1 - results[best]["rmse"] / results["baseline_mean"]["rmse"], 4)
print("\nTest set:", json.dumps(test, indent=2))

imp = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=args.seed,
                             scoring="neg_root_mean_squared_error")
importance = sorted(zip(X_te.columns, imp.importances_mean), key=lambda t: -t[1])
importance = {k: round(float(v)) for k, v in importance}

(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
tag = "_synthetic" if args.synthetic else ""
(ROOT / "artifacts" / f"metrics{tag}.json").write_text(json.dumps(
    {"cv": results, "test": test, "permutation_importance_rmse_increase": importance}, indent=2))
if args.synthetic:
    # CI / smoke run: keep the committed real-data model and plots untouched.
    print("synthetic run: metrics written to artifacts/metrics_synthetic.json; saved model left untouched")
    sys.exit(0)

joblib.dump(final, ROOT / "models" / "model.joblib")
(ROOT / "models" / "metadata.json").write_text(json.dumps({
    "model": best, "trained_on": "california_housing_1990",
    "test_rmse": test["rmse"], "test_r2": test["r2"], "features": NUMERIC + CATEGORICAL,
    "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
}, indent=2))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_te, pred, s=3, alpha=0.3)
ax.plot([0, CAP], [0, CAP], "k--", lw=1)
ax.set_xlabel("actual median house value ($)")
ax.set_ylabel("predicted ($)")
ax.set_title(f"{best}: test set (R2 {test['r2']:.3f})")
fig.tight_layout()
fig.savefig(ROOT / "artifacts" / "pred_vs_actual.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
names, vals = zip(*list(importance.items()))
ax.barh(names[::-1], vals[::-1])
ax.set_xlabel("test RMSE increase when shuffled ($)")
ax.set_title("Permutation importance")
fig.tight_layout()
fig.savefig(ROOT / "artifacts" / "importance.png", dpi=130)
plt.close(fig)
