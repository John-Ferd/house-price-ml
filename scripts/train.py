"""Tune and compare six models with 5-fold cross-validation, then train and save the best.

Steps
1. Hold out 20% of the data as a test set and do not touch it until the end.
2. On the other 80%, tune every model that has hyperparameters with a
   randomized search (5-fold CV) and record the CV score of its best setting.
3. Refit the model with the lowest CV RMSE on the full 80% and score it once
   on the test set. One deployment constraint: the served model's compressed
   file must fit a size budget (25 MB), because it is committed to the repo and
   baked into the Docker image. A model over budget is skipped in favour of
   the next best by CV, and the skip is recorded in artifacts/metrics.json.

The test set plays no part in tuning or in choosing the model. The CV score of
a tuned model is slightly optimistic (it is the best of many settings), which
is one more reason the reported number is the single test-set score.

Writes models/model.joblib, models/metadata.json, artifacts/metrics.json and plots.
"""
import argparse
import io
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
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split

from house_price.data import CAP, CATEGORICAL, NUMERIC, TARGET, load_housing, synthetic_housing
from house_price.env import environment
from house_price.models import candidates, make_pipeline, search_space

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=str(ROOT / "data" / "housing.csv"))
ap.add_argument("--synthetic", action="store_true", help="tiny synthetic data (CI / no download)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-iter", type=int, default=30, help="random-search settings per tunable model")
ap.add_argument("--max-model-mb", type=float, default=25, help="size budget for the served model file")
args = ap.parse_args()

df = synthetic_housing() if args.synthetic else load_housing(args.data)
X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=args.seed)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


folds = KFold(5, shuffle=True, random_state=args.seed)


def _jsonable(v):
    return v.item() if hasattr(v, "item") else v


def tune(name, model):
    """CV score of the model's best setting; also returns that fitted-ready pipeline."""
    space = search_space(name)
    pipe = make_pipeline(model)
    if not space or args.synthetic:
        r = cross_validate(pipe, X_tr, y_tr, cv=folds, scoring=scoring, n_jobs=-1)
        rmse, sd, mae, r2, params = (-r["test_rmse"].mean(), r["test_rmse"].std(), -r["test_mae"].mean(),
                                     r["test_r2"].mean(), {})
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=args.n_iter, cv=folds, scoring=scoring, refit="rmse",
                                    random_state=args.seed, n_jobs=-1)
        search.fit(X_tr, y_tr)
        i, cvr = search.best_index_, search.cv_results_
        rmse, sd, mae, r2 = (-cvr["mean_test_rmse"][i], cvr["std_test_rmse"][i], -cvr["mean_test_mae"][i],
                             cvr["mean_test_r2"][i])
        params = {k.split("__")[-1]: _jsonable(v) for k, v in search.best_params_.items()}
        pipe = pipe.set_params(**search.best_params_)
    res = {"rmse": round(float(rmse)), "rmse_sd": round(float(sd)), "mae": round(float(mae)),
           "r2": round(float(r2), 4), "best_params": params}
    return res, pipe


results, pipes = {}, {}
for name, model in candidates(args.seed).items():
    results[name], pipes[name] = tune(name, model)
    r = results[name]
    print(f"{name:22s} CV RMSE ${r['rmse']:>7,} (sd {r['rmse_sd']:,})  MAE ${r['mae']:>7,}  R2 {r['r2']:.3f}  "
          f"{r['best_params'] or ''}")

# Serve the best model by CV RMSE whose compressed file fits the size budget: the model
# is committed to the repo and baked into the Docker image (GitHub rejects files > 100 MB).
ranking = sorted((k for k in results if k != "baseline_mean"), key=lambda k: results[k]["rmse"])
for best in ranking:
    final = pipes[best].fit(X_tr, y_tr)
    buf = io.BytesIO()
    joblib.dump(final, buf, compress=3)
    size_mb = buf.getbuffer().nbytes / 1e6
    results[best]["model_file_mb"] = round(size_mb, 1)
    if size_mb <= args.max_model_mb:
        break
    print(f"{best}: {size_mb:.0f} MB compressed > {args.max_model_mb} MB budget, trying the next model")
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
    {"cv": results, "test": test, "permutation_importance_rmse_increase": importance,
     "environment": environment()}, indent=2))
if args.synthetic:
    # CI / smoke run: keep the committed real-data model and plots untouched.
    print("synthetic run: metrics written to artifacts/metrics_synthetic.json; saved model left untouched")
    sys.exit(0)

joblib.dump(final, ROOT / "models" / "model.joblib", compress=3)
(ROOT / "models" / "metadata.json").write_text(json.dumps({
    "model": best, "best_params": results[best]["best_params"], "trained_on": "california_housing_1990",
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
