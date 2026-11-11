import numpy as np
from sklearn.model_selection import train_test_split

from house_price.data import CATEGORICAL, NUMERIC, TARGET
from house_price.models import candidates, make_pipeline, search_space


def test_every_model_beats_the_mean_baseline(housing):
    X, y = housing[NUMERIC + CATEGORICAL], housing[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    rmse = {}
    for name, est in candidates().items():
        pred = make_pipeline(est).fit(X_tr, y_tr).predict(X_te)
        rmse[name] = np.sqrt(np.mean((pred - y_te) ** 2))
    for name, value in rmse.items():
        if name != "baseline_mean":
            assert value < rmse["baseline_mean"], name


def test_pipeline_goes_from_raw_rows_to_prices(housing):
    X, y = housing[NUMERIC + CATEGORICAL], housing[TARGET]
    model = make_pipeline(candidates()["gradient_boosting"]).fit(X, y)
    raw = X.head(5).copy()
    raw.loc[raw.index[0], "total_bedrooms"] = np.nan      # missing value is imputed inside the pipeline
    pred = model.predict(raw)
    assert pred.shape == (5,) and np.all(np.isfinite(pred))


def test_search_spaces_name_real_pipeline_parameters():
    for name, est in candidates().items():
        pipe = make_pipeline(est)
        valid = pipe.get_params().keys()
        for key in search_space(name):
            assert key in valid, (name, key)


def test_log_target_model_predicts_in_dollars(housing):
    X, y = housing[NUMERIC + CATEGORICAL], housing[TARGET]
    pred = make_pipeline(candidates()["gradient_boosting_log"]).fit(X, y).predict(X.head(20))
    assert pred.min() > 1_000          # exponentiated back to dollars, not log units
