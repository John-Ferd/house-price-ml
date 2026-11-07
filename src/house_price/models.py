"""The five candidate models, each wrapped in the same preprocessing pipeline.

    baseline_mean      always predicts the average training price (the bar to beat)
    linear_regression  ordinary least squares on the engineered features
    knn                average price of the 10 most similar districts
    random_forest      average of 200 decision trees, each fit on a bootstrap sample
    gradient_boosting  trees added one at a time, each fitting the previous errors

Putting preprocessing and model in one Pipeline means cross-validation refits
the imputer and scaler inside every fold, and the saved model file contains
everything the API needs to go from raw input to a price.
"""
from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def candidates(seed: int = 0) -> dict:
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                           max_leaf_nodes=31, random_state=seed),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
