"""Candidate models, each wrapped in the same preprocessing pipeline, plus their
hyperparameter search spaces.

    baseline_mean          always predicts the average training price (the bar to beat)
    linear_regression      ordinary least squares on the engineered features
    knn                    average price of the k most similar districts
    random_forest          average of many decision trees, each fit on a bootstrap sample
    gradient_boosting      trees added one at a time, each fitting the previous errors
    gradient_boosting_log  the same, but trained on log(price) and exponentiated back.
                           Prices are right-skewed, so errors on cheap and expensive
                           districts are put on a comparable (relative) scale.

Putting preprocessing and model in one Pipeline means cross-validation refits
the imputer and scaler inside every fold, and the saved model file contains
everything the API needs to go from raw input to a price.

``search_space`` gives the hyperparameter distributions that scripts/train.py
samples with RandomizedSearchCV. The search only ever sees the training 80%.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def _hgb(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)


def candidates(seed: int = 0) -> dict:
    """Default (untuned) configuration of every candidate."""
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gradient_boosting": _hgb(seed),
        "gradient_boosting_log": TransformedTargetRegressor(regressor=_hgb(seed), func=np.log, inverse_func=np.exp),
    }


_HGB_SPACE = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_iter": randint(300, 1500),
    "max_leaf_nodes": randint(15, 128),
    "min_samples_leaf": randint(5, 60),
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": uniform(0.5, 0.5),        # fraction of features per split, in [0.5, 1.0]
}


def search_space(name: str) -> dict:
    """Parameter distributions for RandomizedSearchCV, keyed for the pipeline.
    An empty dict means the model has nothing worth tuning."""
    if name == "knn":
        return {"model__n_neighbors": randint(3, 40), "model__weights": ["uniform", "distance"]}
    if name == "random_forest":
        return {"model__max_features": uniform(0.2, 0.8), "model__min_samples_leaf": randint(1, 8),
                "model__n_estimators": [300]}
    if name == "gradient_boosting":
        return {f"model__{k}": v for k, v in _HGB_SPACE.items()}
    if name == "gradient_boosting_log":
        return {f"model__regressor__{k}": v for k, v in _HGB_SPACE.items()}
    return {}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
