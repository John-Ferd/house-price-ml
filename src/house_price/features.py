"""
Feature engineering and preprocessing, as one scikit-learn pipeline.

Raw counts per block group (total_rooms, population, ...) mostly measure how
*big* the district is, not what its houses are like, so they are turned into
per-household ratios. All steps live inside the pipeline, so imputation
medians and scaling statistics are learned on the training fold only (no
leakage into CV or test), and the exact same transform runs in the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, OCEAN_CATEGORIES

RATIOS = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Add per-household ratio features; keep the original columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        hh = X["households"].replace(0, np.nan)
        X["rooms_per_household"] = X["total_rooms"] / hh
        X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"].replace(0, np.nan)
        X["population_per_household"] = X["population"] / hh
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features or []) + RATIOS)


def build_preprocessor() -> Pipeline:
    numeric = NUMERIC + RATIOS
    columns = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(categories=[OCEAN_CATEGORIES], handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("ratios", RatioFeatures()), ("columns", columns)])


def feature_names() -> list[str]:
    return NUMERIC + RATIOS + [f"ocean_proximity={c}" for c in OCEAN_CATEGORIES]


def as_frame(records) -> pd.DataFrame:
    """Records (list of dicts) -> DataFrame with the expected column order."""
    return pd.DataFrame(records)[NUMERIC + CATEGORICAL]
