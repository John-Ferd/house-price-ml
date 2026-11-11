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
GEO = ["dist_sf_km", "dist_la_km", "dist_nearest_city_km", "rot45_u", "rot45_v"]
DERIVED = RATIOS + GEO

# Approximate centres of the two big job markets (San Francisco, Los Angeles).
SF = (37.7749, -122.4194)
LA = (34.0522, -118.2437)


def haversine_km(lat, lon, lat0: float, lon0: float):
    """Great-circle distance in km from each (lat, lon) to a fixed point."""
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Add per-household ratios and location features; keep the original columns.

    Ratios: raw counts measure district size, ratios describe the houses.
    Location: distance to the SF and LA job centres, and the coordinates rotated
    by 45 degrees. Trees split on one axis at a time, so a diagonal boundary
    such as the California coastline takes many splits in (lat, lon) but fewer
    when the rotated axes are also available. All of these are fixed formulas
    with nothing learned from data, so they cannot leak.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        hh = X["households"].replace(0, np.nan)
        X["rooms_per_household"] = X["total_rooms"] / hh
        X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"].replace(0, np.nan)
        X["population_per_household"] = X["population"] / hh
        X["dist_sf_km"] = haversine_km(X["latitude"], X["longitude"], *SF)
        X["dist_la_km"] = haversine_km(X["latitude"], X["longitude"], *LA)
        X["dist_nearest_city_km"] = np.minimum(X["dist_sf_km"], X["dist_la_km"])
        X["rot45_u"] = (X["latitude"] + X["longitude"]) / np.sqrt(2)
        X["rot45_v"] = (X["latitude"] - X["longitude"]) / np.sqrt(2)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features or []) + DERIVED)


def build_preprocessor() -> Pipeline:
    numeric = NUMERIC + DERIVED
    columns = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(categories=[OCEAN_CATEGORIES], handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("derived", DerivedFeatures()), ("columns", columns)])


def feature_names() -> list[str]:
    return NUMERIC + DERIVED + [f"ocean_proximity={c}" for c in OCEAN_CATEGORIES]


def as_frame(records) -> pd.DataFrame:
    """Records (list of dicts) -> DataFrame with the expected column order."""
    return pd.DataFrame(records)[NUMERIC + CATEGORICAL]
