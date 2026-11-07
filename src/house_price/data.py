"""
Loading the 1990 California census housing data (20,640 block groups).

This is the version with raw counts and the categorical ``ocean_proximity``
column (from Géron's *Hands-On Machine Learning* repository), not the
pre-normalised copy bundled in scikit-learn. Target: ``median_house_value``
in US dollars.

Known quirk: values were top-coded at $500,001 in the census release, so
~4.8% of districts sit exactly at the cap. The true value is "at least
$500,001", which no regression model can recover. We keep these rows (dropping
them would make test metrics look better than reality) and report errors
on the uncapped subset separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
TARGET = "median_house_value"
CAP = 500_001.0
NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
           "population", "households", "median_income"]
CATEGORICAL = ["ocean_proximity"]
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def load_housing(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(NUMERIC + CATEGORICAL + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def synthetic_housing(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the real schema, for tests and CI only."""
    rng = np.random.default_rng(seed)
    households = rng.integers(50, 2000, n).astype(float)
    income = rng.gamma(4.0, 1.0, n)
    df = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, n),
        "latitude": rng.uniform(32, 42, n),
        "housing_median_age": rng.integers(1, 52, n).astype(float),
        "total_rooms": households * rng.uniform(3, 8, n),
        "total_bedrooms": households * rng.uniform(0.9, 1.3, n),
        "population": households * rng.uniform(1.5, 4.5, n),
        "households": households,
        "median_income": income,
        "ocean_proximity": rng.choice(OCEAN_CATEGORIES[:2] + OCEAN_CATEGORIES[3:], n),
    })
    df.loc[rng.random(n) < 0.01, "total_bedrooms"] = np.nan
    inland = (df["ocean_proximity"] == "INLAND").to_numpy()
    value = 40_000 * income - 60_000 * inland + rng.normal(0, 30_000, n) + 50_000
    df[TARGET] = np.clip(value, 15_000, CAP)
    return df
