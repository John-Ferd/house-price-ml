import numpy as np
import pandas as pd

from house_price.data import CATEGORICAL, NUMERIC, TARGET
from house_price.features import RatioFeatures, build_preprocessor, feature_names


def test_ratio_features_are_per_household(housing):
    out = RatioFeatures().transform(housing.head(3))
    expected = housing["total_rooms"].head(3) / housing["households"].head(3)
    np.testing.assert_allclose(out["rooms_per_household"], expected)


def test_preprocessor_imputes_missing_values_and_handles_unseen_category(housing):
    prep = build_preprocessor().fit(housing[NUMERIC + CATEGORICAL])
    row = housing[NUMERIC + CATEGORICAL].head(1).copy()
    row["total_bedrooms"] = np.nan
    row["ocean_proximity"] = "ISLAND"            # never appears in the synthetic training data
    Z = prep.transform(row)
    assert Z.shape == (1, len(feature_names()))
    assert np.all(np.isfinite(Z))


def test_imputation_uses_training_data_only(housing):
    train = housing[NUMERIC + CATEGORICAL].iloc[:1000]
    prep = build_preprocessor().fit(train)
    imputer = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    col = NUMERIC.index("total_bedrooms")
    assert np.isclose(imputer.statistics_[col], train["total_bedrooms"].median())


def test_synthetic_data_has_real_schema(housing):
    assert set(NUMERIC + CATEGORICAL + [TARGET]) <= set(housing.columns)
    assert isinstance(housing, pd.DataFrame)
