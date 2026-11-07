import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor

from house_price.data import CATEGORICAL, NUMERIC, TARGET, synthetic_housing
from house_price.models import make_pipeline


@pytest.fixture(scope="session")
def housing():
    return synthetic_housing(n=1500, seed=0)


@pytest.fixture(scope="session")
def model_file(tmp_path_factory, housing):
    """A small model trained on synthetic data, saved like the real one."""
    model = make_pipeline(RandomForestRegressor(n_estimators=30, random_state=0))
    model.fit(housing[NUMERIC + CATEGORICAL], housing[TARGET])
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(model, path)
    (path.parent / "metadata.json").write_text('{"model": "random_forest", "test_rmse": 1, "test_r2": 0.9}')
    return path
