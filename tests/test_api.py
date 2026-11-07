import pytest
from fastapi.testclient import TestClient

import house_price.api as api

DISTRICT = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880,
    "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


@pytest.fixture
def client(model_file, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", model_file)
    with TestClient(api.app) as c:
        yield c


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "random_forest"


def test_predict_returns_one_price_per_instance(client):
    missing_bedrooms = {**DISTRICT, "total_bedrooms": None}
    r = client.post("/predict", json={"instances": [DISTRICT, missing_bedrooms]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and all(p > 0 for p in preds)


def test_invalid_input_is_rejected(client):
    bad = {**DISTRICT, "ocean_proximity": "MOON"}
    assert client.post("/predict", json={"instances": [bad]}).status_code == 422
    no_field = {k: v for k, v in DISTRICT.items() if k != "median_income"}
    assert client.post("/predict", json={"instances": [no_field]}).status_code == 422


def test_missing_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nope.joblib")
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "model not loaded"
        assert c.post("/predict", json={"instances": [DISTRICT]}).status_code == 503
