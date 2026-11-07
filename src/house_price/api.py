"""
FastAPI service that serves the trained model.

    GET  /health    -> is the model loaded, which model, its test metrics
    POST /predict   -> {"instances": [ {district features}, ... ]}
                       returns {"predictions": [dollars, ...]}

Input validation is done by pydantic: a missing field, a wrong type, or an
unknown ocean_proximity value is rejected with HTTP 422 before it reaches the
model. The model file is read once at startup (path from the MODEL_PATH
environment variable, default models/model.joblib).
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import as_frame

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
STATE: dict = {}


class District(BaseModel):
    longitude: float = Field(ge=-125, le=-114)
    latitude: float = Field(ge=32, le=42.5)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(gt=0)
    total_bedrooms: float | None = Field(default=None, ge=0)   # may be missing; the pipeline imputes it
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0, description="in tens of thousands of dollars, as in the census data")
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class PredictRequest(BaseModel):
    instances: list[District] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    predictions: list[float]
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        STATE["model"] = joblib.load(MODEL_PATH)
        meta = MODEL_PATH.with_name("metadata.json")
        STATE["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    yield
    STATE.clear()


app = FastAPI(title="California house price API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    meta = STATE.get("meta", {})
    return {"status": "ok" if "model" in STATE else "model not loaded",
            "model": meta.get("model"), "test_rmse": meta.get("test_rmse"), "test_r2": meta.get("test_r2")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail=f"model file not found at {MODEL_PATH}; run `make train`")
    X = as_frame([d.model_dump() for d in req.instances])
    preds = STATE["model"].predict(X)
    return PredictResponse(predictions=[round(float(p), 2) for p in preds],
                           model=STATE.get("meta", {}).get("model", "unknown"))
