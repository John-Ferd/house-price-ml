PY ?= python
DATA_URL = https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

.PHONY: setup data train explain test lint serve docker

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	mkdir -p data
	curl -L -o data/housing.csv $(DATA_URL)

train:            # tune + CV 6 models, saves models/model.joblib (a few minutes on a multi-core machine)
	$(PY) scripts/train.py

explain:          # SHAP values of the saved model (pip install -e ".[explain]")
	$(PY) scripts/explain.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

serve:            # http://localhost:8000/docs
	uvicorn house_price.api:app --reload

docker:
	docker build -t house-price-api .
	docker run --rm -p 8000:8000 house-price-api
