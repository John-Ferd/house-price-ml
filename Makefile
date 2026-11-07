PY ?= python
DATA_URL = https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

.PHONY: setup data train test lint serve docker

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	mkdir -p data
	curl -L -o data/housing.csv $(DATA_URL)

train:            # CV over 5 models, saves models/model.joblib (~3 minutes)
	$(PY) scripts/train.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

serve:            # http://localhost:8000/docs
	uvicorn house_price.api:app --reload

docker:
	docker build -t house-price-api .
	docker run --rm -p 8000:8000 house-price-api
