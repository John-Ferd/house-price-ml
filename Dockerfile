# Serve the trained model with uvicorn.
# Build after `make train` so models/model.joblib exists:
#   docker build -t house-price-api .
#   docker run -p 8000:8000 house-price-api
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "house_price.api:app", "--host", "0.0.0.0", "--port", "8000"]
