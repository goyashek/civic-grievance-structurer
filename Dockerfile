FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi==0.138.0 uvicorn==0.49.0

COPY src ./src
COPY data/final_results/frozen_system_manifest.json ./data/final_results/frozen_system_manifest.json
COPY data/model_registry/registry_record.json ./data/model_registry/registry_record.json

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
