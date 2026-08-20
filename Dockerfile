ARG CIVICSTRUCT_PLATFORM=linux/amd64
FROM --platform=${CIVICSTRUCT_PLATFORM} pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN python -m pip install --no-cache-dir \
    accelerate==1.14.0 \
    bitsandbytes==0.50.0 \
    fastapi==0.138.0 \
    peft==0.20.0 \
    transformers==5.10.1 \
    uvicorn==0.49.0

COPY src ./src
COPY data/final_results/frozen_system_manifest.json ./data/final_results/frozen_system_manifest.json
COPY data/model_registry/registry_record.json ./data/model_registry/registry_record.json

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
