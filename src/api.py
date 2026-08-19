"""FastAPI service backed by the shared frozen inference module."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .inference import CivicStructInference, InferenceResult, MAX_COMPLAINT_CHARS


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RECORD = json.loads(
    (ROOT / "data/model_registry/registry_record.json").read_text(encoding="utf-8")
)
MODEL_INFO = {
    "model_version": f"{REGISTRY_RECORD['registered_model']}:{REGISTRY_RECORD['version']}",
    "model_alias": REGISTRY_RECORD["alias"],
    "mlflow_run_id": REGISTRY_RECORD["registration_run_id"],
    "frozen_training_run_id": REGISTRY_RECORD["frozen_training_run_id"],
    "schema_version": REGISTRY_RECORD["schema_version"],
    "evaluator_version": REGISTRY_RECORD["evaluator_version"],
    "dataset_version": REGISTRY_RECORD["dataset_version"],
    "base_model": REGISTRY_RECORD["base_model"],
    "base_model_revision": REGISTRY_RECORD["base_model_revision"],
    "adapter_sha256": REGISTRY_RECORD["adapter_sha256"],
}

app = FastAPI(title="CivicStruct", version=MODEL_INFO["model_version"])
_inference: CivicStructInference | None = None
_inference_lock = Lock()


class StructureRequest(BaseModel):
    complaint: str = Field(min_length=1, max_length=MAX_COMPLAINT_CHARS)

    @field_validator("complaint")
    @classmethod
    def reject_blank_complaints(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("complaint must contain non-whitespace text")
        return value


class StructureResponse(BaseModel):
    ok: bool
    prediction: dict[str, Any] | None
    raw_response: str | None
    error: dict[str, str] | None
    latency_seconds: float


def get_inference() -> CivicStructInference:
    global _inference
    if _inference is None:
        with _inference_lock:
            if _inference is None:
                _inference = CivicStructInference()
    return _inference


def _failure_response(result: InferenceResult) -> JSONResponse:
    error_type = (result.error or {}).get("type")
    status_code = 422 if error_type in {"invalid_input", "invalid_json", "schema_invalid"} else 503
    return JSONResponse(status_code=status_code, content=result.as_dict())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return MODEL_INFO


@app.post("/structure", response_model=StructureResponse)
def structure(request: StructureRequest) -> StructureResponse | JSONResponse:
    try:
        result = get_inference().infer(request.complaint)
    except Exception as exc:
        result = InferenceResult(
            ok=False,
            prediction=None,
            raw_response=None,
            error={"type": "load_error", "message": str(exc)},
            latency_seconds=0.0,
        )
    if not result.ok:
        return _failure_response(result)
    return StructureResponse(**result.as_dict())
