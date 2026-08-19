"""Shared frozen QLoRA inference path for the local CLI and later services."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import SchemaError, validate_gold


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = json.loads(
    (ROOT / "data/final_results/frozen_system_manifest.json").read_text(encoding="utf-8")
)
MODEL_NAME = FROZEN_MANIFEST["model_name"]
MODEL_REVISION = FROZEN_MANIFEST["model_revision"]
MAX_NEW_TOKENS = FROZEN_MANIFEST["decoding"]["max_new_tokens"]
MAX_COMPLAINT_CHARS = 4000
ADAPTER_DIR = ROOT / "data/model_registry/artifacts/qlora_final_adapter"
SYSTEM_PROMPT = (
    "Structure one public-service complaint as exactly one JSON object. Use these fields in this order: "
    "service_domain, issue_type, location, event_date_or_time, amount_inr, service_identifier, urgency, "
    "missing_information, formal_summary. "
    "Allowed service_domain values: ['public_transport', 'water_supply', 'sanitation_and_waste', "
    "'roads_and_streetlights', 'electricity', 'welfare_or_document_service', 'other']. "
    "Allowed issue_type values: ['delay_or_non_arrival', 'service_outage_or_non_delivery', "
    "'damaged_infrastructure', 'overcharging_or_payment_problem', 'record_or_document_error', "
    "'staff_conduct', 'safety_or_health_hazard', 'other']. "
    "Allowed urgency values: ['routine', 'time_sensitive', 'safety_critical']. "
    "Allowed missing_information values: ['exact_location', 'date_or_time', 'service_identifier', "
    "'transaction_or_reference_id', 'amount', 'supporting_evidence', 'affected_person_or_group', 'none']. "
    "Use null for absent scalar facts. Missing information must be a non-empty ordered list. "
    "Use the none label only when no important detail is missing. Do not guess facts. "
    "The formal summary must be one neutral sentence. Return no reasoning, markdown, or commentary."
)


@dataclass(frozen=True)
class InferenceResult:
    """One explicit success or failure from the shared inference path."""

    ok: bool
    prediction: dict[str, Any] | None
    raw_response: str | None
    error: dict[str, str] | None
    latency_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def messages_for(complaint: str) -> list[dict[str, str]]:
    """Build the same no-demonstration prompt used by the frozen QLoRA run."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]


def validate_response(raw_response: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Parse one raw response without applying the evaluator's optional repair."""

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return None, {"type": "invalid_json", "message": str(exc)}
    if not isinstance(parsed, dict):
        return None, {"type": "schema_invalid", "message": "response JSON must be an object"}
    try:
        validate_gold(parsed)
    except (SchemaError, TypeError) as exc:
        return None, {"type": "schema_invalid", "message": str(exc)}
    return parsed, None


def resolve_adapter_path(adapter_path: Path | None = None) -> Path:
    """Use the registered adapter, extracting it from the local frozen archive if needed."""

    if adapter_path is not None:
        if not (adapter_path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"adapter weights not found at {adapter_path}")
        return adapter_path
    if (ADAPTER_DIR / "adapter_model.safetensors").exists():
        return ADAPTER_DIR
    from .model_registry import extract_adapter

    return extract_adapter()


def load_frozen_model(adapter_path: Path | None = None):
    """Load the frozen base model and registered adapter lazily."""

    adapter_dir = resolve_adapter_path(adapter_path)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=quantization_config,
        dtype=model_dtype,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.eos_token_id = tokenizer.eos_token_id
    base_model.config.use_cache = True
    return tokenizer, PeftModel.from_pretrained(base_model, adapter_dir)


class CivicStructInference:
    """Generate and strictly validate one complaint at a time."""

    def __init__(self, adapter_path: Path | None = None):
        self.tokenizer, self.model = load_frozen_model(adapter_path)

    def infer(self, complaint: str) -> InferenceResult:
        started = time.perf_counter()
        if not isinstance(complaint, str) or not complaint.strip():
            return InferenceResult(
                ok=False,
                prediction=None,
                raw_response=None,
                error={"type": "invalid_input", "message": "complaint must be non-empty text"},
                latency_seconds=0.0,
            )
        try:
            encoded = self.tokenizer.apply_chat_template(
                [messages_for(complaint)],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                enable_thinking=False,
            )
        except TypeError:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    [messages_for(complaint)],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
            except Exception as exc:
                return InferenceResult(
                    ok=False,
                    prediction=None,
                    raw_response=None,
                    error={"type": "preprocessing_error", "message": str(exc)},
                    latency_seconds=time.perf_counter() - started,
                )
        except Exception as exc:
            return InferenceResult(
                ok=False,
                prediction=None,
                raw_response=None,
                error={"type": "preprocessing_error", "message": str(exc)},
                latency_seconds=time.perf_counter() - started,
            )
        device = next(self.model.parameters()).device
        encoded = encoded.to(device)
        prompt_length = encoded["input_ids"].shape[1]
        try:
            import torch

            self.model.eval()
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            raw_response = self.tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
        except Exception as exc:
            return InferenceResult(
                ok=False,
                prediction=None,
                raw_response=None,
                error={"type": "generation_error", "message": str(exc)},
                latency_seconds=time.perf_counter() - started,
            )
        prediction, error = validate_response(raw_response)
        return InferenceResult(
            ok=prediction is not None,
            prediction=prediction,
            raw_response=raw_response,
            error=error,
            latency_seconds=time.perf_counter() - started,
        )
