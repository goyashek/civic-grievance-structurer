"""Shared labels and validation for structured grievance outputs."""

import math

from typing import Any

SCHEMA_VERSION = "1.0"


class SchemaError(ValueError):
    """Raised when a structured grievance output violates the contract."""


GOLD_FIELDS = frozenset(
    {
        "service_domain",
        "issue_type",
        "location",
        "event_date_or_time",
        "amount_inr",
        "service_identifier",
        "urgency",
        "missing_information",
        "formal_summary",
    }
)
ALLOWED_DOMAINS = frozenset(
    {
        "public_transport",
        "water_supply",
        "sanitation_and_waste",
        "roads_and_streetlights",
        "electricity",
        "welfare_or_document_service",
        "other",
    }
)
ALLOWED_ISSUES = frozenset(
    {
        "delay_or_non_arrival",
        "service_outage_or_non_delivery",
        "damaged_infrastructure",
        "overcharging_or_payment_problem",
        "record_or_document_error",
        "staff_conduct",
        "safety_or_health_hazard",
        "other",
    }
)
ALLOWED_URGENCY = frozenset({"routine", "time_sensitive", "safety_critical"})
ALLOWED_MISSING = frozenset(
    {
        "exact_location",
        "date_or_time",
        "service_identifier",
        "transaction_or_reference_id",
        "amount",
        "supporting_evidence",
        "affected_person_or_group",
        "none",
    }
)
MISSING_INFORMATION_ORDER = (
    "exact_location",
    "date_or_time",
    "service_identifier",
    "transaction_or_reference_id",
    "amount",
    "supporting_evidence",
    "affected_person_or_group",
    "none",
)


def validate_gold(gold: dict[str, Any]) -> None:
    """Validate one structured grievance output."""

    if set(gold) != GOLD_FIELDS:
        raise SchemaError("gold output must contain exactly the shared fields")
    if gold["service_domain"] not in ALLOWED_DOMAINS:
        raise SchemaError("unknown service domain")
    if gold["issue_type"] not in ALLOWED_ISSUES:
        raise SchemaError("unknown issue type")
    if gold["urgency"] not in ALLOWED_URGENCY:
        raise SchemaError("unknown urgency")

    missing = gold["missing_information"]
    if not isinstance(missing, list) or not missing:
        raise SchemaError("missing_information must be a non-empty list")
    if not set(missing) <= ALLOWED_MISSING:
        raise SchemaError("unknown missing-information label")
    if len(missing) != len(set(missing)):
        raise SchemaError("missing_information cannot contain duplicates")
    missing_order = {label: index for index, label in enumerate(MISSING_INFORMATION_ORDER)}
    if missing != sorted(missing, key=missing_order.__getitem__):
        raise SchemaError("missing_information must use the shared label order")
    if "none" in missing and missing != ["none"]:
        raise SchemaError("none cannot be combined with another missing label")

    for field in ("location", "event_date_or_time", "service_identifier"):
        if gold[field] is not None and not isinstance(gold[field], str):
            raise SchemaError(f"{field} must be a string or null")
        if isinstance(gold[field], str) and not gold[field].strip():
            raise SchemaError(f"{field} cannot be empty")

    amount = gold["amount_inr"]
    if amount is not None and (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(amount)
        or amount < 0
    ):
        raise SchemaError("amount_inr must be a non-negative number or null")
    if not isinstance(gold["formal_summary"], str) or not gold["formal_summary"].strip():
        raise SchemaError("formal_summary must be a non-empty string")
