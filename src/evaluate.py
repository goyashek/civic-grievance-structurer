"""Shared metrics for raw structured grievance outputs."""

import json
from typing import Any

from .schema import SchemaError, validate_gold


CATEGORICAL_FIELDS = ("service_domain", "issue_type", "urgency")
FACT_FIELDS = ("location", "event_date_or_time", "amount_inr", "service_identifier")
EVALUATION_VERSION = "1.0"


def _macro_f1(gold_values: list[Any], predicted_values: list[Any]) -> float:
    scores = []
    pairs = list(zip(gold_values, predicted_values))
    for label in sorted(set(gold_values)):
        true_positive = sum(gold == predicted == label for gold, predicted in pairs)
        false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
        false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
        scores.append(
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
    return sum(scores) / len(scores)


def _missing_macro_f1(golds: list[dict], predictions: list[dict | None]) -> float:
    labels = sorted({label for gold in golds for label in gold["missing_information"]})
    scores = []
    for label in labels:
        gold_values = [label in gold["missing_information"] for gold in golds]
        predicted_values = [
            prediction is not None and label in prediction["missing_information"]
            for prediction in predictions
        ]
        pairs = list(zip(gold_values, predicted_values))
        true_positive = sum(gold and predicted for gold, predicted in pairs)
        false_positive = sum(not gold and predicted for gold, predicted in pairs)
        false_negative = sum(gold and not predicted for gold, predicted in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _normalize_text(value: Any) -> Any:
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def _match_rate(
    golds: list[dict],
    predictions: list[dict | None],
    field: str,
    *,
    normalize: bool = False,
) -> float:
    def value(item: Any) -> Any:
        return _normalize_text(item) if normalize else item

    matches = sum(
        prediction is not None and value(prediction[field]) == value(gold[field])
        for gold, prediction in zip(golds, predictions)
    )
    return matches / len(golds)


def _field_metrics(golds: list[dict], predictions: list[dict | None]) -> dict[str, float]:
    metrics = {
        f"{field}_macro_f1": _macro_f1(
            [gold[field] for gold in golds],
            [prediction[field] if prediction is not None else None for prediction in predictions],
        )
        for field in CATEGORICAL_FIELDS
    }
    metrics.update(
        {
            "missing_information_macro_f1": _missing_macro_f1(golds, predictions),
            "location_normalized_match": _match_rate(
                golds, predictions, "location", normalize=True
            ),
            "event_date_or_time_normalized_match": _match_rate(
                golds, predictions, "event_date_or_time", normalize=True
            ),
            "amount_inr_exact_match": _match_rate(golds, predictions, "amount_inr"),
            "service_identifier_exact_match": _match_rate(golds, predictions, "service_identifier"),
        }
    )
    return metrics


def _hallucination(
    golds: list[dict], predictions: list[dict | None]
) -> dict[str, int | float | None]:
    pairs = [
        (field, gold[field], prediction[field])
        for gold, prediction in zip(golds, predictions)
        if prediction is not None
        for field in FACT_FIELDS
        if prediction[field] is not None
    ]
    hallucinated = sum(
        (_normalize_text(gold) != _normalize_text(predicted))
        if field in {"location", "event_date_or_time"}
        else gold != predicted
        for field, gold, predicted in pairs
    )
    return {
        "count": hallucinated,
        "predicted_non_null_count": len(pairs),
        "rate": None if not pairs else hallucinated / len(pairs),
    }


def _parse_and_validate(raw_output: Any) -> tuple[bool, dict | None]:
    if not isinstance(raw_output, str):
        return False, None
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return False, None
    if not isinstance(parsed, dict):
        return True, None
    try:
        validate_gold(parsed)
    except (SchemaError, TypeError):
        return True, None
    return True, parsed


def _repair_wrapped_json(raw_output: Any) -> dict | None:
    if not isinstance(raw_output, str):
        return None
    start, end = raw_output.find("{"), raw_output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw_output[start : end + 1])
        if not isinstance(parsed, dict):
            return None
        validate_gold(parsed)
    except (json.JSONDecodeError, SchemaError, TypeError):
        return None
    return parsed


def evaluate_outputs(golds: list[dict], raw_outputs: list[Any]) -> dict[str, Any]:
    """Score raw model text while keeping strict and repaired views separate."""

    if not golds or len(golds) != len(raw_outputs):
        raise ValueError("gold outputs and raw outputs must have the same non-zero length")
    for gold in golds:
        validate_gold(gold)

    parsed = [_parse_and_validate(raw) for raw in raw_outputs]
    strict_predictions = [prediction for _, prediction in parsed]
    strict_valid_count = sum(prediction is not None for prediction in strict_predictions)
    json_parse_count = sum(parsed_ok for parsed_ok, _ in parsed)

    repaired_predictions = []
    repairable_count = 0
    for raw, strict_prediction in zip(raw_outputs, strict_predictions):
        repaired = strict_prediction or _repair_wrapped_json(raw)
        repairable_count += strict_prediction is None and repaired is not None
        repaired_predictions.append(repaired)

    valid_pairs = [
        (gold, prediction)
        for gold, prediction in zip(golds, strict_predictions)
        if prediction is not None
    ]
    conditional_metrics = None
    if valid_pairs:
        conditional_golds, conditional_predictions = map(list, zip(*valid_pairs))
        conditional_metrics = _field_metrics(conditional_golds, conditional_predictions)

    total = len(golds)
    repaired_valid_count = sum(prediction is not None for prediction in repaired_predictions)
    return {
        "evaluation_version": EVALUATION_VERSION,
        "total_outputs": total,
        "strict": {
            "json_parse_count": json_parse_count,
            "invalid_json_count": total - json_parse_count,
            "schema_invalid_count": json_parse_count - strict_valid_count,
            "schema_valid_count": strict_valid_count,
            "schema_validity_rate": strict_valid_count / total,
            "end_to_end_field_metrics": _field_metrics(golds, strict_predictions),
            "hallucinated_non_null_fields": _hallucination(golds, strict_predictions),
        },
        "conditional_valid": {
            "denominator": strict_valid_count,
            "field_metrics": conditional_metrics,
        },
        "repaired": {
            "repairable_count": repairable_count,
            "repairable_json_rate": repairable_count / total,
            "schema_valid_count": repaired_valid_count,
            "schema_validity_rate": repaired_valid_count / total,
            "end_to_end_field_metrics": _field_metrics(golds, repaired_predictions),
            "hallucinated_non_null_fields": _hallucination(golds, repaired_predictions),
        },
    }
