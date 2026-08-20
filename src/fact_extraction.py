"""Supplemental fact extraction metrics built on the frozen strict parser."""

from typing import Any

from .evaluate import FACT_FIELDS, TEXT_FACT_FIELDS, _normalize_text, _parse_and_validate
from .schema import validate_gold


def fact_extraction_metrics(
    golds: list[dict], raw_outputs: list[Any]
) -> dict[str, int | float | None]:
    """Score strict extraction of non-null factual fields."""

    if not golds or len(golds) != len(raw_outputs):
        raise ValueError("gold outputs and raw outputs must have the same non-zero length")
    for gold in golds:
        validate_gold(gold)
    predictions = [_parse_and_validate(raw)[1] for raw in raw_outputs]

    gold_non_null = sum(gold[field] is not None for gold in golds for field in FACT_FIELDS)
    predicted_non_null = sum(
        prediction is not None and prediction[field] is not None
        for prediction in predictions
        for field in FACT_FIELDS
    )
    covered = sum(
        gold[field] is not None
        and prediction is not None
        and prediction[field] is not None
        for gold, prediction in zip(golds, predictions)
        for field in FACT_FIELDS
    )
    correct = sum(
        gold[field] is not None
        and prediction is not None
        and prediction[field] is not None
        and (
            _normalize_text(gold[field]) == _normalize_text(prediction[field])
            if field in TEXT_FACT_FIELDS
            else gold[field] == prediction[field]
        )
        for gold, prediction in zip(golds, predictions)
        for field in FACT_FIELDS
    )

    return {
        "correct_count": correct,
        "predicted_non_null_count": predicted_non_null,
        "gold_non_null_count": gold_non_null,
        "covered_gold_non_null_count": covered,
        "precision": None if not predicted_non_null else correct / predicted_non_null,
        "recall": None if not gold_non_null else correct / gold_non_null,
        "coverage": None if not gold_non_null else covered / gold_non_null,
        "f1": (
            None
            if predicted_non_null + gold_non_null == 0
            else 2 * correct / (predicted_non_null + gold_non_null)
        ),
    }
