"""Validate the manually checked annotation examples."""

from pathlib import Path

if __package__:
    from .check_examples import load_examples
else:
    from check_examples import load_examples

ANNOTATION_PATH = Path(__file__).resolve().parents[1] / "data/annotation_examples.jsonl"


def load_annotation_examples() -> list[dict]:
    rows = load_examples(ANNOTATION_PATH, expected_count=20)
    assert {row["split"] for row in rows} == {"development"}
    assert {row["source_type"] for row in rows} == {"manual_annotation"}
    assert {row["gold"]["service_domain"] for row in rows} == {
        "public_transport",
        "water_supply",
        "sanitation_and_waste",
        "roads_and_streetlights",
        "electricity",
        "welfare_or_document_service",
        "other",
    }
    assert {row["gold"]["issue_type"] for row in rows} == {
        "delay_or_non_arrival",
        "service_outage_or_non_delivery",
        "damaged_infrastructure",
        "overcharging_or_payment_problem",
        "record_or_document_error",
        "staff_conduct",
        "safety_or_health_hazard",
        "other",
    }
    assert {row["gold"]["urgency"] for row in rows} == {
        "routine",
        "time_sensitive",
        "safety_critical",
    }
    assert {
        label
        for row in rows
        for label in row["gold"]["missing_information"]
    } == {
        "exact_location",
        "date_or_time",
        "service_identifier",
        "transaction_or_reference_id",
        "amount",
        "supporting_evidence",
        "affected_person_or_group",
        "none",
    }
    return rows


if __name__ == "__main__":
    print(f"validated {len(load_annotation_examples())} annotation examples")
