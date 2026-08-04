"""Validate the small gold-example contract."""

import json
from pathlib import Path

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data/gold_examples.jsonl"

REQUIRED_ROW_FIELDS = {"case_id", "surface_id", "split", "source_type", "style", "complaint", "gold"}
REQUIRED_GOLD_FIELDS = {
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
ALLOWED_DOMAINS = {
    "public_transport",
    "water_supply",
    "sanitation_and_waste",
    "roads_and_streetlights",
    "electricity",
    "welfare_or_document_service",
    "other",
}
ALLOWED_ISSUES = {
    "delay_or_non_arrival",
    "service_outage_or_non_delivery",
    "damaged_infrastructure",
    "overcharging_or_payment_problem",
    "record_or_document_error",
    "staff_conduct",
    "safety_or_health_hazard",
    "other",
}
ALLOWED_URGENCY = {"routine", "time_sensitive", "safety_critical"}
ALLOWED_MISSING = {
    "exact_location",
    "date_or_time",
    "service_identifier",
    "transaction_or_reference_id",
    "amount",
    "supporting_evidence",
    "affected_person_or_group",
    "none",
}


def _validate_gold(gold: dict) -> None:
    assert set(gold) == REQUIRED_GOLD_FIELDS
    assert gold["service_domain"] in ALLOWED_DOMAINS
    assert gold["issue_type"] in ALLOWED_ISSUES
    assert gold["urgency"] in ALLOWED_URGENCY
    assert isinstance(gold["missing_information"], list)
    assert gold["missing_information"]
    assert set(gold["missing_information"]) <= ALLOWED_MISSING
    assert "none" not in gold["missing_information"] or gold["missing_information"] == ["none"]
    for field in ("location", "event_date_or_time", "service_identifier"):
        assert gold[field] is None or isinstance(gold[field], str)
    assert gold["amount_inr"] is None or (isinstance(gold["amount_inr"], (int, float)) and gold["amount_inr"] >= 0)
    assert isinstance(gold["formal_summary"], str) and gold["formal_summary"].strip()


def load_examples(path: Path = EXAMPLES_PATH) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert set(row) == REQUIRED_ROW_FIELDS, f"line {line_number}: wrong row fields"
        assert row["complaint"].strip()
        _validate_gold(row["gold"])
        rows.append(row)
    assert len(rows) == 5, f"expected five examples, found {len(rows)}"
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert len({row["surface_id"] for row in rows}) == len(rows)
    return rows


if __name__ == "__main__":
    print(f"validated {len(load_examples())} examples")
