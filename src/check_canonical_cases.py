"""Validate the manually written train and validation canonical cases."""

import json
from collections import Counter
from pathlib import Path

if __package__:
    from .schema import ALLOWED_DOMAINS, ALLOWED_ISSUES, ALLOWED_MISSING, ALLOWED_URGENCY
else:
    from schema import ALLOWED_DOMAINS, ALLOWED_ISSUES, ALLOWED_MISSING, ALLOWED_URGENCY


CANONICAL_PATH = Path(__file__).resolve().parents[1] / "data/canonical_cases.jsonl"
COVERAGE_PATH = Path(__file__).resolve().parents[1] / "data/coverage_matrix.json"
ROW_FIELDS = {
    "case_id",
    "split",
    "source_type",
    "service_domain",
    "issue_type",
    "facts",
    "intentionally_missing_fields",
    "urgency",
    "clean_formal_summary",
    "case_flags",
}
FACT_FIELDS = {"location", "event_date_or_time", "amount_inr", "service_identifier"}
MISSING_ORDER = (
    "exact_location",
    "date_or_time",
    "service_identifier",
    "transaction_or_reference_id",
    "amount",
    "supporting_evidence",
    "affected_person_or_group",
    "none",
)


EXPECTED_SPLIT_COUNTS = {"train": 120, "validation": 25}


def load_canonical_cases(path: Path = CANONICAL_PATH, expected_count: int = 145) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != ROW_FIELDS:
            raise ValueError(f"line {line_number}: wrong row fields")
        if row["split"] not in EXPECTED_SPLIT_COUNTS or row["source_type"] != "manual_canonical":
            raise ValueError(f"line {line_number}: row must be manual train or validation data")
        if row["service_domain"] not in ALLOWED_DOMAINS:
            raise ValueError(f"line {line_number}: unknown service domain")
        if row["issue_type"] not in ALLOWED_ISSUES:
            raise ValueError(f"line {line_number}: unknown issue type")
        if row["urgency"] not in ALLOWED_URGENCY:
            raise ValueError(f"line {line_number}: unknown urgency")
        facts = row["facts"]
        if set(facts) != FACT_FIELDS:
            raise ValueError(f"line {line_number}: wrong fact fields")
        for field in ("location", "event_date_or_time", "service_identifier"):
            if facts[field] is not None and not isinstance(facts[field], str):
                raise ValueError(f"line {line_number}: {field} must be a string or null")
        amount = facts["amount_inr"]
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0):
            raise ValueError(f"line {line_number}: amount_inr must be a non-negative number or null")
        missing = row["intentionally_missing_fields"]
        if not isinstance(missing, list) or not missing or not set(missing) <= ALLOWED_MISSING:
            raise ValueError(f"line {line_number}: invalid missing-information labels")
        if len(missing) != len(set(missing)) or missing != sorted(missing, key=MISSING_ORDER.index):
            raise ValueError(f"line {line_number}: missing-information labels are unordered or duplicated")
        if "none" in missing and missing != ["none"]:
            raise ValueError(f"line {line_number}: none cannot be combined with another label")
        if "exact_location" in missing and facts["location"] is not None:
            raise ValueError(f"line {line_number}: exact_location is present")
        if "date_or_time" in missing and facts["event_date_or_time"] is not None:
            raise ValueError(f"line {line_number}: date_or_time is present")
        if "service_identifier" in missing and facts["service_identifier"] is not None:
            raise ValueError(f"line {line_number}: service_identifier is present")
        if "amount" in missing and facts["amount_inr"] is not None:
            raise ValueError(f"line {line_number}: amount is present")
        if not row["clean_formal_summary"].strip():
            raise ValueError(f"line {line_number}: empty summary")
        if not isinstance(row["case_flags"], list) or not set(row["case_flags"]) <= {
            "hard_negative",
            "ambiguous_but_defensible",
        }:
            raise ValueError(f"line {line_number}: invalid case flag")
        rows.append(row)

    if len(rows) != expected_count:
        raise AssertionError(f"expected {expected_count} canonical cases, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise AssertionError("canonical case IDs must be unique")
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"expected split counts {EXPECTED_SPLIT_COUNTS}, found {dict(split_counts)}")
    if {row["service_domain"] for row in rows} != set(ALLOWED_DOMAINS):
        raise AssertionError("the batch must cover every service domain")
    if {row["issue_type"] for row in rows} != set(ALLOWED_ISSUES):
        raise AssertionError("the batch must cover every issue type")
    if {row["urgency"] for row in rows} != set(ALLOWED_URGENCY):
        raise AssertionError("the batch must cover every urgency label")
    if {
        label for row in rows for label in row["intentionally_missing_fields"]
    } != set(ALLOWED_MISSING):
        raise AssertionError("the batch must cover every missing-information label")
    flags = {flag for row in rows for flag in row["case_flags"]}
    if flags != {"hard_negative", "ambiguous_but_defensible"}:
        raise AssertionError("the batch must include both review flag types")
    return rows


def validate_coverage(rows: list[dict], path: Path = COVERAGE_PATH) -> None:
    coverage = json.loads(path.read_text(encoding="utf-8"))
    if coverage["batch"]["count"] != len(rows):
        raise AssertionError("coverage matrix count does not match the canonical batch")
    dimensions = coverage["written_batch_counts"]
    for field, counts in (
        ("service_domain", Counter(row["service_domain"] for row in rows)),
        ("issue_type", Counter(row["issue_type"] for row in rows)),
        ("urgency", Counter(row["urgency"] for row in rows)),
        (
            "intentionally_missing_fields",
            Counter(label for row in rows for label in row["intentionally_missing_fields"]),
        ),
    ):
        if dict(sorted(counts.items())) != dimensions[field]:
            raise AssertionError(f"coverage matrix does not match {field}")
    flags = Counter(flag for row in rows for flag in row["case_flags"])
    if dict(sorted(flags.items())) != coverage["batch"]["case_flags"]:
        raise AssertionError("coverage matrix does not match case flags")


if __name__ == "__main__":
    rows = load_canonical_cases()
    validate_coverage(rows)
    print(f"validated {len(rows)} canonical cases")
    print(f"splits: {dict(sorted(Counter(row['split'] for row in rows).items()))}")
    for field in ("service_domain", "issue_type", "urgency"):
        print(f"{field}: {dict(sorted(Counter(row[field] for row in rows).items()))}")
    print(
        "missing_information:",
        dict(sorted(Counter(label for row in rows for label in row["intentionally_missing_fields"]).items())),
    )
    print(
        "case_flags:",
        dict(sorted(Counter(flag for row in rows for flag in row["case_flags"]).items())),
    )
