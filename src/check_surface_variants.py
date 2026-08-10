"""Validate the grouped surface variants and split leakage checks."""

import difflib
import json
import re
from collections import Counter
from pathlib import Path

if __package__:
    from .build_surface_variants import draft_complaint
    from .check_canonical_cases import load_canonical_cases
    from .schema import validate_gold
else:
    from build_surface_variants import draft_complaint
    from check_canonical_cases import load_canonical_cases
    from schema import validate_gold


SURFACE_PATH = Path(__file__).resolve().parents[1] / "data/surface_variants.jsonl"
ROW_FIELDS = {"surface_id", "case_id", "split", "source_type", "style", "complaint", "gold"}
ALLOWED_STYLES = {
    "train": {"formal_english", "informal_english", "concise_english", "spelling_noise", "hinglish_roman"},
    "validation": {"formal_english", "informal_english"},
}
ALLOWED_SOURCE_TYPES = {"manual_surface", "llm_assisted_surface"}
EXPECTED_SPLIT_COUNTS = {"train": 600, "validation": 50}
EXPECTED_STYLE_COUNTS = {"train": 120, "validation": 25}
NEAR_DUPLICATE_THRESHOLD = 0.96


def expected_gold(canonical: dict) -> dict:
    facts = canonical["facts"]
    return {
        "service_domain": canonical["service_domain"],
        "issue_type": canonical["issue_type"],
        "location": facts["location"],
        "event_date_or_time": facts["event_date_or_time"],
        "amount_inr": facts["amount_inr"],
        "service_identifier": facts["service_identifier"],
        "urgency": canonical["urgency"],
        "missing_information": canonical["intentionally_missing_fields"],
        "formal_summary": canonical["clean_formal_summary"],
    }


def load_surface_variants(path: Path = SURFACE_PATH) -> list[dict]:
    canonical = {row["case_id"]: row for row in load_canonical_cases()}
    rows = []
    seen_surface_ids = set()
    seen_case_styles = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != ROW_FIELDS:
            raise ValueError(f"line {line_number}: wrong surface row fields")
        if row["surface_id"] in seen_surface_ids:
            raise ValueError(f"line {line_number}: duplicate surface ID")
        if row["case_id"] not in canonical:
            raise ValueError(f"line {line_number}: unknown canonical case ID")
        if row["split"] != canonical[row["case_id"]]["split"]:
            raise ValueError(f"line {line_number}: surface split differs from canonical split")
        if row["source_type"] not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"line {line_number}: unknown surface source type")
        if row["style"] not in ALLOWED_STYLES[row["split"]]:
            raise ValueError(f"line {line_number}: style is not allowed for the split")
        if not isinstance(row["complaint"], str) or not row["complaint"].strip():
            raise ValueError(f"line {line_number}: complaint must be non-empty")
        if row["source_type"] == "llm_assisted_surface":
            expected_complaint = draft_complaint(
                canonical[row["case_id"]]["clean_formal_summary"], row["style"]
            )
            if row["complaint"] != expected_complaint:
                raise ValueError(f"line {line_number}: assistant-drafted wording drifted from the builder")
        validate_gold(row["gold"])
        if row["gold"] != expected_gold(canonical[row["case_id"]]):
            raise ValueError(f"line {line_number}: gold does not match the canonical case")
        pair = (row["case_id"], row["style"])
        if pair in seen_case_styles:
            raise ValueError(f"line {line_number}: duplicate style for canonical case")
        seen_surface_ids.add(row["surface_id"])
        seen_case_styles.add(pair)
        rows.append(row)
    if not rows:
        raise AssertionError("surface variant file is empty")
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"expected split counts {EXPECTED_SPLIT_COUNTS}, found {dict(split_counts)}")
    style_counts = Counter((row["split"], row["style"]) for row in rows)
    for split, count in EXPECTED_STYLE_COUNTS.items():
        for style in ALLOWED_STYLES[split]:
            if style_counts[(split, style)] != count:
                raise AssertionError(f"expected {count} {split} rows for {style}")
    return rows


def normalized_complaint(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def check_duplicates(rows: list[dict]) -> None:
    exact = {}
    for row in rows:
        normalized = normalized_complaint(row["complaint"])
        previous = exact.setdefault(normalized, row)
        if previous["case_id"] != row["case_id"]:
            raise AssertionError(f"exact duplicate across canonical cases: {row['surface_id']}")

    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    # ponytail: O(n²) is enough for 600 x 50 split comparisons; use an index if this grows.
    for left in train:
        for right in validation:
            score = difflib.SequenceMatcher(
                None,
                normalized_complaint(left["complaint"]),
                normalized_complaint(right["complaint"]),
            ).ratio()
            if score >= NEAR_DUPLICATE_THRESHOLD:
                raise AssertionError(
                    f"near duplicate crosses splits ({score:.3f}): {left['surface_id']} and {right['surface_id']}"
                )


if __name__ == "__main__":
    rows = load_surface_variants()
    check_duplicates(rows)
    print(f"validated {len(rows)} surface variants")
    print("splits:", dict(sorted(Counter(row["split"] for row in rows).items())))
    print("styles:", dict(sorted(Counter(row["style"] for row in rows).items())))
