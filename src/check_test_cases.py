"""Validate the independent final test cases without generating predictions."""

import difflib
import re
from collections import Counter
from pathlib import Path

if __package__:
    from .check_examples import load_examples
    from .check_public_data import load_public_data, normalized
    from .check_surface_variants import load_surface_variants
else:
    from check_examples import load_examples
    from check_public_data import load_public_data, normalized
    from check_surface_variants import load_surface_variants


TEST_PATH = Path(__file__).resolve().parents[1] / "data/test_cases.jsonl"
NEAR_DUPLICATE_THRESHOLD = 0.90


def load_test_cases() -> list[dict]:
    rows = load_examples(TEST_PATH, expected_count=50)
    assert {row["split"] for row in rows} == {"test"}
    assert {row["source_type"] for row in rows} == {"manual_test"}
    assert {row["style"] for row in rows} == {"independent_manual"}
    assert len({row["gold"]["service_domain"] for row in rows}) == 7
    assert len({row["gold"]["issue_type"] for row in rows}) == 8
    assert len({row["gold"]["urgency"] for row in rows}) == 3
    return rows


def check_leakage(test_rows: list[dict]) -> None:
    controlled = load_surface_variants()
    public_training, _ = load_public_data()
    training_and_validation = controlled + public_training
    for test in test_rows:
        test_text = normalized(test["complaint"])
        for prior in training_and_validation:
            score = difflib.SequenceMatcher(None, test_text, normalized(prior["complaint"])).ratio()
            if score >= NEAR_DUPLICATE_THRESHOLD:
                raise AssertionError(
                    f"near duplicate ({score:.3f}): {test['case_id']} and {prior['case_id']}"
                )


if __name__ == "__main__":
    rows = load_test_cases()
    check_leakage(rows)
    print(f"validated {len(rows)} independent test cases")
    for field in ("service_domain", "issue_type", "urgency"):
        print(field + ":", dict(sorted(Counter(row["gold"][field] for row in rows).items())))
