"""Validate the small gold-example contract."""

import json
from pathlib import Path

if __package__:
    from .schema import SchemaError, validate_gold
else:
    from schema import SchemaError, validate_gold

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data/gold_examples.jsonl"

REQUIRED_ROW_FIELDS = {"case_id", "surface_id", "split", "source_type", "style", "complaint", "gold"}
def load_examples(path: Path = EXAMPLES_PATH, expected_count: int = 5) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != REQUIRED_ROW_FIELDS:
            raise SchemaError(f"line {line_number}: wrong row fields")
        if not row["complaint"].strip():
            raise SchemaError(f"line {line_number}: empty complaint")
        validate_gold(row["gold"])
        rows.append(row)
    assert len(rows) == expected_count, f"expected {expected_count} examples, found {len(rows)}"
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert len({row["surface_id"] for row in rows}) == len(rows)
    return rows


if __name__ == "__main__":
    print(f"validated {len(load_examples())} examples")
