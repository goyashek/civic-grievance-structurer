"""Validate the audited model-selection complaints."""

from pathlib import Path

if __package__:
    from .check_examples import load_examples
else:
    from check_examples import load_examples

BAKEOFF_PATH = Path(__file__).resolve().parents[1] / "data/model_selection_development.jsonl"


def load_bakeoff() -> list[dict]:
    rows = load_examples(BAKEOFF_PATH, expected_count=40)
    assert {row["split"] for row in rows} == {"development"}
    assert {row["source_type"] for row in rows} == {"manual_audit"}
    return rows


if __name__ == "__main__":
    rows = load_bakeoff()
    print(f"validated {len(rows)} audited development complaints")
