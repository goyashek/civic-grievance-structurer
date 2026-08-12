"""Validate the licensed public training and external evaluation slices."""

import json
import re
from collections import Counter
from pathlib import Path

if __package__:
    from .schema import validate_gold
else:
    from schema import validate_gold


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TRAINING_PATH = ROOT / "data/public_training_examples.jsonl"
EXTERNAL_EVAL_PATH = ROOT / "data/external_civic_eval.jsonl"
MANIFEST_PATH = ROOT / "data/public_data_manifest.json"
ROW_FIELDS = {
    "case_id",
    "split",
    "source_type",
    "style",
    "source_dataset",
    "source_row_sha256",
    "complaint",
    "gold",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
URL = re.compile(r"\b(?:https?://|www\.)", re.IGNORECASE)
PHONE = re.compile(r"\b(?:\+?\d[\s().-]*){10,}\b")


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def load_slice(path: Path, split: str, source_dataset: str, expected_count: int) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != ROW_FIELDS:
            raise ValueError(f"{path.name}:{line_number}: wrong row fields")
        if row["split"] != split:
            raise ValueError(f"{path.name}:{line_number}: wrong split")
        if row["source_type"] != "licensed_public_derived":
            raise ValueError(f"{path.name}:{line_number}: wrong source type")
        if row["style"] != "public_english" or row["source_dataset"] != source_dataset:
            raise ValueError(f"{path.name}:{line_number}: wrong style or dataset")
        if not HEX_SHA256.fullmatch(row["source_row_sha256"]):
            raise ValueError(f"{path.name}:{line_number}: invalid source-row hash")
        complaint = row["complaint"]
        if not isinstance(complaint, str) or not complaint.strip():
            raise ValueError(f"{path.name}:{line_number}: complaint must be non-empty")
        if EMAIL.search(complaint) or URL.search(complaint) or PHONE.search(complaint):
            raise ValueError(f"{path.name}:{line_number}: possible direct identifier")
        validate_gold(row["gold"])
        location = row["gold"]["location"]
        if not location or location.casefold() not in complaint.casefold():
            raise ValueError(f"{path.name}:{line_number}: generalized location is not grounded")
        if row["gold"]["missing_information"] != ["exact_location"]:
            raise ValueError(f"{path.name}:{line_number}: deidentified rows must flag exact_location")
        rows.append(row)

    if len(rows) != expected_count:
        raise AssertionError(f"expected {expected_count} rows in {path.name}, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise AssertionError(f"duplicate case ID in {path.name}")
    if len({row["source_row_sha256"] for row in rows}) != len(rows):
        raise AssertionError(f"duplicate source-row hash in {path.name}")
    return rows


def load_public_data() -> tuple[list[dict], list[dict]]:
    training = load_slice(PUBLIC_TRAINING_PATH, "train", "opencity_ichangemycity", 40)
    external = load_slice(EXTERNAL_EVAL_PATH, "external_test", "sandiego_get_it_done", 20)
    all_rows = training + external
    if len({normalized(row["complaint"]) for row in all_rows}) != len(all_rows):
        raise AssertionError("duplicate complaint across public slices")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["derived_row_counts"] != {"public_training": len(training), "external_evaluation": len(external)}:
        raise AssertionError("public-data manifest counts do not match the files")
    return training, external


if __name__ == "__main__":
    training, external = load_public_data()
    print(f"validated {len(training)} licensed public training rows")
    print(f"validated {len(external)} external evaluation rows")
    print("training domains:", dict(sorted(Counter(row["gold"]["service_domain"] for row in training).items())))
    print("external domains:", dict(sorted(Counter(row["gold"]["service_domain"] for row in external).items())))
