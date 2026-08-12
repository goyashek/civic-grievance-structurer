"""Verify the frozen data files without reading or producing model predictions."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/dataset_manifest.json"


def check_dataset_freeze() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"dataset hash changed: {relative_path}")


if __name__ == "__main__":
    check_dataset_freeze()
    print("dataset hashes match the frozen manifest")
