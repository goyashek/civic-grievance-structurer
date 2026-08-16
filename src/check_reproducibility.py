"""Write the DVC check record for the saved frozen evaluation."""

import json
from pathlib import Path

from .report_final import load_and_verify


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/reproducibility_results/dvc_report_check.json"


if __name__ == "__main__":
    runs, rows = load_and_verify()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "verified",
                "splits": {name: len(value) for name, value in rows.items()},
                "systems": {name: sorted(records) for name, records in runs.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
