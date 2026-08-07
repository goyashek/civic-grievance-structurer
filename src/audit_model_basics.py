"""Audit raw model outputs saved in the model-basics notebook."""

import json
import re
from pathlib import Path

if __package__:
    from .schema import SchemaError, validate_gold
else:
    from schema import SchemaError, validate_gold

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks/model_basics.ipynb"
OUTPUT_PATTERN = re.compile(
    r"case: (?P<case_id>[^\n]+)\n"
    r"latency: (?P<latency>[0-9.]+) seconds\n"
    r"(?P<body>\{.*?\})(?=\n\ncase:|\s*\Z)",
    re.DOTALL,
)


def load_saved_outputs(path: Path = NOTEBOOK_PATH) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    streams = [
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    ]
    text = next(stream for stream in streams if "case: civic-example-" in stream)
    rows = []
    for match in OUTPUT_PATTERN.finditer(text):
        row = {
            "case_id": match["case_id"],
            "latency_seconds": float(match["latency"]),
            "response": json.loads(match["body"]),
        }
        try:
            validate_gold(row["response"])
        except SchemaError as error:
            row["schema_valid"] = False
            row["schema_error"] = str(error)
        else:
            row["schema_valid"] = True
        rows.append(row)
    assert len(rows) == 5, f"expected five saved outputs, found {len(rows)}"
    return rows


if __name__ == "__main__":
    rows = load_saved_outputs()
    print(json.dumps(rows, indent=2))
    print(f"schema-valid outputs: {sum(row['schema_valid'] for row in rows)}/{len(rows)}")
