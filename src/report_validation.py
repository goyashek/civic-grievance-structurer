"""Recheck the saved validation runs and print their result tables.

Run it from the repository root with `python3 -m src.report_validation`, because the
shared evaluator is imported as part of the package.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .check_surface_variants import load_surface_variants
from .evaluate import CATEGORICAL_FIELDS, FACT_FIELDS
from .evaluate import _normalize_text, _parse_and_validate, evaluate_outputs


RESULTS_DIR = Path(__file__).resolve().parents[1] / "data/validation_results"
FAILURE_PATH = RESULTS_DIR / "validation_failures.json"
SYSTEM_ORDER = ("deterministic_rules", "zero_shot", "static_few_shot", "retrieved_few_shot", "qlora")


def load_runs() -> dict[str, dict]:
    """Read the saved runs in reporting order."""

    runs = json.loads((RESULTS_DIR / "validation_baselines.json").read_text())
    runs["qlora"] = json.loads((RESULTS_DIR / "qlora_validation_predictions.json").read_text())
    for run in runs.values():
        tokens = [output["prompt_tokens"] for output in run["outputs"]]
        run["mean_prompt_tokens"] = sum(tokens) / len(tokens)
    return {name: runs[name] for name in SYSTEM_ORDER}


def load_pairs(run: dict) -> list[dict]:
    """Join every saved response to its validation row."""

    rows = {
        row["surface_id"]: row for row in load_surface_variants() if row["split"] == "validation"
    }
    if sorted(output["surface_id"] for output in run["outputs"]) != sorted(rows):
        raise AssertionError("saved outputs do not cover the validation split exactly")
    return [
        {
            "surface_id": output["surface_id"],
            "style": rows[output["surface_id"]]["style"],
            "gold": rows[output["surface_id"]]["gold"],
            "response": output["response"],
        }
        for output in run["outputs"]
    ]


def score(pairs: list[dict]) -> dict:
    return evaluate_outputs([pair["gold"] for pair in pairs], [pair["response"] for pair in pairs])


def rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return round(value, 6) if isinstance(value, float) else value


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95 percent Wilson score interval, which stays sane at rates near 0 and 1."""

    z = 1.96
    rate = successes / total
    divisor = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / divisor
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / divisor
    return max(0.0, centre - spread), min(1.0, centre + spread)


def categorize_failure(gold: dict, raw_output: str) -> list[str]:
    """Name what went wrong in one response, or return nothing for a clean one."""

    parsed_ok, prediction = _parse_and_validate(raw_output)
    if not parsed_ok:
        return ["invalid_json"]
    if prediction is None:
        return ["schema_invalid"]

    problems = [f"wrong_{field}" for field in CATEGORICAL_FIELDS if prediction[field] != gold[field]]
    if prediction["missing_information"] != gold["missing_information"]:
        problems.append("missing_information_mismatch")
    for field in FACT_FIELDS:
        expected, predicted = gold[field], prediction[field]
        if expected is None and predicted is not None:
            problems.append("invented_fact")
        elif expected is not None and predicted is None:
            problems.append("dropped_fact")
        elif _normalize_text(expected) != _normalize_text(predicted):
            problems.append("wrong_fact_value")
    return problems


def number(value: Any, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def print_table(header: list[str], rows: list[list[str]]) -> None:
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def strict_row(name: str, run: dict, result: dict) -> list[str]:
    strict = result["strict"]
    fields = strict["end_to_end_field_metrics"]
    low, high = wilson_interval(strict["schema_valid_count"], result["total_outputs"])
    return [
        name,
        f"{strict['schema_validity_rate']:.2f} ({low:.2f} to {high:.2f})",
        number(fields["service_domain_macro_f1"]),
        number(fields["issue_type_macro_f1"]),
        number(fields["urgency_macro_f1"]),
        number(fields["missing_information_macro_f1"]),
        number(strict["hallucinated_non_null_fields"]["rate"]),
        f"{run['mean_prompt_tokens']:.0f}",
        number(run["mean_latency_seconds"], 2),
    ]


def conditional_row(name: str, result: dict) -> list[str]:
    fields = result["conditional_valid"]["field_metrics"] or {}
    return [
        name,
        str(result["conditional_valid"]["denominator"]),
        number(fields.get("service_domain_macro_f1")),
        number(fields.get("issue_type_macro_f1")),
        number(fields.get("missing_information_macro_f1")),
        number(fields.get("location_normalized_match")),
        number(fields.get("service_identifier_exact_match")),
        number(result["repaired"]["repairable_json_rate"], 2),
        number(result["repaired"]["schema_validity_rate"], 2),
    ]


if __name__ == "__main__":
    runs = load_runs()
    pairs = {name: load_pairs(run) for name, run in runs.items()}
    scores = {}
    for name, run in runs.items():
        scores[name] = score(pairs[name])
        if rounded(scores[name]) != rounded(run["scores"]):
            raise AssertionError(f"recomputed scores do not match the saved scores for {name}")
    print(f"rechecked {len(runs)} systems on {len(pairs['qlora'])} rows against their saved scores")

    print_table(
        ["system", "schema valid", "domain F1", "issue F1", "urgency F1", "missing-info F1",
         "halluc. rate", "prompt tokens", "s per case"],
        [strict_row(name, runs[name], scores[name]) for name in runs],
    )
    print_table(
        ["system", "conditional n", "domain F1", "issue F1", "missing-info F1", "location",
         "service id", "repairable", "repaired valid"],
        [conditional_row(name, scores[name]) for name in runs],
    )

    style_rows = []
    for name in runs:
        for style in sorted({pair["style"] for pair in pairs[name]}):
            styled = score([pair for pair in pairs[name] if pair["style"] == style])
            fields = styled["strict"]["end_to_end_field_metrics"]
            style_rows.append(
                [
                    name,
                    style,
                    number(styled["strict"]["schema_validity_rate"], 2),
                    number(fields["service_domain_macro_f1"]),
                    number(fields["issue_type_macro_f1"]),
                    number(fields["missing_information_macro_f1"]),
                ]
            )
    print_table(
        ["system", "style", "schema valid", "domain F1", "issue F1", "missing-info F1"], style_rows
    )

    records = []
    for name in runs:
        for pair in pairs[name]:
            problems = categorize_failure(pair["gold"], pair["response"])
            if problems:
                records.append(
                    {
                        "system": name,
                        "surface_id": pair["surface_id"],
                        "style": pair["style"],
                        "categories": dict(sorted(Counter(problems).items())),
                    }
                )
    FAILURE_PATH.write_text(json.dumps(records, indent=2) + "\n")

    counts = Counter(
        (record["system"], category) for record in records for category in record["categories"]
    )
    failure_rows = [
        [category] + [str(counts.get((name, category), 0)) for name in runs]
        for category in sorted({category for _, category in counts})
    ]
    failure_rows.append(
        ["responses with any failure"]
        + [str(sum(record["system"] == name for record in records)) for name in runs]
    )
    print_table(["category"] + list(runs), failure_rows)
    print(f"\nwrote {FAILURE_PATH.relative_to(RESULTS_DIR.parents[1])} with {len(records)} records")
