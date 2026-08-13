"""Verify the final run and add confidence intervals and review scores."""

import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import evaluate_outputs
from .report_validation import rounded, wilson_interval


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data/final_results"
ARCHIVE = RESULTS_DIR / "civicstruct_final_results.zip"
REVIEW_PATH = RESULTS_DIR / "summary_review.json"
METRICS_PATH = RESULTS_DIR / "final_metrics.json"
SYSTEM_ORDER = (
    "deterministic_rules",
    "zero_shot",
    "static_few_shot",
    "retrieved_few_shot",
    "qlora",
)
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile for an already finite sample."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def primary_metrics(score: dict) -> dict[str, float | None]:
    strict = score["strict"]
    fields = strict["end_to_end_field_metrics"]
    return {
        "schema_validity_rate": strict["schema_validity_rate"],
        "service_domain_macro_f1": fields["service_domain_macro_f1"],
        "issue_type_macro_f1": fields["issue_type_macro_f1"],
        "missing_information_macro_f1": fields["missing_information_macro_f1"],
        "hallucinated_non_null_field_rate": strict["hallucinated_non_null_fields"]["rate"],
    }


def confidence_intervals(rows: list[dict], runs: dict[str, dict]) -> dict[str, dict]:
    """Use Wilson for validity and paired row bootstrap for semantic metrics."""

    golds = [row["gold"] for row in rows]
    responses = {
        name: {output["case_id"]: output["response"] for output in runs[name]["outputs"]}
        for name in SYSTEM_ORDER
    }
    if any(set(outputs) != {row["case_id"] for row in rows} for outputs in responses.values()):
        raise AssertionError("final outputs do not cover the frozen split exactly")

    samples: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in SYSTEM_ORDER
    }
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sampled_golds = [golds[index] for index in indices]
        for name in SYSTEM_ORDER:
            sampled_outputs = [responses[name][rows[index]["case_id"]] for index in indices]
            for metric, value in primary_metrics(
                evaluate_outputs(sampled_golds, sampled_outputs)
            ).items():
                if metric != "schema_validity_rate" and value is not None:
                    samples[name][metric].append(value)

    report = {}
    for name in SYSTEM_ORDER:
        point = primary_metrics(runs[name]["scores"])
        valid = runs[name]["scores"]["strict"]["schema_valid_count"]
        low, high = wilson_interval(valid, len(rows))
        report[name] = {
            "schema_validity_rate": {
                "point": point["schema_validity_rate"],
                "lower": low,
                "upper": high,
                "method": "95 percent Wilson interval",
            }
        }
        for metric in point.keys() - {"schema_validity_rate"}:
            values = samples[name][metric]
            report[name][metric] = {
                "point": point[metric],
                "lower": None if not values else percentile(values, 0.025),
                "upper": None if not values else percentile(values, 0.975),
                "method": f"paired percentile bootstrap, {BOOTSTRAP_ITERATIONS} resamples",
            }
    return report


def review_metrics(review: dict) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for judgment in review["judgments"]:
        grouped[judgment["system"]].append(judgment)
    if set(grouped) != set(SYSTEM_ORDER) or any(len(rows) != 10 for rows in grouped.values()):
        raise AssertionError("review must contain ten judgments for every system")

    def summarize(rows: list[dict]) -> dict:
        result = {"judgments": len(rows)}
        for field in ("factuality_pass", "completeness_pass", "combined_pass"):
            passes = sum(row[field] for row in rows)
            low, high = wilson_interval(passes, len(rows))
            result[field] = {
                "passes": passes,
                "rate": passes / len(rows),
                "lower": low,
                "upper": high,
                "method": "95 percent Wilson interval",
            }
        return result

    return {
        "overall": summarize(review["judgments"]),
        "by_system": {name: summarize(grouped[name]) for name in SYSTEM_ORDER},
    }


def load_and_verify() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        runs = {
            "internal_test": json.loads(archive.read("internal_test_results.json")),
            "external_transfer": json.loads(archive.read("external_transfer_results.json")),
        }
    rows = {
        "internal_test": load_jsonl(ROOT / "data/test_cases.jsonl"),
        "external_transfer": load_jsonl(ROOT / "data/external_civic_eval.jsonl"),
    }
    for split in runs:
        for name in SYSTEM_ORDER:
            recomputed = evaluate_outputs(
                [row["gold"] for row in rows[split]],
                [output["response"] for output in runs[split][name]["outputs"]],
            )
            if rounded(recomputed) != rounded(runs[split][name]["scores"]):
                raise AssertionError(f"saved final scores do not reproduce for {split}/{name}")
    return runs, rows


def number(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_table(title: str, intervals: dict[str, dict]) -> None:
    print(f"\n## {title}\n")
    print("| system | schema valid | domain F1 | issue F1 | missing-info F1 | halluc. rate |")
    print("|---|---|---|---|---|---|")
    for name in SYSTEM_ORDER:
        columns = []
        for metric in (
            "schema_validity_rate",
            "service_domain_macro_f1",
            "issue_type_macro_f1",
            "missing_information_macro_f1",
            "hallucinated_non_null_field_rate",
        ):
            item = intervals[name][metric]
            columns.append(
                f"{number(item['point'])} ({number(item['lower'])} to {number(item['upper'])})"
            )
        print("| " + " | ".join([name] + columns) + " |")


if __name__ == "__main__":
    final_runs, split_rows = load_and_verify()
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    report = {
        "confidence_level": 0.95,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "automatic": {
            split: confidence_intervals(split_rows[split], final_runs[split])
            for split in final_runs
        },
        "summary_review": review_metrics(review),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("recomputed all ten final system runs from raw responses")
    print_table("internal test, point estimate and 95 percent interval", report["automatic"]["internal_test"])
    print_table("external transfer, point estimate and 95 percent interval", report["automatic"]["external_transfer"])
    print(f"\nwrote {METRICS_PATH.relative_to(ROOT)}")
