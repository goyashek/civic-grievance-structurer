"""Verify the final run and add confidence intervals and review scores."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import CATEGORICAL_LABELS, FACT_FIELDS, evaluate_outputs
from .fact_extraction import fact_extraction_metrics
from .report_validation import wilson_interval


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data/final_results"
RUN_PATHS = {
    "internal_test": RESULTS_DIR / "internal_test_results.json",
    "external_transfer": RESULTS_DIR / "external_transfer_results.json",
}
REVIEW_PATH = RESULTS_DIR / "summary_review.json"
METRICS_PATH = RESULTS_DIR / "final_metrics.json"
V1_METRICS_PATH = RESULTS_DIR / "evaluation_v1_metrics.json"
PAIRWISE_PATH = RESULTS_DIR / "pairwise_comparisons.json"
FACTUALITY_PATH = RESULTS_DIR / "factuality_breakdown.json"
FACT_EXTRACTION_PATH = RESULTS_DIR / "fact_extraction_metrics.json"
SYSTEM_ORDER = (
    "deterministic_rules",
    "zero_shot",
    "static_few_shot",
    "retrieved_few_shot",
    "qlora",
)
PAIRWISE_COMPARISONS = (
    ("qlora", "retrieved_few_shot"),
    ("qlora", "deterministic_rules"),
    ("retrieved_few_shot", "static_few_shot"),
)
PAIRWISE_METRICS = (
    "schema_validity_rate",
    "service_domain_macro_f1",
    "issue_type_macro_f1",
    "missing_information_macro_f1",
    "exact_factual_field_mismatch_rate",
)
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 42
INTERVAL_METRICS = (
    "service_domain_macro_f1",
    "issue_type_macro_f1",
    "missing_information_macro_f1",
    "exact_factual_field_mismatch_rate",
)


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
        "exact_factual_field_mismatch_rate": strict["exact_factual_field_mismatches"]["rate"],
    }


def confidence_intervals(rows: list[dict], runs: dict[str, dict]) -> dict[str, dict]:
    """Use Wilson for validity and row bootstrap for semantic metrics."""

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
        for metric in INTERVAL_METRICS:
            values = samples[name][metric]
            report[name][metric] = {
                "point": point[metric],
                "lower": None if not values else percentile(values, 0.025),
                "upper": None if not values else percentile(values, 0.975),
                "method": f"percentile bootstrap, {BOOTSTRAP_ITERATIONS} resamples",
            }
    return report


def paired_bootstrap_comparisons(rows: list[dict], runs: dict[str, dict]) -> dict[str, dict]:
    """Estimate paired differences from the same complaint resamples."""

    golds = [row["gold"] for row in rows]
    responses = {
        name: {output["case_id"]: output["response"] for output in runs[name]["outputs"]}
        for name in SYSTEM_ORDER
    }
    samples = {
        pair: defaultdict(list)
        for pair in PAIRWISE_COMPARISONS
    }
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sampled_golds = [golds[index] for index in indices]
        scores = {}
        for name in {name for pair in PAIRWISE_COMPARISONS for name in pair}:
            sampled_outputs = [responses[name][rows[index]["case_id"]] for index in indices]
            scores[name] = primary_metrics(evaluate_outputs(sampled_golds, sampled_outputs))
        for pair in PAIRWISE_COMPARISONS:
            left, right = pair
            for metric in PAIRWISE_METRICS:
                left_value = scores[left][metric]
                right_value = scores[right][metric]
                if left_value is not None and right_value is not None:
                    samples[pair][metric].append(left_value - right_value)

    report = {}
    for left, right in PAIRWISE_COMPARISONS:
        left_point = primary_metrics(runs[left]["scores"])
        right_point = primary_metrics(runs[right]["scores"])
        differences = {}
        for metric in PAIRWISE_METRICS:
            point = None
            if left_point[metric] is not None and right_point[metric] is not None:
                point = left_point[metric] - right_point[metric]
            values = samples[(left, right)][metric]
            differences[metric] = {
                "point": point,
                "lower": None if not values else percentile(values, 0.025),
                "upper": None if not values else percentile(values, 0.975),
                "method": f"paired percentile bootstrap, {BOOTSTRAP_ITERATIONS} resamples",
            }
        report[f"{left}_minus_{right}"] = {
            "system_a": left,
            "system_b": right,
            "difference": differences,
        }
    return report


def factuality_reports(rows: list[dict], runs: dict[str, dict]) -> dict[str, dict]:
    """Build the deterministic fact error report from preserved responses."""

    golds = [row["gold"] for row in rows]
    report = {}
    for name in SYSTEM_ORDER:
        outputs = {output["case_id"]: output["response"] for output in runs[name]["outputs"]}
        score = evaluate_outputs(
            golds, [outputs[row["case_id"]] for row in rows]
        )
        report[name] = {
            "exact_factual_field_mismatches": score["strict"]["exact_factual_field_mismatches"],
            "breakdown": score["strict"]["factuality_breakdown"],
        }
    return report


def fact_extraction_reports(rows: list[dict], runs: dict[str, dict]) -> dict[str, dict]:
    """Build supplemental extraction metrics from preserved responses."""

    golds = [row["gold"] for row in rows]
    report = {}
    for name in SYSTEM_ORDER:
        outputs = {output["case_id"]: output["response"] for output in runs[name]["outputs"]}
        report[name] = fact_extraction_metrics(
            golds, [outputs[row["case_id"]] for row in rows]
        )
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
    runs = {
        split: json.loads(path.read_text(encoding="utf-8"))
        for split, path in RUN_PATHS.items()
    }
    rows = {
        "internal_test": load_jsonl(ROOT / "data/test_cases.jsonl"),
        "external_transfer": load_jsonl(ROOT / "data/external_civic_eval.jsonl"),
    }
    for split in runs:
        for name in SYSTEM_ORDER:
            saved = runs[split][name]["scores"]
            recomputed = evaluate_outputs(
                [row["gold"] for row in rows[split]],
                [output["response"] for output in runs[split][name]["outputs"]],
            )
            if saved["total_outputs"] != recomputed["total_outputs"]:
                raise AssertionError(f"saved output count changed for {split}/{name}")
            for field in ("json_parse_count", "schema_invalid_count", "schema_valid_count"):
                if saved["strict"][field] != recomputed["strict"][field]:
                    raise AssertionError(f"saved validity counts changed for {split}/{name}")
            runs[split][name]["scores_v1"] = saved
            runs[split][name]["scores"] = recomputed
    return runs, rows


def number(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_table(title: str, intervals: dict[str, dict]) -> None:
    print(f"\n## {title}\n")
    print("| system | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch |")
    print("|---|---|---|---|---|---|")
    for name in SYSTEM_ORDER:
        columns = []
        for metric in (
            "schema_validity_rate",
            "service_domain_macro_f1",
            "issue_type_macro_f1",
            "missing_information_macro_f1",
            "exact_factual_field_mismatch_rate",
        ):
            item = intervals[name][metric]
            columns.append(
                f"{number(item['point'])} ({number(item['lower'])} to {number(item['upper'])})"
            )
        print("| " + " | ".join([name] + columns) + " |")


if __name__ == "__main__":
    final_runs, split_rows = load_and_verify()
    if not V1_METRICS_PATH.exists():
        V1_METRICS_PATH.write_text(METRICS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    pairwise = {
        split: paired_bootstrap_comparisons(split_rows[split], final_runs[split])
        for split in final_runs
    }
    factuality = {
        split: factuality_reports(split_rows[split], final_runs[split])
        for split in final_runs
    }
    fact_extraction = {
        split: fact_extraction_reports(split_rows[split], final_runs[split])
        for split in final_runs
    }
    PAIRWISE_PATH.write_text(
        json.dumps(
            {
                "evaluation_version": "2.0",
                "confidence_level": 0.95,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "comparisons": pairwise,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    FACTUALITY_PATH.write_text(
        json.dumps(
            {
                "evaluation_version": "2.0",
                "comparison_view": "strict end-to-end",
                "fact_fields": FACT_FIELDS,
                "categories": (
                    "correct",
                    "omitted",
                    "fabricated",
                    "distorted_or_partially_correct",
                    "normalization_only_mismatch",
                ),
                "splits": factuality,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    FACT_EXTRACTION_PATH.write_text(
        json.dumps(
            {
                "metric_version": "fact_extraction_1.0",
                "evaluator_version": "2.0",
                "status": "post_hoc_supplemental",
                "evaluation_v2_changed": False,
                "comparison_view": "strict end-to-end",
                "fact_fields": FACT_FIELDS,
                "definitions": {
                    "precision": "correct extracted facts / predicted non-null facts",
                    "recall": "correct extracted facts / gold non-null facts",
                    "coverage": "predicted non-null facts on gold-present fields / gold non-null facts",
                    "f1": "2 * correct facts / (predicted non-null facts + gold non-null facts)",
                },
                "splits": fact_extraction,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "evaluation_version": "2.0",
        "confidence_level": 0.95,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "macro_f1_taxonomies": CATEGORICAL_LABELS,
        "pairwise_comparisons_file": PAIRWISE_PATH.name,
        "factuality_breakdown_file": FACTUALITY_PATH.name,
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
    print(f"wrote {FACT_EXTRACTION_PATH.relative_to(ROOT)}")
