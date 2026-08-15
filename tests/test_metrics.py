import json
import unittest
from copy import deepcopy

from src.check_examples import load_examples
from src.evaluate import evaluate_outputs
from src.report_final import percentile
from src.report_validation import categorize_failure, wilson_interval
from src.schema import (
    ISSUE_TYPE_LABELS,
    MISSING_INFORMATION_LABELS,
    SERVICE_DOMAIN_LABELS,
    URGENCY_LABELS,
)


class SharedMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_examples()[0]["gold"]

    def test_perfect_output(self):
        report = evaluate_outputs([self.gold], [json.dumps(self.gold)])
        self.assertEqual(report["evaluation_version"], "2.0")
        self.assertEqual(report["strict"]["schema_validity_rate"], 1.0)
        fields = report["strict"]["end_to_end_field_metrics"]
        self.assertAlmostEqual(fields["service_domain_macro_f1"], 1 / len(SERVICE_DOMAIN_LABELS))
        self.assertAlmostEqual(fields["issue_type_macro_f1"], 1 / len(ISSUE_TYPE_LABELS))
        self.assertAlmostEqual(fields["urgency_macro_f1"], 1 / len(URGENCY_LABELS))
        expected_missing = sum(
            label in self.gold["missing_information"] for label in MISSING_INFORMATION_LABELS
        ) / len(MISSING_INFORMATION_LABELS)
        self.assertAlmostEqual(fields["missing_information_macro_f1"], expected_missing)
        self.assertEqual(report["conditional_valid"]["denominator"], 1)

    def test_wrong_but_valid_output(self):
        wrong = deepcopy(self.gold)
        wrong.update(
            service_domain="other",
            issue_type="other",
            urgency="safety_critical",
            missing_information=["none"],
            location="Elsewhere",
            event_date_or_time="tomorrow",
            amount_inr=99,
            service_identifier="wrong-id",
        )
        report = evaluate_outputs([self.gold], [json.dumps(wrong)])
        self.assertEqual(report["strict"]["schema_validity_rate"], 1.0)
        self.assertEqual(
            set(report["strict"]["end_to_end_field_metrics"].values()), {0.0}
        )
        self.assertEqual(
            report["strict"]["exact_factual_field_mismatches"]["rate"], 1.0
        )

    def test_invalid_json_receives_no_semantic_score(self):
        report = evaluate_outputs([self.gold], ["not JSON"])
        self.assertEqual(report["strict"]["json_parse_count"], 0)
        self.assertEqual(
            set(report["strict"]["end_to_end_field_metrics"].values()), {0.0}
        )
        self.assertIsNone(report["conditional_valid"]["field_metrics"])
        self.assertIsNone(report["strict"]["exact_factual_field_mismatches"]["rate"])

    def test_schema_invalid_json_receives_no_semantic_score(self):
        invalid = deepcopy(self.gold)
        invalid["urgency"] = "urgent"
        report = evaluate_outputs([self.gold], [json.dumps(invalid)])
        self.assertEqual(report["strict"]["json_parse_count"], 1)
        self.assertEqual(report["strict"]["schema_valid_count"], 0)
        self.assertEqual(
            set(report["strict"]["end_to_end_field_metrics"].values()), {0.0}
        )

    def test_wrapped_json_is_only_scored_in_repaired_view(self):
        raw = f"```json\n{json.dumps(self.gold)}\n```"
        report = evaluate_outputs([self.gold], [raw])
        self.assertEqual(report["strict"]["schema_valid_count"], 0)
        self.assertEqual(report["repaired"]["repairable_json_rate"], 1.0)
        fields = report["repaired"]["end_to_end_field_metrics"]
        self.assertAlmostEqual(fields["service_domain_macro_f1"], 1 / len(SERVICE_DOMAIN_LABELS))
        self.assertAlmostEqual(fields["issue_type_macro_f1"], 1 / len(ISSUE_TYPE_LABELS))
        self.assertEqual(fields["location_normalized_match"], 1.0)

    def test_macro_f1_counts_a_valid_prediction_for_a_missing_gold_class(self):
        predictions = [self.gold, dict(self.gold, service_domain="other")]
        report = evaluate_outputs(
            [self.gold, deepcopy(self.gold)], [json.dumps(item) for item in predictions]
        )
        self.assertAlmostEqual(
            report["strict"]["end_to_end_field_metrics"]["service_domain_macro_f1"],
            (2 / 3) / len(SERVICE_DOMAIN_LABELS),
        )

    def test_macro_f1_returns_zero_for_classes_with_no_support(self):
        report = evaluate_outputs([self.gold], [json.dumps(self.gold)])
        self.assertEqual(
            report["strict"]["end_to_end_field_metrics"]["service_domain_macro_f1"],
            1 / len(SERVICE_DOMAIN_LABELS),
        )


class FailureReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_examples()[0]["gold"]

    def test_clean_output_has_no_failure_category(self):
        self.assertEqual(categorize_failure(self.gold, json.dumps(self.gold)), [])

    def test_categories_name_the_actual_problem(self):
        self.assertEqual(categorize_failure(self.gold, "not JSON"), ["invalid_json"])
        wrong = deepcopy(self.gold)
        wrong["urgency"] = "safety_critical"
        wrong["service_identifier"] = None
        wrong["amount_inr"] = 250
        self.assertEqual(
            categorize_failure(self.gold, json.dumps(wrong)),
            ["wrong_urgency", "invented_fact", "dropped_fact"],
        )

    def test_wilson_interval_brackets_a_perfect_rate(self):
        low, high = wilson_interval(50, 50)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1.0)
        self.assertGreater(low, 0.9)

    def test_bootstrap_percentile_interpolates_without_dependencies(self):
        self.assertEqual(percentile([0.0, 1.0], 0.5), 0.5)
        self.assertEqual(percentile([0.0, 0.5, 1.0], 0.0), 0.0)
        self.assertEqual(percentile([0.0, 0.5, 1.0], 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
