import json
import unittest
from copy import deepcopy

from src.check_examples import load_examples
from src.evaluate import evaluate_outputs


class SharedMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_examples()[0]["gold"]

    def test_perfect_output(self):
        report = evaluate_outputs([self.gold], [json.dumps(self.gold)])
        self.assertEqual(report["evaluation_version"], "1.0")
        self.assertEqual(report["strict"]["schema_validity_rate"], 1.0)
        self.assertEqual(
            set(report["strict"]["end_to_end_field_metrics"].values()), {1.0}
        )
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
            report["strict"]["hallucinated_non_null_fields"]["rate"], 1.0
        )

    def test_invalid_json_receives_no_semantic_score(self):
        report = evaluate_outputs([self.gold], ["not JSON"])
        self.assertEqual(report["strict"]["json_parse_count"], 0)
        self.assertEqual(
            set(report["strict"]["end_to_end_field_metrics"].values()), {0.0}
        )
        self.assertIsNone(report["conditional_valid"]["field_metrics"])
        self.assertIsNone(report["strict"]["hallucinated_non_null_fields"]["rate"])

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
        self.assertEqual(
            set(report["repaired"]["end_to_end_field_metrics"].values()), {1.0}
        )


if __name__ == "__main__":
    unittest.main()
