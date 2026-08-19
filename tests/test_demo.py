import unittest

from src.demo import review_complaint
from src.inference import InferenceResult


class FakeInference:
    def __init__(self, result):
        self.result = result

    def infer(self, complaint):
        return self.result


VALID_PREDICTION = {
    "service_domain": "water_supply",
    "issue_type": "service_outage_or_non_delivery",
    "location": "Ward 4",
    "event_date_or_time": "Monday morning",
    "amount_inr": None,
    "service_identifier": None,
    "urgency": "routine",
    "missing_information": ["exact_location"],
    "formal_summary": "Water service has been unavailable in Ward 4 since Monday morning.",
}


class DemoContractTests(unittest.TestCase):
    def test_success_exposes_prediction_summary_status_and_raw_response(self):
        result = InferenceResult(True, VALID_PREDICTION, '{"ok": true}', None, 0.12)
        prediction, summary, status, raw = review_complaint("No water since Monday.", FakeInference(result))
        self.assertEqual(prediction, VALID_PREDICTION)
        self.assertEqual(summary, VALID_PREDICTION["formal_summary"])
        self.assertIn("validation passed", status)
        self.assertEqual(raw, '{"ok": true}')

    def test_invalid_model_output_is_visible(self):
        result = InferenceResult(False, None, "not json", {"type": "invalid_json", "message": "bad JSON"}, 0.2)
        prediction, summary, status, raw = review_complaint("No water since Monday.", FakeInference(result))
        self.assertIsNone(prediction)
        self.assertEqual(summary, "")
        self.assertIn("invalid_json", status)
        self.assertEqual(raw, "not json")

    def test_blank_and_oversized_input_fail_before_model_load(self):
        prediction, _, status, _ = review_complaint("   ")
        self.assertIsNone(prediction)
        self.assertIn("invalid_input", status)
        prediction, _, status, _ = review_complaint("x" * 4001)
        self.assertIsNone(prediction)
        self.assertIn("4000", status)


if __name__ == "__main__":
    unittest.main()
