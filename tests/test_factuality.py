import json
import unittest
from copy import deepcopy

from src.check_examples import load_examples
from src.evaluate import FACT_FIELDS, factuality_breakdown
from src.fact_extraction import fact_extraction_metrics


class FactualityBreakdownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_examples()[0]["gold"]

    def test_breakdown_separates_omissions_and_normalization(self):
        prediction = deepcopy(self.gold)
        prediction["location"] = "  " + self.gold["location"].upper() + "  "
        prediction["event_date_or_time"] = None
        prediction["service_identifier"] = "made-up"
        result = factuality_breakdown([self.gold], [json.dumps(prediction)])
        self.assertEqual(result["by_field"]["location"]["normalization_only_mismatch"], 1)
        self.assertEqual(result["by_field"]["event_date_or_time"]["omitted"], 1)
        self.assertEqual(
            result["by_field"]["service_identifier"]["distorted_or_partially_correct"], 1
        )

    def test_invalid_output_omits_present_facts_and_keeps_nulls_correct(self):
        result = factuality_breakdown([self.gold], ["not JSON"])
        expected_omitted = sum(self.gold[field] is not None for field in FACT_FIELDS)
        self.assertEqual(result["overall"]["omitted"], expected_omitted)
        self.assertEqual(result["by_field"]["location"]["omitted"], 1)
        self.assertEqual(result["by_field"]["amount_inr"]["correct"], 1)

    def test_extraction_metrics_penalize_omissions_and_fabrications(self):
        prediction = deepcopy(self.gold)
        prediction["location"] = "  " + self.gold["location"].upper() + "  "
        prediction["event_date_or_time"] = None
        prediction["amount_inr"] = 99
        prediction["service_identifier"] = "made-up"
        result = fact_extraction_metrics([self.gold], [json.dumps(prediction)])
        self.assertEqual(result["correct_count"], 1)
        self.assertEqual(result["predicted_non_null_count"], 3)
        self.assertEqual(result["gold_non_null_count"], 3)
        self.assertEqual(result["covered_gold_non_null_count"], 2)
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 1 / 3)

        invalid = fact_extraction_metrics([self.gold], ["not JSON"])
        self.assertIsNone(invalid["precision"])
        self.assertEqual(invalid["recall"], 0.0)
        self.assertEqual(invalid["coverage"], 0.0)
        self.assertEqual(invalid["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
