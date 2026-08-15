import json
import unittest
from copy import deepcopy

from src.check_examples import load_examples
from src.evaluate import FACT_FIELDS, factuality_breakdown


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


if __name__ == "__main__":
    unittest.main()
