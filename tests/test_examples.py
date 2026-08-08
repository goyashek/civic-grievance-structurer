import unittest
from copy import deepcopy

from src.check_examples import load_examples
from src.schema import SchemaError, validate_gold


class ExamplesContractTest(unittest.TestCase):
    def test_gold_examples_match_the_contract(self):
        rows = load_examples()
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["split"] for row in rows}, {"development"})

    def test_none_is_the_only_missing_label_when_present(self):
        gold = deepcopy(load_examples()[0]["gold"])
        gold["missing_information"] = ["none", "amount"]
        with self.assertRaises(SchemaError):
            validate_gold(gold)

    def test_unknown_label_is_rejected(self):
        gold = deepcopy(load_examples()[0]["gold"])
        gold["urgency"] = "urgent"
        with self.assertRaises(SchemaError):
            validate_gold(gold)

    def test_missing_labels_are_unique_and_ordered(self):
        gold = deepcopy(load_examples()[0]["gold"])
        gold["missing_information"] = ["amount", "exact_location"]
        with self.assertRaises(SchemaError):
            validate_gold(gold)


if __name__ == "__main__":
    unittest.main()
