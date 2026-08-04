import unittest

from src.check_examples import load_examples


class ExamplesContractTest(unittest.TestCase):
    def test_gold_examples_match_the_contract(self):
        rows = load_examples()
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["split"] for row in rows}, {"development"})


if __name__ == "__main__":
    unittest.main()
