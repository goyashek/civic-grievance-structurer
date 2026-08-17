import unittest

from src.backfill_mlflow import metrics
from src.model_registry import evaluate_quality_gates


class ModelRegistryTests(unittest.TestCase):
    def test_frozen_qlora_passes_gates(self):
        self.assertTrue(evaluate_quality_gates(metrics())["overall_passed"])

    def test_gate_rejects_low_issue_f1(self):
        values = metrics()
        values["issue_type_macro_f1"] = 0.69
        self.assertFalse(evaluate_quality_gates(values)["overall_passed"])
