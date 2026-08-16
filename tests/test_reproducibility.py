import unittest

from src.backfill_mlflow import build_manifest
from src.report_final import load_and_verify


class FinalResultsReproducibilityTest(unittest.TestCase):
    def test_saved_raw_runs_cover_both_frozen_splits(self):
        runs, rows = load_and_verify()
        self.assertEqual(set(runs), {"internal_test", "external_transfer"})
        self.assertEqual(
            {name: len(value) for name, value in rows.items()},
            {"internal_test": 50, "external_transfer": 20},
        )

    def test_manifest_links_the_saved_runs_and_dvc_lock(self):
        manifest = build_manifest()
        self.assertEqual(manifest["evaluation_version"], "2.0")
        self.assertEqual(set(manifest["historical_mlflow_run_ids"]["frozen_predictions"]), {
            "internal_test",
            "external_transfer",
        })
        self.assertTrue(manifest["dvc_lock_sha256"])


if __name__ == "__main__":
    unittest.main()
