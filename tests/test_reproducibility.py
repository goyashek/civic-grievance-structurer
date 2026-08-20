import json
import unittest
from pathlib import Path

from src.backfill_mlflow import build_manifest
from src.evaluate import evaluate_outputs
from src.report_final import fact_extraction_reports, load_and_verify


ROOT = Path(__file__).resolve().parents[1]


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

    def test_saved_fact_extraction_metrics_reproduce(self):
        artifact = json.loads(
            (ROOT / "data/final_results/fact_extraction_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        runs, rows = load_and_verify()
        self.assertFalse(artifact["evaluation_v2_changed"])
        self.assertEqual(
            artifact["splits"],
            {
                split: fact_extraction_reports(rows[split], runs[split])
                for split in runs
            },
        )

    def test_nested_ablation_reproduces_all_nine_validation_runs(self):
        artifact = json.loads(
            (ROOT / "data/ablation/ablation_results.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (ROOT / "data/ablation/run_summary.json").read_text(encoding="utf-8")
        )
        validation_golds = {
            row["case_id"]: row["gold"]
            for line in (ROOT / "data/surface_variants.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if (row := json.loads(line))["split"] == "validation"
        }

        self.assertEqual(summary["summary"], artifact["summary"])
        self.assertEqual(artifact["seeds"], [42, 43, 44])
        self.assertEqual(artifact["row_counts"], [40, 80, 160])
        self.assertFalse(artifact["test_loaded"])
        self.assertEqual(len(artifact["runs"]), 9)
        for run in artifact["runs"]:
            outputs = run["validation"]["outputs"]
            golds = [validation_golds[output["case_id"]] for output in outputs]
            self.assertEqual(
                evaluate_outputs(golds, [output["response"] for output in outputs]),
                run["validation"]["scores"],
            )

        for seed in artifact["seeds"]:
            runs = sorted(
                (run for run in artifact["runs"] if run["seed"] == seed),
                key=lambda run: run["rows"],
            )
            self.assertEqual([run["rows"] for run in runs], [40, 80, 160])
            self.assertEqual(runs[1]["case_ids"][:40], runs[0]["case_ids"])
            self.assertEqual(runs[2]["case_ids"][:80], runs[1]["case_ids"])


if __name__ == "__main__":
    unittest.main()
