"""Record verified historical CivicStruct metadata in a compact MLflow store."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "data/final_results"
MANIFEST_PATH = FINAL_DIR / "evaluation_manifest.json"
TRACKING_DIR = ROOT / "data/reproducibility_results/mlruns"
RESULT_FILES = {
    "internal_test": FINAL_DIR / "internal_test_results.json",
    "external_transfer": FINAL_DIR / "external_transfer_results.json",
}
INPUT_FILES = (
    ROOT / "data/dataset_manifest.json",
    ROOT / "data/test_cases.jsonl",
    ROOT / "data/external_civic_eval.jsonl",
    FINAL_DIR / "frozen_system_manifest.json",
    FINAL_DIR / "revision_training_metadata.json",
    FINAL_DIR / "ablation_results.json",
    FINAL_DIR / "final_metrics.json",
    FINAL_DIR / "pairwise_comparisons.json",
    FINAL_DIR / "factuality_breakdown.json",
    FINAL_DIR / "summary_review.json",
    FINAL_DIR / "environment.json",
    *RESULT_FILES.values(),
)
DVC_LOCK = ROOT / "dvc.lock"
FINAL_RUN_SOURCE_COMMIT = "3e91316"
EVALUATION_V2_COMMIT = "b99fbc28814f2f3dec02446257c1f10bd54c4b63"
ADAPTER_SHA256 = "63934999afe7905ba1441f334b45ec31dd26e44fab5a03fa3c8ff82d611618a5"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest() -> dict:
    missing = [relative(path) for path in (*INPUT_FILES, DVC_LOCK) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing reproducibility inputs: {', '.join(missing)}")

    system = load_json(FINAL_DIR / "frozen_system_manifest.json")
    training = load_json(FINAL_DIR / "revision_training_metadata.json")
    runs = {
        split: {method: record["mlflow_run_id"] for method, record in load_json(path).items()}
        for split, path in RESULT_FILES.items()
    }
    return {
        "evaluation_version": "2.0",
        "metadata_provenance": "reconstructed from saved artifacts after the frozen run",
        "source_commits": {
            "frozen_run_inputs": FINAL_RUN_SOURCE_COMMIT,
            "evaluation_v2": EVALUATION_V2_COMMIT,
            "metadata_backfill": git_commit(),
        },
        "model": {
            "name": system["model_name"],
            "revision": system["model_revision"],
            "adapter_uri": "local civicstruct_final_results.zip#final_adapter/adapter_model.safetensors",
            "adapter_sha256": ADAPTER_SHA256,
            "adapter_note": "The adapter is not required to regenerate frozen metrics.",
        },
        "dataset": {
            "version": system["dataset_version"],
            "schema_version": "1.0",
            "evaluation_contract_version": "2.0",
        },
        "configuration": {
            "decoding": system["decoding"],
            "retrieval": system["retrieval"],
            "lora": {
                "rank": training["lora"]["r"],
                "alpha": training["lora"]["lora_alpha"],
                "dropout": training["lora"]["lora_dropout"],
            },
            "training": {
                "epochs": training["epochs"],
                "learning_rate": training["configuration"]["learning_rate"],
                "batch_size": training["configuration"]["per_device_train_batch_size"],
                "gradient_accumulation_steps": training["configuration"]["gradient_accumulation_steps"],
                "seed": training["configuration"]["seed"],
                "max_sequence_length": training["configuration"]["max_length"],
                "quantization": "4-bit NF4, float16 compute",
            },
        },
        "historical_mlflow_run_ids": {
            "training": training["mlflow_run_id"],
            "frozen_predictions": runs,
        },
        "files": {relative(path): sha256(path) for path in INPUT_FILES},
        "dvc_lock_sha256": sha256(DVC_LOCK),
    }


def metrics() -> dict[str, float]:
    final_metrics = load_json(FINAL_DIR / "final_metrics.json")
    qlora = final_metrics["automatic"]["internal_test"]["qlora"]
    facts = load_json(FINAL_DIR / "factuality_breakdown.json")["splits"]["internal_test"]["qlora"]["breakdown"]["overall"]
    review = final_metrics["summary_review"]["by_system"]["qlora"]
    return {
        "schema_validity": qlora["schema_validity_rate"]["point"],
        "service_domain_macro_f1": qlora["service_domain_macro_f1"]["point"],
        "issue_type_macro_f1": qlora["issue_type_macro_f1"]["point"],
        "missing_information_macro_f1": qlora["missing_information_macro_f1"]["point"],
        "exact_factual_mismatch": qlora["exact_factual_field_mismatch_rate"]["point"],
        "fabrication_rate": facts["fabricated"] / sum(facts.values()),
        "summary_factuality": review["factuality_pass"]["rate"],
        "summary_completeness": review["completeness_pass"]["rate"],
    }


def log_backfill(manifest: dict) -> str:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    training = load_json(FINAL_DIR / "revision_training_metadata.json")
    manifest_hash = sha256(MANIFEST_PATH)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("civicstruct-metadata-backfill")
    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name("civicstruct-metadata-backfill")
    prior_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.record_type = 'historical_metadata_backfill'",
    )
    for prior in prior_runs:
        if prior.data.tags.get("manifest_sha256") == manifest_hash:
            return prior.info.run_id
    with mlflow.start_run(run_name="evaluation-v2-metadata-backfill") as run:
        mlflow.set_tags(
            {
                "record_type": "historical_metadata_backfill",
                "metric_impact": "none",
                "provenance": manifest["metadata_provenance"],
                "manifest_sha256": manifest_hash,
            }
        )
        mlflow.log_params(
            {
                "base_model": manifest["model"]["name"],
                "base_model_revision": manifest["model"]["revision"],
                "git_sha": manifest["source_commits"]["evaluation_v2"],
                "dataset_version": manifest["dataset"]["version"],
                "schema_version": manifest["dataset"]["schema_version"],
                "evaluator_version": manifest["dataset"]["evaluation_contract_version"],
                "prompt_version": "unversioned_frozen_notebook_prompt",
                "lora_rank": manifest["configuration"]["lora"]["rank"],
                "lora_alpha": manifest["configuration"]["lora"]["alpha"],
                "lora_dropout": manifest["configuration"]["lora"]["dropout"],
                "learning_rate": manifest["configuration"]["training"]["learning_rate"],
                "epochs": manifest["configuration"]["training"]["epochs"],
                "batch_size": manifest["configuration"]["training"]["batch_size"],
                "gradient_accumulation": manifest["configuration"]["training"]["gradient_accumulation_steps"],
                "seed": manifest["configuration"]["training"]["seed"],
                "quantization": manifest["configuration"]["training"]["quantization"],
                "max_sequence_length": manifest["configuration"]["training"]["max_sequence_length"],
                "decoding": json.dumps(manifest["configuration"]["decoding"], sort_keys=True),
            }
        )
        metrics_to_log = metrics()
        metrics_to_log.update(
            training_duration_seconds=training["training_seconds"],
            peak_gpu_memory_mb=training["peak_gpu_memory_mb"],
            training_loss=training["train_metrics"]["train_loss"],
        )
        mlflow.log_metrics(metrics_to_log)
        for path in (MANIFEST_PATH, *INPUT_FILES, DVC_LOCK):
            mlflow.log_artifact(str(path), artifact_path="reproducibility")
        run_id = run.info.run_id
    for prior in prior_runs:
        client.set_tag(prior.info.run_id, "superseded_by_metadata_backfill", run_id)
    return run_id


if __name__ == "__main__":
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    run_id = log_backfill(manifest)
    print(f"logged metadata backfill run {run_id}")
