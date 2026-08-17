"""Register the frozen QLoRA adapter after Evaluation v2 quality gates pass."""

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.backfill_mlflow import ADAPTER_SHA256, build_manifest, metrics


ROOT = Path(__file__).resolve().parents[1]
FINAL_ZIP = ROOT / "data/final_results/civicstruct_final_results.zip"
REGISTRY_DIR = ROOT / "data/model_registry"
TRACKING_DIR = REGISTRY_DIR / "mlruns"
ADAPTER_DIR = REGISTRY_DIR / "artifacts/qlora_final_adapter"
RECORD_PATH = REGISTRY_DIR / "registry_record.json"
REGISTRY_NAME = "civicstruct-qlora"
ALIAS = "champion"
GATES = {
    "schema_validity": (">=", 0.90),
    "issue_type_macro_f1": (">=", 0.70),
    "missing_information_macro_f1": (">=", 0.60),
    "exact_factual_mismatch": ("<=", 0.40),
    "fabrication_rate": ("<=", 0.05),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def evaluate_quality_gates(values: dict[str, float]) -> dict:
    checks = {}
    for name, (operator, threshold) in GATES.items():
        value = values[name]
        checks[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": value >= threshold if operator == ">=" else value <= threshold,
        }
    return {"overall_passed": all(item["passed"] for item in checks.values()), "checks": checks}


def extract_adapter() -> Path:
    model_path = ADAPTER_DIR / "adapter_model.safetensors"
    if model_path.exists():
        if sha256(model_path) != ADAPTER_SHA256:
            raise ValueError(f"unexpected adapter checksum at {model_path}")
        return ADAPTER_DIR
    if ADAPTER_DIR.exists():
        raise FileExistsError(f"incomplete adapter directory at {ADAPTER_DIR}")

    staging = ADAPTER_DIR.with_name("qlora_final_adapter_extracting")
    if staging.exists():
        raise FileExistsError(f"stale extraction directory at {staging}")
    try:
        with zipfile.ZipFile(FINAL_ZIP) as archive:
            members = [member for member in archive.infolist() if member.filename.startswith("final_adapter/") and not member.is_dir()]
            if not members:
                raise FileNotFoundError("final_adapter is missing from the frozen results ZIP")
            for member in members:
                destination = staging / Path(member.filename).relative_to("final_adapter")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        if sha256(staging / "adapter_model.safetensors") != ADAPTER_SHA256:
            raise ValueError("frozen adapter checksum does not match the recorded manifest")
        staging.replace(ADAPTER_DIR)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ADAPTER_DIR


def register() -> dict:
    gate_report = evaluate_quality_gates(metrics())
    if not gate_report["overall_passed"]:
        raise RuntimeError("frozen QLoRA did not pass the model quality gates")

    adapter_dir = extract_adapter()
    manifest = build_manifest()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_registry_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("civicstruct-model-governance")
    client = MlflowClient()
    try:
        client.get_registered_model(REGISTRY_NAME)
    except MlflowException:
        client.create_registered_model(REGISTRY_NAME)

    adapter_checksum = sha256(adapter_dir / "adapter_model.safetensors")
    matching_version = next(
        (
            version
            for version in client.search_model_versions(f"name='{REGISTRY_NAME}'")
            if version.tags.get("adapter_sha256") == adapter_checksum
        ),
        None,
    )
    if matching_version is None:
        with mlflow.start_run(run_name="frozen-qlora-registry") as run:
            registration_run_id = run.info.run_id
            mlflow.set_tags({"record_type": "model_registry", "evaluation_version": "2.0", "quality_gates": "passed"})
            mlflow.log_params(
                {
                    "base_model": manifest["model"]["name"],
                    "base_model_revision": manifest["model"]["revision"],
                    "dataset_version": manifest["dataset"]["version"],
                    "schema_version": manifest["dataset"]["schema_version"],
                    "evaluator_version": manifest["dataset"]["evaluation_contract_version"],
                    "frozen_training_run_id": manifest["historical_mlflow_run_ids"]["training"],
                    "adapter_sha256": adapter_checksum,
                }
            )
            mlflow.log_metrics(metrics())
            mlflow.log_text(json.dumps(gate_report, indent=2), "quality_gates.json")
        version = client.create_model_version(REGISTRY_NAME, adapter_dir.as_uri(), registration_run_id)
    else:
        version = matching_version
        registration_run_id = version.run_id
    version_number = str(version.version)
    version_tags = {
        "adapter_sha256": adapter_checksum,
        "git_sha": git_sha(),
        "dvc_lock_sha256": manifest["dvc_lock_sha256"],
        "dataset_version": manifest["dataset"]["version"],
        "schema_version": manifest["dataset"]["schema_version"],
        "evaluator_version": manifest["dataset"]["evaluation_contract_version"],
        "base_model_revision": manifest["model"]["revision"],
        "frozen_training_run_id": manifest["historical_mlflow_run_ids"]["training"],
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(REGISTRY_NAME, version_number, key, value)
    client.set_registered_model_tag(REGISTRY_NAME, "task", "civic-grievance-structuring")
    client.set_registered_model_alias(REGISTRY_NAME, ALIAS, version_number)

    record = {
        "registered_model": REGISTRY_NAME,
        "version": version_number,
        "alias": ALIAS,
        "status": "champion",
        "source": "data/model_registry/artifacts/qlora_final_adapter",
        "registration_run_id": registration_run_id,
        "frozen_training_run_id": manifest["historical_mlflow_run_ids"]["training"],
        "adapter_sha256": adapter_checksum,
        "git_sha": git_sha(),
        "dvc_lock_sha256": manifest["dvc_lock_sha256"],
        "dataset_version": manifest["dataset"]["version"],
        "schema_version": manifest["dataset"]["schema_version"],
        "evaluator_version": manifest["dataset"]["evaluation_contract_version"],
        "base_model": manifest["model"]["name"],
        "base_model_revision": manifest["model"]["revision"],
        "quality_gates": gate_report,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(register(), indent=2))
