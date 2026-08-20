# Release checklist

Checked on 2026-08-20. Evaluation v2, the adapter, prompts, data, decoding
settings, and saved predictions remain unchanged.

## Claim lineage

| claim or artifact | saved evidence |
|---|---|
| Internal and 20-row San Diego Evaluation v2 metrics | `data/final_results/final_metrics.json`, raw result JSON, and `data/final_results/README.md` |
| Paired comparisons and factuality breakdown | `data/final_results/pairwise_comparisons.json` and `data/final_results/factuality_breakdown.json` |
| Validation results and failure categories | `data/validation_results/README.md`, saved predictions, and `src/report_validation.py` |
| Three-seed training-size ablation | `data/ablation/ablation_results.json`, `data/ablation/run_summary.json`, and the runnable ablation notebook |
| 240-row San Diego stress test | `data/supplemental_results/civicstruct_san_diego_external_benchmark.zip` |
| Registered champion model | `data/model_registry/registry_record.json` and the saved MLflow metadata |
| Hosted serving path | `deploy/huggingface_space/`, the public Space, and the README screenshot |

## Checks

- [x] 33 focused unit tests pass.
- [x] Validation and final reports rebuild from saved responses and evaluator v2.
- [x] Canonical, surface, public-data, test-case, and dataset-freeze checks pass.
- [x] All notebook JSON and executable code cells parse after removing Kaggle
  install and shell magic lines.
- [x] Source compilation, CLI help, and demo help pass.
- [x] API failure-path and fake-model HTTP checks pass through the focused tests.
- [x] All nine nested ablation runs rescore from their saved validation responses.
- [x] The hosted Space returns HTTP 200 and the README contains its checked
  screenshot.
- [x] Gradio construction was checked in the serving pass.
- [x] The Dockerfile now includes the pinned CUDA inference stack and keeps the
  adapter as a read-only runtime mount.
- [ ] Run the documented `/structure` check on a Linux x86-64 NVIDIA host.

The public hosted Space remains the checked deployment path. The Dockerfile
build check passes without warnings, but this arm64 host has no NVIDIA GPU, so
the heavy container request remains a manual release check rather than a
claimed local result.

## Two-minute explanation

CivicStruct turns an informal civic complaint into one validated JSON record
with a service domain, issue type, urgency, extracted facts, missing
information, and a formal summary. The reason for using structured output is
simple: a useful complaint system needs to preserve what the resident said and
make the result easy for a downstream workflow to inspect.

I compared deterministic rules, zero-shot prompting, static few-shot prompting,
retrieved few-shot prompting, and QLoRA on a frozen evaluation split. An early
dataset version inflated the training count with mechanical rewrites, so I
removed those rows and kept 120 controlled plus 40 deidentified public training
complaints. The final test stayed untouched until the prompt, retriever, model,
and evaluator were frozen.

On the 50-complaint internal test, QLoRA reached 0.940 strict schema validity,
0.829 domain F1, 0.745 issue F1, and 0.670 missing-information F1. It was the
strongest learned system and much better than retrieval on missing information,
but deterministic rules still reached 0.883 domain F1. The adapter also had a
0.365 exact factual-field mismatch rate, so valid JSON does not mean that every
extracted fact is safe.

The final artifact is tied to saved predictions, evaluator v2, DVC inputs,
MLflow metadata, a registry quality gate, and one shared inference module used
by the CLI, FastAPI service, and Gradio demo. The public Space runs the frozen
adapter on ZeroGPU. It is a review tool, not an automatic case-routing system;
accepted outputs still need human review.
