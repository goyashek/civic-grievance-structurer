# CivicStruct

I am building a student NLP project that turns informal public-service complaints into validated structured records. The main experiment asks whether QLoRA fine-tuning provides a measurable improvement over simpler approaches on a controlled grievance dataset.

## Planned task

The input is one complaint in plain text. The output is JSON containing:

- service domain and issue type;
- facts stated in the complaint, such as location, time, amount, or service identifier;
- bounded urgency;
- important missing information;
- a short formal summary that adds no unsupported facts.

The planned comparison includes deterministic rules, zero-shot prompting, static few-shot prompting, retrieval-selected few-shot examples, and one QLoRA adapter. Evaluation will separate strict schema validity, end-to-end field scores, field scores among valid outputs, repairable JSON, hallucinated fields, summary faithfulness, latency, and training cost.

## Data boundary

The controlled benchmark uses fictional grievance cases with manual review. Canonical cases are split before surface wording is created so that related complaints cannot cross training, validation, and test sets. A small licensed public training slice adds real writing patterns, while a separate public transfer slice stays outside model selection. Retrieval uses training examples only and selects at most one row from each case.

Private grievance portals, real personal details, and automatic complaint submission are outside the project.

## Data construction

I first write the controlled canonical cases and their gold labels. These rows
define the facts, issue labels, urgency, and intentionally missing information.
The retained controlled surface set has one training complaint per canonical
case and two validation complaints per case. I removed four mechanical
training rewrites per case because they inflated the dataset from 120 to 600
rows without adding enough independent language variation.

The current training pool contains 120 controlled complaints and 40 manually
mapped, deidentified complaints derived from OpenCity's IChangeMyCity data.
Every public row is reduced to ward-level location, has direct identifiers and
street-level fields removed, and is labeled against schema version 1.0. The
20-row San Diego Get It Done slice is reserved for external transfer testing.
Raw public downloads are not stored in the repository. Source links, licenses,
hashes, transformations, and review counts are recorded in
[`data/public_data_manifest.json`](data/public_data_manifest.json).

The 50 final controlled test complaints were independently written, checked,
and frozen before model training. Their predictions remain unopened. Internal
test results and the narrower external transfer result will be reported
separately.

## Current state

The repository and working rules are initialized. The two model-selection notebooks contain the recovered executed Colab runs for Qwen, SmolLM3, and Phi. On the 40-case development bake-off, SmolLM3 fixed few-shot was selected as the base because it gave the strongest overall structured output with lower peak memory than the other strongest candidate. Those saved bake-off metrics used the original strict evaluator. The shared evaluator now keeps strict, conditional, and repaired scores separate, and its definitions are frozen in [`docs/evaluation_contract.md`](docs/evaluation_contract.md). The final test set is written and frozen, but no system has generated predictions for it.

## Recorded experiments

The saved responses, summaries, timestamps, and complete MLflow file store are in [`data/model_selection_results/`](data/model_selection_results/README.md). From the repository root, the saved experiment can be opened with:

```bash
mlflow ui --backend-store-uri data/model_selection_results/mlruns
```

The runnable notebooks are [`model_bakeoff.ipynb`](notebooks/model_bakeoff.ipynb), [`model_bakeoff_phi.ipynb`](notebooks/model_bakeoff_phi.ipynb), and the earlier smoke check [`model_basics.ipynb`](notebooks/model_basics.ipynb). These are development results, not final test-set quality claims.

## Scope

CivicStruct is an evaluation project, not a production grievance portal. It will not provide legal advice, make final department-routing decisions, or replace human review. The intended output is a small local demonstration backed by saved evaluation results.
