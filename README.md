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

The project will use fictional, controlled grievance cases with manual review. Canonical cases will be split before paraphrases are created so that variants of the same complaint cannot cross training, validation, and test sets. Retrieval will use training examples only and will select at most one surface variant from each canonical case.

Private grievance portals, real personal details, and automatic complaint submission are outside the project.

## Data construction

I use a two-layer synthetic dataset. I write the canonical cases and their gold
labels first. These rows define the facts, issue labels, urgency, and
intentionally missing information. They are the source of truth.

For training and validation surface variants, I use a reasoning-capable
language model as a constrained wording assistant. The prompt provides one
canonical case and requests a formal, informal, concise, spelling-noise, or
Roman-script Hinglish complaint in structured JSON. The assistant drafts the
complaint wording, not the labels. The pipeline copies the gold output from
the canonical case and validates the result against schema version 1.0.

Strict JSON helps catch formatting errors, but it does not prove that the
wording preserved the facts. The builder checks schema validity, canonical gold
alignment, split inheritance, and exact or near-duplicate leakage. I reviewed
all 50 validation rows and one formal variant for each of the 120 training
cases. The current surface file has 650 rows: 35 manually written rows and 615
assistant-drafted rows. The generation record and audit manifest contain the
prompt version, method, source counts, file hashes, and review status.

The final test complaints will be independently written rather than copied
from training variants. Results will describe performance on this controlled
synthetic benchmark, not on real civic complaint portals.

## Current state

The repository and working rules are initialized. The two model-selection notebooks contain the recovered executed Colab runs for Qwen, SmolLM3, and Phi. On the 40-case development bake-off, SmolLM3 fixed few-shot was selected as the base because it gave the strongest overall structured output with lower peak memory than the other strongest candidate. Those saved bake-off metrics used the original strict evaluator, so the final evaluator will report format failure and semantic quality separately. The final test set has not been opened.

## Recorded experiments

The saved responses, summaries, timestamps, and complete MLflow file store are in [`data/model_selection_results/`](data/model_selection_results/README.md). From the repository root, the saved experiment can be opened with:

```bash
mlflow ui --backend-store-uri data/model_selection_results/mlruns
```

The runnable notebooks are [`model_bakeoff.ipynb`](notebooks/model_bakeoff.ipynb), [`model_bakeoff_phi.ipynb`](notebooks/model_bakeoff_phi.ipynb), and the earlier smoke check [`model_basics.ipynb`](notebooks/model_basics.ipynb). These are development results, not final test-set quality claims.

## Scope

CivicStruct is an evaluation project, not a production grievance portal. It will not provide legal advice, make final department-routing decisions, or replace human review. The intended output is a small local demonstration backed by saved evaluation results.
