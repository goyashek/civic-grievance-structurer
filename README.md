# CivicStruct

I am building a student NLP project that turns informal public-service complaints into validated structured records. The main experiment asks whether QLoRA fine-tuning provides a measurable improvement over simpler approaches on a controlled grievance dataset.

## Planned task

The input is one complaint in plain text. The output is JSON containing:

- service domain and issue type;
- facts stated in the complaint, such as location, time, amount, or service identifier;
- bounded urgency;
- important missing information;
- a short formal summary that adds no unsupported facts.

The planned comparison includes deterministic rules, zero-shot prompting, static few-shot prompting, retrieval-selected few-shot examples, and one QLoRA adapter. Model quality will be judged using schema validity, field-level metrics, hallucinated fields, summary faithfulness, latency, and training cost.

## Data boundary

The project will use fictional, controlled grievance cases with manual review. Canonical cases will be split before paraphrases are created so that variants of the same complaint cannot cross training, validation, and test sets. Retrieval will use training examples only.

Private grievance portals, real personal details, and automatic complaint submission are outside the project.

## Current state

The repository and working rules are initialized. Dataset construction, baseline implementation, model selection, training, evaluation, and the local demo have not started, so there are no model results to report yet.

## Scope

CivicStruct is an evaluation project, not a production grievance portal. It will not provide legal advice, make final department-routing decisions, or replace human review. The intended output is a small local demonstration backed by saved evaluation results.
