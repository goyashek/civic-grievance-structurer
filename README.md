# CivicStruct

CivicStruct is my attempt to answer a narrow NLP question properly: can a
small language model turn an informal civic complaint into reliable structured
JSON, and does QLoRA fine-tuning earn its complexity over rules and prompting?

The answer is mixed in a useful way. The final QLoRA model is the strongest
learned system on the controlled test set, with 0.94 strict schema validity and
0.765 missing-information F1. Simple rules still beat it on service-domain F1,
and the adapter still gets too many stated fact values wrong. That tension is
the project. I wanted an evaluation where a model could not hide weak factual
extraction behind a valid-looking JSON object.

| item | final choice |
|---|---|
| task | complaint text to validated structured JSON |
| base model | `HuggingFaceTB/SmolLM3-3B` |
| adaptation | 4-bit QLoRA, 30.2M trainable parameters |
| training data | 120 controlled and 40 licensed public complaints |
| evaluation | 50 validation, 50 frozen internal test, 20 San Diego transfer rows, 60-row Baton Rouge source-aligned diagnostic |
| comparison | rules, zero-shot, static few-shot, retrieved few-shot, QLoRA |
| experiment runtime | Kaggle Tesla T4 |
| current state | experiment and metric closeout complete; local demo still pending |

## What the model produces

The input is one public-service complaint:

> Bus 724 did not arrive near Nehru Place yesterday evening. I waited for
> almost an hour and do not know where to report it.

The target is one schema-valid JSON object:

```json
{
  "service_domain": "public_transport",
  "issue_type": "delay_or_non_arrival",
  "location": "near Nehru Place",
  "event_date_or_time": "yesterday evening",
  "amount_inr": null,
  "service_identifier": "bus 724",
  "urgency": "routine",
  "missing_information": ["exact_location"],
  "formal_summary": "Bus 724 did not arrive near Nehru Place yesterday evening, causing a wait of almost one hour."
}
```

Schema version 1.0 has seven service domains, eight issue types, three urgency
levels, four nullable fact fields, eight missing-information labels, and one
grounded formal summary. The contract is deliberately closed. A plausible new
label is still wrong if it is outside the agreed taxonomy.

This task has three separate failure modes:

- the model may understand the complaint but wrap the JSON in extra text;
- it may return valid JSON with an invalid label or missing key;
- it may satisfy the schema while inventing, dropping, or changing a fact.

The evaluator keeps those failures separate instead of reducing them to one
accuracy number.

## Experiment design

Every system receives the same complaint and eventually meets the same schema
validator.

```text
                              +-> deterministic rules ---------+
                              +-> zero-shot SmolLM3 -----------+
complaint -> frozen prompt ---+-> static three-shot SmolLM3 ---+-> raw response
                              +-> retrieved three-shot SmolLM3 +
                              +-> QLoRA SmolLM3 ----------------+

raw response -> strict JSON parse -> schema validation -> field metrics
            -> narrow JSON unwrap -> repaired metrics
            -> fact comparison    -> exact factual field mismatch rate
```

The five systems answer different questions:

| system | reason for including it |
|---|---|
| deterministic rules | tests how far obvious lexical cues can go without a language model |
| zero-shot | measures the untuned model with no examples |
| static few-shot | measures the value of three fixed audited demonstrations |
| retrieved few-shot | selects three training-only demonstrations with TF-IDF and unique case IDs |
| QLoRA | tests whether parameter-efficient training improves format and semantic extraction |

Static and retrieved prompting use the same demonstration count and roughly
the same prompt budget. QLoRA uses no demonstrations at inference, which cuts
its prompt length roughly in half. All model systems use greedy decoding.

## How the project changed while I built it

The work did not move in a straight line. Several early assumptions failed,
and fixing them changed the project more than adding another model would have.

### I started with the output contract

The first model smoke test used five fictional complaints. Qwen produced
JSON-shaped responses, but none passed the shared schema. That result made the
main problem clear: structured generation needs a precise contract before it
needs training code.

I wrote the schema and annotation guide first. The guide defines label
boundaries, tie-breaking, null handling, urgency, missing information, and the
rule that summaries may reorganize complaint facts but may not add them. The
schema stays in the Python standard library because the output is a flat
object and a larger validation dependency would add little here.

### I compared base models before fine-tuning

I ran a 40-case development bake-off across Qwen3-4B, SmolLM3-3B, and
Phi-4-mini. Every zero-shot run failed strict schema validity. Fixed few-shot
prompting made the comparison useful.

| base model, fixed few-shot | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch | peak T4 memory |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.175 | 0.268 | 0.200 | 0.302 | 0.176 | 5,771 MB |
| SmolLM3-3B | 0.425 | 0.483 | 0.221 | 0.563 | 0.056 | 2,668 MB |
| Phi-4-mini | 0.400 | 0.485 | 0.199 | 0.508 | 0.143 | 4,107 MB |

SmolLM3 had the best balance of schema validity, issue classification,
missing-information extraction, exact fact mismatch rate, and memory. Phi was a
little better on domain F1, but not enough to outweigh the rest. I selected
one base model here and did not fine-tune the full shortlist.

These development scores came from the earlier strict evaluator and were used
only for model selection. They are not final test claims.

### I retracted an inflated dataset

An early version counted five controlled wording styles for every training
case and reported 600 controlled training rows. Four of those five styles were
mechanical rewrites. They increased row count without giving 480 genuinely
independent complaints.

I removed those rows, retracted the earlier freeze, and kept one controlled
training surface per canonical case. The corrected training pool has 120
controlled complaints and 40 licensed public complaints. This made the
dataset smaller, but the claim became defensible.

The public-data pass needed the same restraint. Two automatic curation drafts
retained street details, unclear text, and incorrect labels, so I discarded
them. The final public training slice contains 40 manually corrected and
deidentified IChangeMyCity complaints. A separate 20-row San Diego slice never
enters training, retrieval, prompt design, or model selection.

### I froze evaluation before opening the test set

Validation was used for model choice, one controlled training-length revision,
and the final configuration. The dataset hashes, model revision, prompts,
retriever, decoding settings, adapter settings, and evaluator were then saved
in a frozen manifest. Only after that did the final notebook fetch the internal
test and external transfer files.

Test predictions were generated once. Their results did not trigger another
prompt, data, threshold, or model change.

## Data design

| split | rows | construction | role |
|---|---:|---|---|
| controlled training | 120 | one checked surface per fictional canonical case | QLoRA and retrieval |
| licensed public training | 40 | manually mapped and deidentified IChangeMyCity rows | QLoRA and retrieval |
| validation | 50 | formal and informal surfaces from 25 canonical cases | model and prompt choices |
| internal test | 50 | independently written fictional complaints | one final controlled evaluation |
| external transfer | 20 | deidentified San Diego Get It Done rows | separate transfer check |

The split unit is the canonical `case_id`, not the surface sentence. All
wording variants from one case stay in one split. Retrieval indexes training
rows only and may select at most one example from a canonical case. Exact and
cross-split near-duplicate checks run before freezing.

The controlled cases cover all seven domains, all eight issue labels, and all
three urgency levels. Coverage is deliberate, but I did not force every label
to have the same count. Rare labels remain rare, and the report does not treat
their scores as stable estimates.

For the public slices, I removed direct identifiers, request IDs, exact
addresses, coordinates, postcodes, and street-level fields. Locations are
reduced to ward or community area. The repository stores source-row hashes for
traceability but not the raw public downloads. Source URLs, licenses,
transformations, hashes, and review counts are in
[`data/public_data_manifest.json`](data/public_data_manifest.json).

The complete data contract is documented in the
[`dataset card`](docs/dataset_card.md) and
[`annotation guide`](docs/annotation_guide.md). The frozen counts and file
hashes are in [`data/dataset_manifest.json`](data/dataset_manifest.json).

## QLoRA configuration

The final adapter fine-tunes a pinned SmolLM3 revision in 4-bit NF4. Training
uses completion-only loss so prompt tokens do not contribute to the target
loss.

| setting | value |
|---|---|
| base revision | `a07cc9a04f16550a088caea529712d1d335b0ac1` |
| training rows | 160 |
| epochs | 2 |
| LoRA rank and alpha | 16 and 32 |
| LoRA dropout | 0.05 |
| target modules | query, key, value, output, gate, up, and down projections |
| learning rate | `2e-4`, cosine schedule |
| effective batch | batch size 1 with 8 accumulation steps |
| maximum sequence length | 768 tokens |
| trainable parameters | 30,228,480, about 1.78 percent |
| decoding | greedy, up to 256 new tokens |
| seed | 42 |
| device | Kaggle Tesla T4 |

The two-epoch run took 327.3 seconds, reached a recorded training loss of
0.139, and peaked at about 1,586 MB of allocated GPU memory. The saved adapter
is about 121 MB. A six-step smoke run first proved that the adapter could be
trained, saved, reloaded, and used for generation.

I changed one principal factor after the first full validation run: training
length increased from one epoch to two. Rank, alpha, learning rate, data,
prompt, and decoding stayed fixed.

| QLoRA validation run | schema valid | domain F1 | issue F1 | missing-info F1 | training seconds |
|---|---:|---:|---:|---:|---:|
| one epoch | 0.980 | 0.816 | 0.829 | 0.936 | 163.9 |
| two epochs | 1.000 | 0.964 | 0.908 | 0.977 | 327.3 |

## Evaluation contract

The primary table uses strict end-to-end scoring. If a raw response is invalid
JSON or violates schema version 1.0, every field receives zero for that row.
This makes format reliability part of the task instead of scoring only the
easy subset that parsed successfully.

The report also keeps two diagnostic views:

- conditional metrics score only already-valid outputs and show their
  denominator;
- repaired metrics may unwrap one complete JSON object from a code fence or
  surrounding text, but never alter a key, label, type, or value.

Service domain, issue type, and urgency use macro-F1 over the complete frozen
taxonomy, including labels with zero support in a resample. Missing information
uses the same rule over all eight labels. Location and time use normalized
exact matching; amount and service identifier use exact matching.

The exact factual field mismatch rate checks location, time, amount, and service
identifier. It divides unsupported or mismatched predicted facts by all
non-null facts the system predicted. The rate is `n/a` when a system predicts
no non-null facts. The separate factuality breakdown counts correct, omitted,
fabricated, distorted or partly correct, and normalization-only cases by
field.

Schema-validity and rubric pass rates use 95 percent Wilson intervals. The
primary semantic metrics use percentile bootstrap intervals from 2,000 row
resamples with seed 42. Paired system differences are reported separately.
See the full
[`evaluation contract`](docs/evaluation_contract.md) for the exact rules.

## Final controlled test results

These are strict end-to-end results on 50 independently written complaints.
Parentheses contain 95 percent intervals.

| system | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.929 to 1.000) | 0.883 (0.768 to 0.960) | 0.730 (0.598 to 0.819) | 0.010 (0.000 to 0.023) | n/a |
| zero-shot | 0.000 (0.000 to 0.071) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.720 (0.583 to 0.825) | 0.522 (0.366 to 0.626) | 0.397 (0.268 to 0.496) | 0.219 (0.116 to 0.250) | 0.570 (0.471 to 0.670) |
| retrieved few-shot | 0.820 (0.692 to 0.902) | 0.638 (0.487 to 0.754) | 0.638 (0.495 to 0.733) | 0.348 (0.209 to 0.437) | 0.378 (0.267 to 0.490) |
| QLoRA | 0.940 (0.838 to 0.979) | 0.829 (0.697 to 0.914) | 0.745 (0.572 to 0.854) | 0.670 (0.364 to 0.723) | 0.365 (0.279 to 0.450) |

QLoRA is the strongest learned system across the main structured fields. It
also uses about 328 prompt tokens per complaint, compared with 657 for
retrieved few-shot prompting. Its mean generation time was 3.59 seconds per
case on the recorded T4 run.

The rules result prevents an easy victory story. Rules score 0.883 on domain,
above QLoRA's 0.829, and reach 1.00 schema validity by construction. They score
only 0.011 on missing information and emit no non-null facts. The controlled
complaints contain strong lexical cues for routing, while deciding what an
investigator still needs requires more than keyword matching.

Zero-shot is mainly a formatting failure. None of its 50 raw responses passes
strict validation, but the narrow JSON unwrap recovers 40. This is why repaired
output is useful as a diagnostic and dangerous as the headline score.

QLoRA is still not fact-safe. Its 0.365 exact factual field mismatch rate means
46 of 126 predicted non-null facts do not match the gold value under the
evaluator. The adapter solves schema reliability much better than exact factual
extraction.

The primary automatic metric table, raw-response checks, confidence intervals,
and machine-readable scores are in the
[`final results report`](data/final_results/README.md).

## External transfer results

The 20-row San Diego slice is reported separately. It covers road,
streetlight, drainage, and waste complaints from one civic system and is too
small for broad deployment claims.

| system | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.839 to 1.000) | 0.198 (0.108 to 0.261) | 0.163 (0.083 to 0.217) | 0.125 (0.125 to 0.125) | n/a |
| zero-shot | 0.000 (0.000 to 0.161) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.950 (0.764 to 0.991) | 0.186 (0.131 to 0.250) | 0.090 (0.036 to 0.146) | 0.000 (0.000 to 0.000) | 0.947 (0.883 to 1.000) |
| retrieved few-shot | 0.950 (0.764 to 0.991) | 0.220 (0.134 to 0.280) | 0.116 (0.048 to 0.174) | 0.089 (0.065 to 0.107) | 0.250 (0.091 to 0.400) |
| QLoRA | 1.000 (0.839 to 1.000) | 0.233 (0.139 to 0.286) | 0.156 (0.070 to 0.236) | 0.125 (0.125 to 0.125) | 0.125 (0.000 to 0.286) |

QLoRA transfers best among the learned systems on this slice, especially for
fact fields. The 1.00 missing-information score needs context: every external
row has generalized location and the same `exact_location` omission. It does
not show broad missing-information reasoning across all eight labels.

## Cross-city source-aligned diagnostic

The executed external-validation notebook also queried the official Baton
Rouge 311 API and kept 60 safe comments: ten each from garbage, recycling,
drainage, sewer, road maintenance, and street or traffic categories. The
source category was mapped only to the project's broad service-domain labels.

| system | strict schema valid | service-domain agreement, end to end |
|---|---:|---:|
| deterministic rules | 1.000 | 0.150 |
| zero-shot | 0.000 | 0.000 |
| static few-shot | 0.967 | 0.367 |
| retrieved few-shot | 0.933 | 0.350 |
| QLoRA | 0.983 | 0.500 |

QLoRA produced 59 valid records and matched the broad source category on 30
of 60 rows. This is a source-aligned diagnostic, not a full gold evaluation:
the public rows do not contain audited labels for issue type, urgency, missing
information, or summary faithfulness. It is evidence that the frozen system
keeps its output format on a different civic system, while semantic transfer
is still limited.

## Data-size ablation

The ablation keeps the two-epoch recipe fixed and changes only the number of
training rows. Subsets are selected by canonical case group.

| training rows | validation schema | domain F1 | issue F1 | missing-info F1 | loss | seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.980 | 0.659 | 0.856 | 0.531 | 0.269 | 85.9 |
| 80 | 0.980 | 0.692 | 0.843 | 0.936 | 0.186 | 171.2 |
| 160 | 1.000 | 0.964 | 0.908 | 0.977 | 0.139 | 327.3 |

Missing-information F1 benefits most from additional examples. Issue F1 is
not monotonic between 40 and 80 rows, which is a reminder that this is one run
per size, not a repeated learning-curve study.

## Summary rubric pass

The review sampled ten internal complaints with seed 42 and scored every
system on the same complaints. System names were hidden and the 50
complaint-summary pairs were shuffled before scoring. Factuality required no
unsupported or contradictory fact; completeness required the core issue and
material facts to remain.

| system | factuality | completeness | both pass |
|---|---:|---:|---:|
| deterministic rules | 10/10 | 10/10 | 10/10 |
| zero-shot | 9/10 | 10/10 | 9/10 |
| static few-shot | 10/10 | 10/10 | 10/10 |
| retrieved few-shot | 10/10 | 10/10 | 10/10 |
| QLoRA | 10/10 | 10/10 | 10/10 |
| overall | 49/50 | 50/50 | 49/50 |

The only failed judgment was a zero-shot summary that changed intermittent low
pressure into a service outage. This is a single-reviewer qualitative check on
ten complaints, not a multi-rater study. It has no inter-rater reliability
statistic and should not be read as a 98 percent population estimate of
summary quality.

## What I learned

### Format reliability is part of model quality

Zero-shot SmolLM3 often understood enough to produce a useful object, but it
wrapped that object in Markdown or extra text. If an application needs machine
readable output, recovering JSON afterward is operationally different from
receiving valid JSON in the first place. Strict and repaired scores answer
different questions.

### A strong rules baseline makes the model claim more honest

The rules baseline exposed how much of domain and issue classification could
be solved from obvious words in a clean synthetic benchmark. QLoRA earns its
place on missing information, fact extraction, and adaptable structured
generation, not by crushing every simple baseline.

### Retrieval and fine-tuning fail differently

On validation, retrieval had a lower exact fact mismatch rate than the one-epoch
adapter. On the final test, QLoRA's exact mismatch point estimate is slightly lower than
retrieval's, and their intervals overlap. Retrieval uses more context and is
cheaper to change. QLoRA is more reliable about schema and missing information
with a shorter prompt. Neither removes the need to validate output.

### Data quality mattered more than the nominal row count

Removing 480 weak rewrites reduced the training count but improved the meaning
of every later comparison. The ablation also suggests that genuine additional
examples help missing-information learning. Repeated template rows and new
cases are not interchangeable.

### Held-out evaluation changed the tone of the result

The selected two-epoch adapter reached 0.977 missing-information F1 on
validation and 0.765 on the internal test. That gap is the reason the README
leads with final test results and keeps validation in the experiment record.
The model improved, but the 50-row development split was still optimistic for
some fields.

## Reproducing the saved results

The automatic evaluator and report scripts use the Python standard library.
From the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.report_validation
python3 -m src.report_final
```

The final report command opens the preserved Kaggle ZIP, recomputes all ten
internal and external system runs from raw responses, checks them against the
saved scores, and rebuilds `data/final_results/final_metrics.json`.

Run the cheap data checks with:

```bash
python3 src/check_canonical_cases.py
python3 src/check_surface_variants.py
python3 src/check_public_data.py
python3 src/check_test_cases.py
python3 src/check_dataset_freeze.py
```

The GPU workflow stays visible in notebooks rather than behind a training CLI.
[`final_kaggle_run.ipynb`](notebooks/final_kaggle_run.ipynb) contains its own
install cell, fetches the pinned repository inputs, trains the revision and
ablations, freezes the system, and only then loads held-out data. The untouched
executed copy is
[`final_kaggle_run_executed.ipynb`](notebooks/final_kaggle_run_executed.ipynb).

The cross-city handoff is
[`final_kaggle_external_validation.ipynb`](notebooks/final_kaggle_external_validation.ipynb),
with its executed copy in
[`final_kaggle_external_validation_executed.ipynb`](notebooks/final_kaggle_external_validation_executed.ipynb).
It reruns the frozen final recipe and then performs the separate Baton Rouge
source-aligned diagnostic with Kaggle Internet enabled.

The full Kaggle ZIPs contain the roughly 121 MB adapter, so GitHub's 100 MB
file limit makes them unsuitable for normal Git tracking. They are preserved
locally under `data/final_results/` and ignored by Git. The preserved files
have these SHA-256 hashes:

```text
civicstruct_final_results_kaggle.zip
85f42be61bde8235a4c6d133c825685a15359b2110afcf30dd68aafcdf285527
civicstruct_final_external_validation.zip
771d2859cad977677a55ef4367dad81e2cd6fdb482887dd7c6a058a3a054df36
```

The two saved development MLflow stores can be opened with:

```bash
mlflow ui --backend-store-uri data/model_selection_results/mlruns
mlflow ui --backend-store-uri data/validation_results/mlruns
```

## Repository map

| path | contents |
|---|---|
| [`src/schema.py`](src/schema.py) | schema version 1.0 and dependency-free validation |
| [`src/evaluate.py`](src/evaluate.py) | strict, conditional, repaired, field, and factuality metrics |
| [`src/report_validation.py`](src/report_validation.py) | reproducible validation tables and failure categories |
| [`src/report_final.py`](src/report_final.py) | final score verification and confidence intervals |
| [`docs/annotation_guide.md`](docs/annotation_guide.md) | label definitions and annotation decisions |
| [`docs/dataset_card.md`](docs/dataset_card.md) | data sources, privacy transformations, splits, and limits |
| [`docs/evaluation_contract.md`](docs/evaluation_contract.md) | frozen metric definitions |
| [`data/model_selection_results/`](data/model_selection_results/README.md) | six model-selection runs and MLflow records |
| [`data/validation_results/`](data/validation_results/README.md) | baseline outputs, QLoRA run, failure audit, and MLflow records |
| [`data/final_results/`](data/final_results/README.md) | frozen final run, adapter, raw outputs, intervals, and summary judgments |
| [`notebooks/`](notebooks/) | model smoke test, bake-off, validation training, and final Kaggle runs |

## Limitations

- The controlled benchmark is synthetic and cleaner than ordinary grievance
  text. Strong rule performance confirms that some labels have obvious lexical
  cues.
- The public training set has 40 rows from four domains. The external transfer
  set has 20 rows from one civic system and covers only two project domains.
- Public labels are manual project mappings, not official target labels from
  the source portals.
- Confidence intervals are wide. There is one training run per ablation size,
  one final test run, and one summary reviewer.
- The exact factual field mismatch metric is strict about extracted values. QLoRA still
  needs downstream validation and should not route or submit complaints without
  review.
- Roman-script Hinglish appears only as limited annotation coverage, not a
  large audited robustness slice.
- CivicStruct does not provide legal advice, choose the final department, or
  replace a caseworker. No private complaint portal data belongs in this
  repository.

## Project status

The research deliverable is complete: frozen data, model selection, QLoRA
training and reload, one controlled revision, data-size ablation, untouched
internal test, external transfer check, confidence intervals, summary rubric,
raw predictions, notebooks, and MLflow records are preserved.

The next work is engineering rather than another model experiment: a small
local complaint-to-JSON demo, focused CPU checks for that inference path, and
release verification. The evaluated adapter, prompts, decoding settings, and
metrics remain frozen.
