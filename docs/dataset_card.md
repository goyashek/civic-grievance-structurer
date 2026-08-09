# dataset card

## scope

CivicStruct uses fictional public-service complaints to compare rules,
prompting, retrieval-augmented few-shot prompting, and SmolLM3 QLoRA. The
first batch contains 70 manually written canonical cases. It is a training
pool only; it is not the final validation or test set.

## canonical and surface tables

Each canonical row in `data/canonical_cases.jsonl` has:

```text
case_id
split
source_type
service_domain
issue_type
facts
intentionally_missing_fields
urgency
clean_formal_summary
case_flags
```

`facts` contains `location`, `event_date_or_time`, `amount_inr`, and
`service_identifier`. `case_flags` marks hard negatives and cases that need a
second look during the manual audit.

Later surface rows will keep the canonical `case_id` and add:

```text
surface_id
split
source_type
style
complaint
gold
```

All surface variants from one canonical case stay in its assigned split. The
retrieval index will contain training rows only.

## provenance

The first 70 cases are manually written fictional cases with
`source_type=manual_canonical`. They use invented routes, account numbers,
ward names, buildings, and service locations. No private complaint text,
personal name, phone number, email address, or real case identifier is used.

The model-selection complaints in `data/model_selection_development.jsonl`
remain a separate development record. They are not copied into this batch and
will not be used as the final test set.

## planned split and surface scope

| split | canonical cases | planned surface rows | current canonical rows |
| --- | ---: | ---: | ---: |
| train | 120 | 600 | 70 |
| validation | 25 | 50 | 0 |
| test | 50 | 50 | 0 |

Training variants will cover formal English, informal English, short
mobile-style complaints, spelling noise, and a small manually checked Roman
script Hinglish stress slice. Validation will use two manually checked
styles. Test complaints will be independently written rather than copied from
training templates.

## annotation and audit rules

The shared contract is schema version 1.0 and the label definitions are in
`docs/annotation_guide.md`. Canonical cases are checked before surface
variants are written. The audit must confirm that summaries add no unsupported
facts, every validation and test row is inspected, at least 20 percent of
training rows are inspected, and no exact or high-similarity duplicate crosses
splits.

## limitations

The controlled core is synthetic, so results may not transfer to real civic
complaint language. The project will report any external transfer check
separately and will not add public text to training or retrieval without a
clear provenance and privacy review.
