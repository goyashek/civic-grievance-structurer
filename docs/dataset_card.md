# dataset card

## scope

CivicStruct uses fictional public-service complaints to compare rules,
prompting, retrieval-augmented few-shot prompting, and SmolLM3 QLoRA. The
current batch contains 120 training and 25 validation canonical cases. No
test canonical cases have been written yet.

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
retrieval index will contain training rows only and each prompt will use at
most one surface form from a canonical case.

## provenance

The current 145 canonical cases are manually written fictional cases with
`source_type=manual_canonical`. They use invented routes, account numbers,
ward names, buildings, and service locations. No private complaint text,
personal name, phone number, email address, or real case identifier is used.

The current surface file contains 35 manually written rows and 615 rows from a
constrained assistant-drafting pass. The drafting workflow receives one
canonical case and a target style, then returns only a complaint record in
structured JSON. The pipeline copies the gold output from the canonical case
instead of trusting the assistant to assign labels. The 615 generated rows are
marked `source_type=llm_assisted_surface`; the 35 starter rows remain marked
`manual_surface`.

The generation record is in `data/surface_generation_metadata.json` and the
review record is in `data/surface_audit_manifest.json`. The current audit read
all 50 validation rows and one formal variant for each of the 120 training
cases. No external API model revision was recorded for this constrained draft
pass, so that field must be replaced if a named ChatGPT or other provider run
is used for a future regeneration.

The model-selection complaints in `data/model_selection_development.jsonl`
remain a separate development record. They are not copied into this batch and
will not be used as the final test set.

## planned split and surface scope

| split | canonical cases | planned surface rows | current canonical rows |
| --- | ---: | ---: | ---: |
| train | 120 | 600 | 120 |
| validation | 25 | 50 | 25 |
| test | 50 | 50 | 0 |

Training variants will cover formal English, informal English, short
mobile-style complaints, spelling noise, and a small manually checked Roman
script Hinglish stress slice. Validation will use two manually checked
styles. Test complaints will be independently written rather than copied from
training templates.

The remaining training cases will increase support for rare
missing-information labels, record errors, hard negatives, and
ambiguous-but-defensible cases. The target is useful support rather than a
perfectly uniform label table. Any label with weak validation or test support
will stay marked as a limitation.

The repository currently contains 600 training surface rows and 50 validation
surface rows. The rows pass schema, canonical-gold, split, and duplicate
checks. The training and validation surface set is frozen as version 1 for the
next evaluation stage. The final test set is still unwritten.

## assisted variant workflow

The generation prompt asks the assistant to preserve every stated fact and
every intentional omission while changing only the wording and requested
style. The assistant may draft formal English, informal English, concise
English, spelling-noise English, or Roman-script Hinglish for training rows.
Validation rows use two styles.

The checks are deliberately separate:

- JSON parsing and the shared schema check catch formatting and label-contract
  errors.
- Canonical gold alignment checks that the structured target did not change.
- Fact and omission review checks that a variant did not invent, remove, or
  reinterpret information.
- Duplicate review checks that a new surface row is not just a copy of another
  row or a case from another split.

Strict JSON is only a format check. It is not evidence that the complaint
preserved its meaning. The final test cases will be independently written and
will not be generated as paraphrases of training cases.

## annotation and audit rules

The shared contract is schema version 1.0 and the label definitions are in
`docs/annotation_guide.md`. Canonical cases are checked before surface
variants are written. The audit must confirm that summaries add no unsupported
facts, every validation and test row is inspected, at least 20 percent of
training rows are inspected, and no exact or high-similarity duplicate crosses
splits. The evaluator will keep strict schema validity, end-to-end field
scores, valid-output-conditional scores, and repaired-output scores separate.
Summary review will use the same complaint sample for every system with hidden
system names and randomized order.

## limitations

The controlled core is synthetic, so results may not transfer to real civic
complaint language. The project will report any external transfer check
separately. A small 20 to 30 row public civic-text slice is optional when its
licence, field mapping, and privacy handling are clear. Public text will not
enter training or retrieval.
