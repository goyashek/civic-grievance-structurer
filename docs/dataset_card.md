# dataset card

## scope

CivicStruct compares rules, prompting, retrieval-selected prompting, and
SmolLM3 QLoRA on public-service complaint structuring. The working data has a
controlled synthetic core, a small licensed public training slice, and a
separate public transfer slice. The final controlled test set is written,
audited, and frozen, but predictions remain unopened.

## files and current counts

| file | role | rows |
| --- | --- | ---: |
| `data/canonical_cases.jsonl` | controlled canonical train and validation cases | 145 |
| `data/surface_variants.jsonl` | controlled train and validation complaints | 170 |
| `data/public_training_examples.jsonl` | licensed public-derived training complaints | 40 |
| `data/external_civic_eval.jsonl` | licensed public-derived transfer complaints | 20 |
| `data/test_cases.jsonl` | independent controlled final-test complaints | 50 |

The controlled surfaces contain 120 training rows and 50 validation rows. The
effective training pool is the 120 controlled training rows plus 40 public
training rows, for 160 examples. Validation remains the 50 controlled rows.
The final test contains 50 independently written controlled rows.

## controlled data

Each canonical row has:

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

Each surface row keeps the canonical `case_id` and adds:

```text
surface_id
split
source_type
style
complaint
gold
```

The 145 canonical cases are manually written fictional cases. Their routes,
account numbers, ward names, buildings, and service locations are invented.
The retained controlled surfaces include 15 manually written complaints, 135
direct canonical-summary surfaces, and 20 simple validation templates.

An earlier version counted five mechanical training styles per case and
reported 600 training rows. Four styles per training case were removed because
they did not add enough independent wording to justify treating them as 480
extra examples. Rows previously described as assistant-drafted are now labeled
by their actual construction method. The correction is recorded in
`data/surface_generation_metadata.json`.

## licensed public training data

The 40 public training rows are derived from OpenCity's IChangeMyCity
complaints log. The source resource declares CC BY-SA 2.0 and credits
Janaagraha iCMyC, Vivek Mathew, and Haji Shariefullah. The raw download had
16,071 rows. No raw source file is committed.

The curation discarded detected names, phone numbers, emails, URLs,
account-like identifiers, unclear text, and sensitive personal details. It
also removed request IDs, exact addresses, coordinates, postcodes, and
street-level fields. Retained complaints use only a ward-level location and
therefore carry `missing_information=["exact_location"]`. Every retained row
was manually labeled and reviewed.

This slice covers electricity, roads and streetlights, sanitation and waste,
and water supply. It does not balance the full taxonomy and does not replace
the controlled cases that cover rarer domains and issue labels.

## external transfer data

The 20 external rows are derived from San Diego Get It Done requests. The
source links the Open Data Commons Public Domain Dedication and License. These
rows cover road, streetlight, drainage, and waste complaints. They stay outside training,
retrieval, prompt design, model selection, and the internal test set.

The same privacy transformation is applied: direct identifiers and source
request IDs are discarded, exact addresses and coordinates are not retained,
and location is generalized to community area. Every row was manually mapped
to the project schema. The resulting transfer score will be reported
separately because the slice has narrow domain coverage and comes from a
different civic system.

## split and leakage rules

All controlled surfaces from one canonical case stay in the same split.
Validation and internal test rows never enter retrieval. The public training
slice may enter QLoRA training and retrieval, while the external slice may not.
Retrieved prompts use unique case IDs.

The final controlled test complaints were written independently rather than
copied from the training templates. Test predictions remain closed until
the prompts, retriever, adapter, decoding settings, and evaluator are frozen.

## provenance and audit

`data/public_data_manifest.json` records source URLs, licenses, access date,
raw file hashes, transformations, row counts, and limitations. Each derived
public row stores only a SHA-256 hash of its source row for traceability. It
does not store the source request ID.

All rows pass schema checks. Controlled surfaces also pass canonical-gold,
split inheritance, exact duplicate, and cross-split near-duplicate checks. All
50 validation rows, all 120 retained controlled training rows, all 40 public
training rows, and all 20 external rows received manual review.

## limitations

The controlled core is synthetic and much cleaner than ordinary portal text.
The public training slice is small and covers only four domains. The external
slice is smaller and covers only two project domains. Public category mapping
and project labels are manual annotations, not official target fields supplied
by either source. Neither internal nor external results should be described as
performance on all real grievance portals.
