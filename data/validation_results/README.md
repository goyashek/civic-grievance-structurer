# validation results

These are the development results for the 50 validation complaints. At the time
of this run, nothing here touched the internal test set or the external transfer
slice, and both were still unopened.

The frozen final evaluation is now documented separately in
[`data/final_results/`](../final_results/README.md). This file stays focused on
validation because validation was used for system selection and the controlled
training-length revision.

Regenerate every number from the saved responses by running
`python3 -m src.report_validation` from the repository root. That script rescores
the raw model text with the shared evaluator and stops if a recomputed score
disagrees with the score that was saved during the run, so the tables below cannot
drift away from the predictions. It also writes `validation_failures.json`.

## the run

SmolLM3-3B at revision `a07cc9a04f16550a088caea529712d1d335b0ac1`, loaded in 4-bit
NF4 on a Kaggle Tesla T4. Greedy decoding for every prompted system. The retrieval
index is TF-IDF over training rows only, three demonstrations from unique canonical
cases, matched to the static few-shot demonstration budget.

The QLoRA adapter used rank 16, alpha 32, dropout 0.05 on seven projection modules,
which is 30,228,480 trainable parameters. One epoch over the 160 training rows of
`frozen_full_v2` is only 20 optimizer steps at batch size 1 with 8 gradient
accumulation steps. Training took 164 seconds and peaked at 1571 MB of GPU memory.
Loss on the last logged step was 0.210. The six-step smoke run trained, saved, and
reloaded the adapter before the full run, and its before-and-after outputs are in
`smoke_run.json`.

## strict scores

Field scores are end to end, so an unparsable or schema-invalid response scores
zero. The interval on schema validity is a 95 percent Wilson interval, and it is
wide enough that only the large gaps here mean anything.

| system | schema valid | domain F1 | issue F1 | urgency F1 | missing-info F1 | halluc. rate | prompt tokens | s per case |
|---|---|---|---|---|---|---|---|---|
| deterministic_rules | 1.00 (0.93 to 1.00) | 0.883 | 0.895 | 0.707 | 0.027 | n/a | 0 | 0.00 |
| zero_shot | 0.00 (0.00 to 0.07) | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 322 | 2.67 |
| static_few_shot | 0.74 (0.60 to 0.84) | 0.605 | 0.459 | 0.494 | 0.420 | 0.316 | 640 | 1.85 |
| retrieved_few_shot | 0.88 (0.76 to 0.94) | 0.856 | 0.870 | 0.725 | 0.884 | 0.138 | 662 | 1.97 |
| qlora | 0.98 (0.90 to 1.00) | 0.816 | 0.829 | 0.788 | 0.936 | 0.248 | 322 | 3.19 |

The hallucination rate counts predicted non-null facts that disagree with the gold
value, over the number of non-null facts the system predicted. It is not
applicable for the rule baseline and zero-shot because neither produced any
non-null fact to check.

## valid-output scores and repair

The conditional columns score only the responses that were already schema valid,
so each system has its own denominator and the columns are not comparable across
rows without reading that denominator. Repair unwraps one JSON object from
surrounding text and changes nothing else.

| system | conditional n | domain F1 | issue F1 | missing-info F1 | location | service id | repairable | repaired valid |
|---|---|---|---|---|---|---|---|---|
| deterministic_rules | 50 | 0.883 | 0.895 | 0.027 | 0.120 | 0.280 | 0.00 | 1.00 |
| zero_shot | 0 | n/a | n/a | n/a | n/a | n/a | 0.76 | 0.76 |
| static_few_shot | 37 | 0.701 | 0.647 | 0.866 | 0.622 | 0.486 | 0.00 | 0.74 |
| retrieved_few_shot | 44 | 0.908 | 0.924 | 0.933 | 0.864 | 0.705 | 0.02 | 0.90 |
| qlora | 49 | 0.826 | 0.842 | 1.000 | 0.816 | 0.490 | 0.00 | 0.98 |

## formal against informal wording

The validation split is 25 formal and 25 informal complaints written from the same
25 canonical cases, so this is a wording-sensitivity check on paired rows rather
than two independent samples.

| system | style | schema valid | domain F1 | issue F1 | missing-info F1 |
|---|---|---|---|---|---|
| deterministic_rules | formal_english | 1.00 | 0.900 | 0.907 | 0.027 |
| deterministic_rules | informal_english | 1.00 | 0.864 | 0.882 | 0.027 |
| zero_shot | formal_english | 0.00 | 0.000 | 0.000 | 0.000 |
| zero_shot | informal_english | 0.00 | 0.000 | 0.000 | 0.000 |
| static_few_shot | formal_english | 0.76 | 0.626 | 0.459 | 0.454 |
| static_few_shot | informal_english | 0.72 | 0.581 | 0.459 | 0.370 |
| retrieved_few_shot | formal_english | 0.88 | 0.862 | 0.887 | 0.903 |
| retrieved_few_shot | informal_english | 0.88 | 0.849 | 0.849 | 0.868 |
| qlora | formal_english | 1.00 | 0.829 | 0.841 | 1.000 |
| qlora | informal_english | 0.96 | 0.802 | 0.816 | 0.825 |

## failure categories

One response can fail in several ways, so these are field-level counts across the
50 rows. The last line counts responses with at least one problem. Per-row records
are in `validation_failures.json`.

| category | deterministic_rules | zero_shot | static_few_shot | retrieved_few_shot | qlora |
|---|---|---|---|---|---|
| dropped_fact | 50 | 0 | 7 | 3 | 14 |
| invalid_json | 0 | 50 | 0 | 2 | 0 |
| invented_fact | 0 | 0 | 8 | 3 | 2 |
| missing_information_mismatch | 44 | 0 | 2 | 3 | 0 |
| schema_invalid | 0 | 0 | 13 | 4 | 1 |
| wrong_fact_value | 0 | 0 | 15 | 10 | 16 |
| wrong_issue_type | 13 | 0 | 15 | 3 | 8 |
| wrong_service_domain | 7 | 0 | 10 | 4 | 8 |
| wrong_urgency | 13 | 0 | 16 | 10 | 10 |
| responses with any failure | 50 | 50 | 46 | 31 | 36 |

## what I read from this

Zero-shot prompting fails on format, not on understanding. No response was valid
JSON on the first pass, but unwrapping the fenced block recovers 38 of 50, and
those recovered outputs score 0.603 domain F1. The failure is decoration around
the object.

Fine-tuning bought format reliability and the missing-information field. QLoRA
reached 0.98 strict validity against 0.88 for retrieval, scored 0.936 on
missing-information F1, and did it with 322 prompt tokens instead of 662, because
it needs no demonstrations. Among its schema-valid responses it labeled
missing-information perfectly.

Retrieval is still better at reading the stated facts. It leads on service
identifier (0.705 against 0.490 conditional) and hallucinates less often (0.138
against 0.248). The adapter fills fact fields more eagerly than it should, which
shows up as 16 wrong fact values and 14 dropped facts.

The rule baseline is the uncomfortable result. Keyword rules reach 0.883 domain F1
and 0.895 issue F1, close to the best learned system, which says the controlled
complaints carry strong lexical cues for those two fields. The rules collapse to
0.027 on missing-information because they never reason about what is absent, and
their fact matches are inflated by agreeing with null: they emit no non-null facts
at all, so amount matches 0.92 only because most gold amounts are null. The honest
reading is that this benchmark is easy for domain and issue type, and the learned
systems earn their place on the fields that need reading rather than matching.

One epoch was 20 optimizer steps, so the adapter is probably undertrained. That is
the single change I want to test next, keeping everything else fixed.

Fifty rows from 25 canonical cases cannot separate close numbers. I am treating
the format and missing-information gaps as real and the few-point differences in
domain and issue F1 as noise.

## files

`validation_baselines.json` has the four non-trained systems with their responses,
latency, prompt tokens, and scores. `qlora_validation_predictions.json` has the
adapter run. `qlora_training_metadata.json` has the training record, package
versions, and LoRA settings. `smoke_run.json` has the wiring check and the adapter
reload comparison. `validation_failures.json` has the per-row failure records.

`mlruns/` is the MLflow file store for the six retained runs: four baselines, the
training run, and the adapter validation run. Open it with
`mlflow ui --backend-store-uri data/validation_results/mlruns`. The Kaggle session
logged the baseline cell three times while I was fixing later cells, so I kept the
pass that the saved JSON files reference and dropped the two identical earlier
passes and one aborted training run. Scores were not touched.

The trained adapter and the training checkpoints are not in the repository. The
adapter is 121 MB, and retraining it from the recorded settings takes under three
minutes on a T4.
