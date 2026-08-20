# evaluation contract

Evaluation contract version 2.0 receives one schema-valid gold object and one
raw model string for each complaint. It keeps three score views separate:

- strict output accepts the entire raw string only when it parses as one JSON
  object and passes schema version 1.0;
- conditional output scores only strict schema-valid objects and records that
  denominator; and
- repaired output may unwrap one JSON object from a Markdown fence or
  surrounding text, but it does not alter keys, labels, types, or values.

The strict end-to-end view counts every invalid JSON or schema-invalid object
as wrong on every field. This prevents a malformed answer from receiving a
semantic score for the few values that happened to parse. The repaired view
is reported beside the strict view and never replaces it.

The repairable-JSON rate divides newly schema-valid repaired objects by all
raw outputs in the run.

Service domain, issue type, and urgency use macro-F1 over their complete frozen
taxonomies. Missing information uses the same rule over all eight labels. A
class with zero support in a sample contributes zero, so the estimand does not
change across bootstrap resamples. The report also includes normalized location
and date or time match, exact amount and service-identifier match, and the
support denominator for conditional scores.

The exact factual field mismatch rate uses location, date or time, amount, and
service identifier. Its numerator counts a predicted non-null value that does
not match the gold field. Location and date or time use the same normalized
comparison as their field scores; amount and service identifier use exact
match. Its denominator is the number of predicted non-null values across those
fields. The rate is `null` when that denominator is zero.

A separate post-hoc extraction report uses the same strict parsing and field
matching rules. It does not revise Evaluation v2. Fact precision divides
correct extracted non-null values by all predicted non-null values. Fact
recall divides the same correct count by all gold non-null values. Coverage
divides predictions on gold-present fields by all gold non-null values, so it
stays between zero and one even when a system fabricates values. Fact F1 is
`2 * correct / (predicted non-null + gold non-null)`. Precision is `null` when
the system predicts no facts. Recall and coverage are `null` only when the gold
set contains no facts; otherwise a system that predicts none receives zero
recall, coverage, and F1.

The deterministic factuality breakdown scores every fact field in every row.
It separates correct, omitted, fabricated, distorted or partly correct, and
normalization-only mismatches. It does not use a model-based judge. A mismatch
that remains after the defined normalization is placed in the distorted or
partly correct bucket because the automatic evaluator cannot reliably tell
which part of the value was right.

Summary quality uses a separate single-reviewer blinded rubric pass. Each
system uses the same frozen complaint sample, and all complaint-summary pairs
are shuffled with system names hidden during scoring. Factuality requires no
unsupported or contradictory facts. Completeness requires the core issue and
material facts to remain. An exact string match would penalize faithful
paraphrases, so it is not part of the automatic evaluator.
