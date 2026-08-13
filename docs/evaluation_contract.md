# evaluation contract

Evaluation contract version 1.0 receives one schema-valid gold object and one
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

Service domain, issue type, and urgency use macro-F1 over labels represented
in the gold slice. Missing information uses label-wise macro-F1 over labels
represented in the same slice. This avoids giving a perfect system a zero for
a label that has no gold support. The report also includes normalized location
and date or time match, exact amount and service-identifier match, and the
support denominator for conditional scores.

The hallucinated non-null field rate uses location, date or time, amount, and
service identifier. Its numerator counts a predicted non-null value that does
not match the gold field. Location and date or time use the same normalized
comparison as their field scores; amount and service identifier use exact
match. Its denominator is the number of predicted non-null values across those
fields. The rate is `null` when that denominator is zero.

Summary quality uses a separate single-reviewer blinded rubric pass. Each
system uses the same frozen complaint sample, and all complaint-summary pairs
are shuffled with system names hidden during scoring. Factuality requires no
unsupported or contradictory facts. Completeness requires the core issue and
material facts to remain. An exact string match would penalize faithful
paraphrases, so it is not part of the automatic evaluator.
