# annotation guide

## purpose

This guide defines schema version 1.0 for public-service complaints.
Annotators record only facts stated in the complaint. They do not infer a
department, location, amount, date, or affected group from context.

Each row has a complaint and one `gold` object with these fields:

```text
service_domain
issue_type
location
event_date_or_time
amount_inr
service_identifier
urgency
missing_information
formal_summary
```

The repository's `src/schema.py` is the dependency-free implementation of this
contract. Pydantic is not needed while the output is a flat JSON object with
fixed labels and scalar values.

## surface-variant review

Canonical rows define the truth for controlled complaints. A surface row keeps
the canonical case ID and copies its checked gold object. A minimal record
looks like this:

```json
{
  "case_id": "canonical-071",
  "style": "informal_english",
  "complaint": "Route 31 did not show up at Lotus Gate stop at 8:15 Monday."
}
```

Do not count mechanical template rewrites as independent training evidence.
Label each row by how its wording was actually made, such as
`manual_surface`, `canonical_summary_surface`, or `template_surface`.

Review each validation variant and every independently written final-test row.
Review at least 20 percent of training variants. Reject or correct a variant
when it:

- adds a person, cause, amount, date, location, identifier, or promised action;
- removes a fact or intentional omission from the canonical case;
- changes the service domain, issue type, urgency, or missing-information list;
- uses spelling noise or Roman-script Hinglish that changes the meaning; or
- copies another complaint closely enough to weaken the split audit.

Strict JSON and schema validation catch useful format errors, but they cannot
prove semantic preservation. Human review remains part of the data contract;
the repository records full review for the validation and final-test rows.

## licensed public rows

Licensed public text may enter the training slice only after a source and
privacy audit. Discard rows containing names, phone numbers, email addresses,
URLs, account-like identifiers, or sensitive personal details. Do not retain
request IDs, exact addresses, coordinates, postcodes, or street-level source
fields. Generalize location to ward or community area, mark `exact_location`
as missing, and record a hash of the source row for traceability.

Public labels are project annotations. They are not official labels from the
source portal. Every retained public row requires manual review of the text,
field mapping, summary, and privacy transformation.

## scalar fields

Use `null` when a scalar fact is not stated. Do not use an empty string,
"unknown", or a guessed value.

| field | annotation rule |
| --- | --- |
| `location` | Copy the most useful stated place. Use `null` when the complaint gives no place that can help route the case. |
| `event_date_or_time` | Copy the stated date, time, or duration. Use `null` when no timing is given. |
| `amount_inr` | Store a stated rupee amount as a JSON number. Use `null` when no amount is stated. |
| `service_identifier` | Store the stated route, account, consumer number, application ID, portal, machine, or similar identifier. Use `null` when none is given and the case does not need one to distinguish the service. |
| `formal_summary` | Write one short neutral sentence using only complaint facts. Keep uncertainty such as "the resident disputes" when the complaint makes a claim rather than proving it. |

## normalization

- Trim leading and trailing whitespace and reduce repeated internal spaces.
- Keep proper names, route numbers, IDs, and hyphens as written. Do not
  geocode or replace a location with a guessed address.
- Keep relative time phrases such as `yesterday evening`, `since Monday`, and
  `this morning` when the complaint has no reference date. Convert an explicit
  unambiguous calendar date to `YYYY-MM-DD` only when a dataset reference date
  is available.
- Convert `Rs 3,200`, `₹3200`, and similar stated rupee values to `3200` in
  `amount_inr`. Do not calculate an amount from a bill rate or add a currency
  conversion.
- Preserve identifier case and punctuation after whitespace cleanup. Do not
  invent an ID when the complaint omits one.
- Order `missing_information` using the schema order below. Use only one
  occurrence of a label.

## service domain

Choose the service that the complaint is about, not a location mentioned in
passing.

| label | use it for | positive example | confusing negative |
| --- | --- | --- | --- |
| `public_transport` | buses, trains, stations, stops, ticket machines, and passenger information | a bus route does not arrive | a pothole outside a bus stop, which is `roads_and_streetlights` |
| `water_supply` | piped water, public taps, tankers, water meters, and water quality | a dry tap or unsafe drinking water | a blocked drain, which is `sanitation_and_waste` |
| `sanitation_and_waste` | garbage collection, bins, drains, sewage, and waste handling | missed collection or overflowing sewage | a leaking drinking-water pipe, which is `water_supply` |
| `roads_and_streetlights` | roads, footpaths, traffic signals, streetlights, and related public infrastructure | a pothole or failed signal | a broken station display, which is `public_transport` |
| `electricity` | power supply, meters, bills, transformers, and electrical hazards | a disputed electricity bill | a water bill, which is `water_supply` |
| `welfare_or_document_service` | certificates, applications, benefits, records, and civic document portals | a rejected certificate correction | a bus ticket problem, which is `public_transport` |
| `other` | a clearly civic service that does not fit the listed domains | a public library booking page | an unclear complaint that could fit a listed domain |

## issue type

Pick the main operational problem. Use the first applicable label in this
tie-break order when a complaint contains more than one symptom:

1. `safety_or_health_hazard` when an immediate danger or health risk is
   explicitly present.
2. `overcharging_or_payment_problem` when the central request is about a
   charge, payment, refund, or duplicate billing.
3. `staff_conduct` when a worker's behavior is the complaint.
4. `record_or_document_error` when a record, certificate, or application is
   wrong or rejected.
5. `delay_or_non_arrival` when a scheduled person, vehicle, delivery, or
   service did not arrive on time.
6. `service_outage_or_non_delivery` when an expected service is unavailable
   or not delivered, without a single scheduled arrival being the main issue.
7. `damaged_infrastructure` when a physical public asset is broken but no
   immediate danger is stated.
8. `other` only when none of the labels describes the main problem.

Examples:

- A bus that never reaches a stop is `delay_or_non_arrival`; a route with no
  service for several days is `service_outage_or_non_delivery`.
- A cracked display is `damaged_infrastructure`; a cracked display that is
  falling onto pedestrians is `safety_or_health_hazard`.
- A rude worker is `staff_conduct` even if the collection was also late, unless
  the complaint is mainly about the missed service.
- Anger, inconvenience, or a request for quick action is not by itself a
  safety hazard.

## urgency

- `safety_critical`: the complaint states an immediate safety or health risk,
  such as a live wire, open manhole, unsafe drinking water, or a signal that
  creates a collision risk.
- `time_sensitive`: the complaint describes a repeated or ongoing essential
  service failure, a stated deadline, or an imminent loss. Examples include a
  multi-day water outage and repeated missed waste collection.
- `routine`: all other complaints, including ordinary delays, billing
  disputes, document corrections, and rude conduct.

Do not use `safety_critical` because the writer sounds upset. Use the explicit
risk in the text.

## missing information

These labels describe information needed to route, verify, or act on this
specific complaint. Do not mark every field that happens to be absent.

| label | use it when | do not use it when |
| --- | --- | --- |
| `exact_location` | no usable place is stated for a location-dependent issue | the complaint gives a building, lane, stop, portal, or other actionable place |
| `date_or_time` | timing is needed to understand or investigate the event and none is stated | a current physical defect can be acted on without knowing when it began |
| `service_identifier` | a route, account, consumer number, application, machine, or named service is needed to distinguish the case and is absent | the location and complaint type already identify a general service and no identifier is needed for the annotation |
| `transaction_or_reference_id` | a payment, booking, or application issue has no reference that can be used to look it up | the complaint is not about a traceable transaction |
| `amount` | a billing or payment complaint does not state the amount involved | the complaint is not about money, or the amount is already stated |
| `supporting_evidence` | a disputed charge or similar claim needs a bill, receipt, photo, or other proof and none is supplied | the complaint already includes the needed evidence or the issue can be acted on directly |
| `affected_person_or_group` | a document, benefit, or service-impact complaint does not identify whose case or which group is affected and that scope is needed | the complaint already identifies the resident, children, passengers, or another affected group |
| `none` | no material detail is missing for the stated action | never combine it with another missing label |

Use this fixed order when several labels apply:

```text
exact_location
date_or_time
service_identifier
transaction_or_reference_id
amount
supporting_evidence
affected_person_or_group
none
```

## review and tie-breaking

1. Read the complaint once for the main problem and once for the facts.
2. Mark only facts that are written or directly quoted.
3. Apply the domain and issue tie-break rules before choosing urgency.
4. Check that every non-null scalar appears in the complaint in the same
   sense. A summary may shorten wording but cannot add a cause, diagnosis,
   person, amount, or promised action.
5. If two labels still seem possible, choose the narrower operational label
   and record the ambiguity for review instead of using `other`.

The checked examples in `data/annotation_examples.jsonl` cover all service
domains, all issue labels, the three urgency levels, missing-field cases, and
one simple Roman-script Hinglish complaint.
