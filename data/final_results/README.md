# final results

This folder preserves the final frozen CivicStruct experiment. The original
Kaggle ZIP contains the adapter, raw responses, automatic scores, training
revision, data-size ablations, run metadata, and MLflow store. The Evaluation
v2 files correct the macro-F1 taxonomy, recompute uncertainty, and add paired
comparisons and a deterministic factuality breakdown without changing any
saved prediction.

Recompute all ten system evaluations from the checked raw responses and rebuild
the interval report from the repository root with:

```bash
python3 -m src.report_final
```

The two raw result files, frozen-system manifest, training metadata, ablation,
environment record, and Evaluation v2 manifest are small enough to stay in
Git. The local Kaggle ZIP still holds the adapter and is not needed for this
report command.

Schema validity and rubric pass rates use 95 percent Wilson intervals. The
field metrics and paired system differences use percentile bootstrap intervals
from 2,000 row resamples with seed 42. These intervals are uncertainty
estimates for the small frozen samples, not guarantees about performance on
other portals.

## internal controlled test

| system | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.929 to 1.000) | 0.883 (0.768 to 0.960) | 0.730 (0.598 to 0.819) | 0.010 (0.000 to 0.023) | n/a |
| zero-shot | 0.000 (0.000 to 0.071) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.720 (0.583 to 0.825) | 0.522 (0.366 to 0.626) | 0.397 (0.268 to 0.496) | 0.219 (0.116 to 0.250) | 0.570 (0.471 to 0.670) |
| retrieved few-shot | 0.820 (0.692 to 0.902) | 0.638 (0.487 to 0.754) | 0.638 (0.495 to 0.733) | 0.348 (0.209 to 0.437) | 0.378 (0.267 to 0.490) |
| QLoRA | 0.940 (0.838 to 0.979) | 0.829 (0.697 to 0.914) | 0.745 (0.572 to 0.854) | 0.670 (0.364 to 0.723) | 0.365 (0.279 to 0.450) |

QLoRA is the strongest learned system on schema validity, domain, issue type,
and missing information. Its exact fact mismatch point estimate is slightly lower
than retrieval's, and their intervals overlap. The keyword rules are
competitive on the controlled taxonomy but almost never identify missing
information. Zero-shot produced no strict schema-valid responses; 40 of its
50 responses became valid after the evaluator's narrow JSON unwrapping step.

## external transfer

| system | schema valid | domain F1 | issue F1 | missing-info F1 | fact mismatch |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.839 to 1.000) | 0.198 (0.108 to 0.261) | 0.163 (0.083 to 0.217) | 0.125 (0.125 to 0.125) | n/a |
| zero-shot | 0.000 (0.000 to 0.161) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.950 (0.764 to 0.991) | 0.186 (0.131 to 0.250) | 0.090 (0.036 to 0.146) | 0.000 (0.000 to 0.000) | 0.947 (0.883 to 1.000) |
| retrieved few-shot | 0.950 (0.764 to 0.991) | 0.220 (0.134 to 0.280) | 0.116 (0.048 to 0.174) | 0.089 (0.065 to 0.107) | 0.250 (0.091 to 0.400) |
| QLoRA | 1.000 (0.839 to 1.000) | 0.233 (0.139 to 0.286) | 0.156 (0.070 to 0.236) | 0.125 (0.125 to 0.125) | 0.125 (0.000 to 0.286) |

QLoRA transfers best among the learned systems on this 20-row slice. The
intervals are wide, and the slice covers only road, streetlight, drainage, and
waste complaints from one different civic system. It is a transfer check, not
evidence of general performance across real grievance portals.

## paired bootstrap differences on the internal test

Each difference uses the same 2,000 complaint resamples for both systems.
Positive values favor the first system in the comparison.

| comparison | domain F1 | issue F1 | missing-info F1 |
|---|---:|---:|---:|
| QLoRA minus retrieved | +0.190 (0.052 to 0.343) | +0.106 (-0.066 to 0.257) | +0.322 (0.024 to 0.418) |
| QLoRA minus rules | -0.054 (-0.192 to 0.071) | +0.015 (-0.164 to 0.189) | +0.660 (0.358 to 0.708) |
| retrieved minus static | +0.116 (-0.027 to 0.251) | +0.242 (0.122 to 0.361) | +0.128 (-0.013 to 0.254) |

The paired comparisons are stored in `pairwise_comparisons.json`. The full
fact-field counts are in `factuality_breakdown.json`.

## training revision and ablation

The final adapter uses two epochs over all 160 training rows. Its training
loss was 0.139, training took 327.3 seconds on a Kaggle T4, and peak allocated
GPU memory was about 1,586 MB. It trained 30,228,480 parameters, or 1.78
percent of the loaded model, and the saved adapter is about 121 MB.

| training rows | validation schema | domain F1 | issue F1 | missing-info F1 | training loss | seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.980 | 0.659 | 0.856 | 0.531 | 0.269 | 85.9 |
| 80 | 0.980 | 0.692 | 0.843 | 0.936 | 0.186 | 171.2 |
| 160 | 1.000 | 0.964 | 0.908 | 0.977 | 0.139 | 327.3 |

The ablation is a controlled training-size comparison, not a repeated trial.
It shows that missing-information F1 benefited most from additional examples;
the issue score was not monotonic between 40 and 80 rows.

## blinded summary rubric pass

Ten frozen internal complaints were sampled with seed 42. Every system was
scored on the same complaints, giving 50 shuffled judgments with system names
hidden during scoring. Factuality required no unsupported or contradictory
fact. Completeness required the core issue and material facts to remain.

| system | factuality | completeness | both pass |
|---|---:|---:|---:|
| deterministic rules | 10/10 | 10/10 | 10/10 |
| zero-shot | 9/10 | 10/10 | 9/10 |
| static few-shot | 10/10 | 10/10 | 10/10 |
| retrieved few-shot | 10/10 | 10/10 | 10/10 |
| QLoRA | 10/10 | 10/10 | 10/10 |
| overall | 49/50 (0.895 to 0.996) | 50/50 (0.929 to 1.000) | 49/50 (0.895 to 0.996) |

The only failed judgment was a zero-shot summary that described intermittent
low pressure as a service outage. This is a single-reviewer rubric pass, not a
multi-rater study, so there is no inter-rater reliability statistic. It checks
only ten complaints and should be read as a small qualitative audit.

## saved artifacts

- `civicstruct_final_results.zip`: untouched Kaggle result bundle;
- `evaluation_v1_metrics.json`: the pre-correction published metrics;
- `final_metrics.json`: Evaluation v2 point estimates and confidence intervals;
- `pairwise_comparisons.json`: paired bootstrap system differences;
- `factuality_breakdown.json`: deterministic fact-field error categories;
- `summary_review.json`: sampled cases, rubric, and all 50 judgments.
