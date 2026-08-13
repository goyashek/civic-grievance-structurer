# final results

This folder preserves the final frozen CivicStruct experiment. The original
Kaggle ZIP contains the adapter, raw responses, automatic scores, training
revision, data-size ablations, run metadata, and MLflow store. The separate
metrics file adds confidence intervals and a blinded summary rubric pass
without changing any saved prediction or point estimate.

Recompute all ten system evaluations from the raw responses and rebuild the
interval report from the repository root with:

```bash
python3 -m src.report_final
```

Schema validity and rubric pass rates use 95 percent Wilson intervals. The
field metrics use paired percentile bootstrap intervals from 2,000 row
resamples with seed 42. These intervals are uncertainty estimates for the
small frozen samples, not guarantees about performance on other portals.

## internal controlled test

| system | schema valid | domain F1 | issue F1 | missing-info F1 | halluc. rate |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.929 to 1.000) | 0.883 (0.769 to 0.960) | 0.730 (0.604 to 0.839) | 0.011 (0.000 to 0.030) | n/a |
| zero-shot | 0.000 (0.000 to 0.071) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.720 (0.583 to 0.825) | 0.522 (0.367 to 0.626) | 0.397 (0.271 to 0.515) | 0.251 (0.159 to 0.394) | 0.570 (0.471 to 0.670) |
| retrieved few-shot | 0.820 (0.692 to 0.902) | 0.638 (0.488 to 0.754) | 0.638 (0.497 to 0.778) | 0.397 (0.263 to 0.636) | 0.378 (0.267 to 0.490) |
| QLoRA | 0.940 (0.838 to 0.979) | 0.829 (0.699 to 0.914) | 0.745 (0.582 to 0.867) | 0.765 (0.576 to 0.890) | 0.365 (0.279 to 0.450) |

QLoRA is the strongest learned system on schema validity, domain, issue type,
and missing information. Its hallucination point estimate is slightly lower
than retrieval's, and their intervals overlap. The keyword rules are
competitive on the controlled taxonomy but almost never identify missing
information. Zero-shot produced no strict schema-valid responses; 40 of its
50 responses became valid after the evaluator's narrow JSON unwrapping step.

## external transfer

| system | schema valid | domain F1 | issue F1 | missing-info F1 | halluc. rate |
|---|---|---|---|---|---|
| deterministic rules | 1.000 (0.839 to 1.000) | 0.693 (0.379 to 0.917) | 0.433 (0.222 to 0.581) | 1.000 (1.000 to 1.000) | n/a |
| zero-shot | 0.000 (0.000 to 0.161) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | n/a |
| static few-shot | 0.950 (0.764 to 0.991) | 0.651 (0.459 to 0.883) | 0.241 (0.095 to 0.389) | 0.000 (0.000 to 0.000) | 0.947 (0.883 to 1.000) |
| retrieved few-shot | 0.950 (0.764 to 0.991) | 0.768 (0.469 to 0.984) | 0.309 (0.133 to 0.465) | 0.710 (0.519 to 0.857) | 0.250 (0.091 to 0.400) |
| QLoRA | 1.000 (0.839 to 1.000) | 0.816 (0.486 to 1.000) | 0.417 (0.187 to 0.630) | 1.000 (1.000 to 1.000) | 0.125 (0.000 to 0.286) |

QLoRA transfers best among the learned systems on this 20-row slice. The
intervals are wide, and the slice covers only road, streetlight, drainage, and
waste complaints from one different civic system. It is a transfer check, not
evidence of general performance across real grievance portals.

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
- `final_metrics.json`: recomputed point estimates and confidence intervals;
- `summary_review.json`: sampled cases, rubric, and all 50 judgments.
