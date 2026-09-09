# Phase 10 silver verification

**Protocol:** `phase10-silver-modeled-v1`  
**Run:** `20260909T133304370198Z-be3c4015`  
**Code:** `ce904e9a0d320e543970e5a19800833b4f994130`  
**Completion:** `sha256:c971eed86adf0d5b1e26cfd4dfa4a45436f58553945d3d70d6046f8dab0741d3`  
**Verified files:** 181  
**Decision:** `stop`

## Scope and safety

This run is an exploratory modeled-latency XAGUSD ablation. It is not strict
point-in-time evidence and cannot promote a champion or activate trading.

- frozen reference: Phase-7 15m `mvp/logistic`;
- reference run: `20260907T014255645674Z-b2afe281`;
- outer test years: 2022, 2023, 2024;
- gap: 181 minutes;
- final 2025+ holdout remained closed;
- champion promotion: false;
- trading activation: false.

The validator replayed the persisted run and verified all 181 inventoried files.

## Aggregate predictive result

| Metric | Phase-7 reference | Silver | Delta |
|---|---:|---:|---:|
| Accuracy | 0.595214 | 0.605645 | +0.010431 |
| Balanced accuracy | 0.422542 | 0.418432 | -0.004111 |
| Macro-F1 | 0.425052 | 0.421555 | -0.003497 |
| Brier | 0.579063 | 0.571309 | -0.007754 |
| Log loss | 0.984323 | 0.971756 | -0.012567 |
| Calibration error | 0.152823 | 0.158428 | +0.005605 |
| Return MAE (bps) | 6.008370 | 6.008568 | +0.000198 |

Silver improves ordinary accuracy, Brier and log loss, but the preregistered
primary directional criterion regresses.

Macro-F1 is lower on every outer fold:

| Fold | Reference | Silver | Delta |
|---|---:|---:|---:|
| 2022 | 0.425102 | 0.422220 | -0.002882 |
| 2023 | 0.423582 | 0.418853 | -0.004729 |
| 2024 | 0.426473 | 0.423592 | -0.002881 |

Therefore the frozen predictive admission gate fails.

## Context coverage

Outer-fold usable silver context was:

- 2022: 90.076%;
- 2023: 84.539%;
- 2024: 92.780%.

Fallback fractions were respectively 9.924%, 15.461% and 7.220%. The negative
predictive result therefore cannot be explained by near-total source unavailability.

## Economic result

Both the frozen reference and silver challenger retained the explicit cash/no-trade
policy in every fold:

- executed trades: 0;
- exposure: 0;
- base net return: 0 bps;
- stress net return: 0 bps.

The economic gate therefore fails.

## Interpretation

The modeled-latency XAGUSD source slightly improves probability scoring, but does
not improve robust three-class discrimination. Higher ordinary accuracy is not
sufficient because balanced accuracy and macro-F1 both deteriorate.

The preregistered decision is therefore `stop`:

- keep the frozen Phase-7 15m `mvp/logistic` research champion;
- do not seek a more expensive strict-PIT silver source on the basis of this result;
- do not retain XAGUSD as an admitted Phase-10 context source;
- proceed to the next independent Phase-10 source hypothesis: a dollar proxy;
- do not reopen or reuse the final holdout.

This is a useful negative ablation and remains part of the audit trail.
