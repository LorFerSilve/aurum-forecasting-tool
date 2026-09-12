# Phase 10 rate-proxy verification

**Protocol:** `phase10-rate-dfii10-modeled-v1`  
**Run:** `20260911T003632070648Z-c74646e1`  
**Code:** `c1625179505d0507bc045dee924b8bb13c9334f6`  
**Completion:** `sha256:9551bc6e9eef8aff67dae972c84a071bdbb35a0275a0c654fc45e846afe129f5`  
**Verified files:** 187  
**Decision:** `stop`

## Scope and safety

This run is the preregistered exploratory modeled-latency ablation of FRED
`DFII10`, the 10-year U.S. Treasury inflation-indexed constant-maturity yield.
The ordinary FRED history is a current snapshot rather than historical vintage
evidence, so availability was conservatively modeled from the H.15 schedule and
this run cannot promote a champion or activate trading.

- frozen reference: Phase-7 15m `mvp/logistic`;
- reference run: `20260907T014255645674Z-b2afe281`;
- outer test years: 2022, 2023, 2024;
- gap: 181 minutes;
- final 2025+ holdout remained closed;
- champion promotion: false;
- trading activation: false.

The validator replayed the persisted run and verified all 187 inventoried files.

## Aggregate predictive result

| Metric | Phase-7 reference | DFII10 rate proxy | Delta |
|---|---:|---:|---:|
| Accuracy | 0.595214 | 0.591972 | -0.003242 |
| Balanced accuracy | 0.422542 | 0.420589 | -0.001954 |
| Macro-F1 | 0.425052 | 0.418498 | -0.006554 |
| Brier | 0.579063 | 0.579547 | +0.000484 |
| Log loss | 0.984323 | 0.984219 | -0.000104 |
| Calibration error | 0.152823 | 0.148522 | -0.004301 |
| Return MAE (bps) | 6.008370 | 6.017040 | +0.008670 |

The rate proxy improves calibration error and produces a negligible aggregate
log-loss improvement, but it regresses ordinary accuracy, balanced accuracy,
macro-F1, Brier and return MAE. The preregistered predictive gate therefore
fails.

Macro-F1 by outer fold:

| Fold | Reference | DFII10 | Delta |
|---|---:|---:|---:|
| 2022 | 0.425102 | 0.413844 | -0.011258 |
| 2023 | 0.423582 | 0.425219 | +0.001636 |
| 2024 | 0.426473 | 0.416433 | -0.010040 |

The source helps slightly in 2023 but regresses materially in 2022 and 2024.
Mean macro-F1 is lower, worst-fold macro-F1 is lower, and mean Brier is worse;
all three frozen predictive admission conditions fail.

## Context coverage

Outer-test usable DFII10 context was:

- 2022: 84.814%; fallback 15.186%;
- 2023: 73.003%; fallback 26.997%;
- 2024: 86.030%; fallback 13.970%.

The fallback includes stale observations and the intentional 20-distinct-release
feature warm-up. The negative result is therefore evaluated with substantial,
not near-zero, context coverage in every outer year.

## Economic result

Both the frozen reference and rate challenger retained the explicit cash/no-trade
policy in every outer fold:

- executed trades: 0;
- exposure: 0;
- base net return: 0 bps;
- stress net return: 0 bps.

The economic gate therefore fails.

## Interpretation and decision

DFII10 does not provide robust incremental directional information under the
frozen Phase-10 experiment. Its small 2023 macro-F1 improvement does not offset
the substantially worse 2022 and 2024 results, and the aggregate Brier criterion
also fails.

The preregistered decision is therefore `stop`:

- retain the frozen Phase-7 15m `mvp/logistic` research champion;
- do not admit DFII10 as a Phase-10 context source;
- do not seek ALFRED/vintage strict-PIT DFII10 evidence on the basis of this result;
- do not combine DFII10 with the rejected silver or inverse-EURUSD sources;
- proceed to the next independent Phase-10 source hypothesis: U.S. CPI event context;
- do not reopen or reuse the final holdout.

This negative ablation remains part of the permanent audit trail.
