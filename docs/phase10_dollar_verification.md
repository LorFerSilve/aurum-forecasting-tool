# Phase 10 dollar-proxy verification

**Protocol:** `phase10-dollar-eurusd-modeled-v1`  
**Run:** `20260910T121152928034Z-d85cf75b`  
**Code:** `e5ab89ed28293d0130e575dffb057317ea2bcc3c`  
**Completion:** `sha256:951db3464f4e6afbbfe8d3895e4ae0cc41c4ce1a0dbd74e11b39fcd42468fe0c`  
**Verified files:** 185  
**Decision:** `stop`

## Scope and safety

This run is the preregistered exploratory modeled-latency ablation of HistData
EURUSD M1 bid-close, with inverse EURUSD returns used only as a bilateral
USD-strength hypothesis. It is not ICE DXY, not strict point-in-time evidence,
and cannot promote a champion or activate trading.

- frozen reference: Phase-7 15m `mvp/logistic`;
- reference run: `20260907T014255645674Z-b2afe281`;
- outer test years: 2022, 2023, 2024;
- gap: 181 minutes;
- final 2025+ holdout remained closed;
- champion promotion: false;
- trading activation: false.

The validator replayed the persisted run and verified all 185 inventoried files.

## Aggregate predictive result

| Metric | Phase-7 reference | Dollar proxy | Delta |
|---|---:|---:|---:|
| Accuracy | 0.595214 | 0.602477 | +0.007262 |
| Balanced accuracy | 0.422542 | 0.420228 | -0.002315 |
| Macro-F1 | 0.425052 | 0.423296 | -0.001756 |
| Brier | 0.579063 | 0.573153 | -0.005910 |
| Log loss | 0.984323 | 0.975155 | -0.009168 |
| Calibration error | 0.152823 | 0.155789 | +0.002966 |
| Return MAE (bps) | 6.008370 | 6.008278 | -0.000092 |

The dollar proxy improves ordinary accuracy, Brier and log loss, but the
preregistered directional criteria regress. The return-MAE change is negligible
at approximately 0.00009 bps.

Macro-F1 is lower on every outer fold:

| Fold | Reference | Dollar proxy | Delta |
|---|---:|---:|---:|
| 2022 | 0.425102 | 0.424749 | -0.000353 |
| 2023 | 0.423582 | 0.420619 | -0.002964 |
| 2024 | 0.426473 | 0.424520 | -0.001953 |

Balanced accuracy is also lower on every outer fold. Therefore the frozen
predictive admission gate fails despite improved probability scores.

## Context coverage

Outer-test usable dollar context was:

- 2022: 94.650%; fallback 5.350%;
- 2023: 89.810%; fallback 10.190%;
- 2024: 92.987%; fallback 7.013%.

Coverage is therefore high enough that the negative directional result cannot
reasonably be attributed to near-total source unavailability.

## Economic result

Both the frozen reference and dollar challenger retained the explicit cash/no-trade
policy in every outer fold:

- executed trades: 0;
- exposure: 0;
- base net return: 0 bps;
- stress net return: 0 bps.

The economic gate therefore fails.

## Interpretation and decision

The inverse-EURUSD context changes the probability distribution in a direction
that slightly improves scoring rules and ordinary accuracy, but it does not
improve robust three-class directional discrimination. Macro-F1 and balanced
accuracy both deteriorate consistently across the three frozen outer years.

The preregistered decision is therefore `stop`:

- retain the frozen Phase-7 15m `mvp/logistic` research champion;
- do not admit inverse EURUSD as a Phase-10 context source;
- do not seek a strict-PIT EURUSD source on the basis of this result;
- do not reinterpret this negative result as evidence against every possible
  broad-dollar index hypothesis such as authenticated ICE DXY;
- proceed to the next independent Phase-10 source hypothesis: one rate proxy;
- do not reopen or reuse the final holdout.

This negative ablation remains part of the permanent audit trail.
