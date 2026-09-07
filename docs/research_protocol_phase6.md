# Phase 6 research protocol — phase6-v1

Frozen before running the phase-6 benchmark. This extends, and does not replace,
the v0.0 contract used by the immutable v0.1 MVP.

## Data and time

Public HistData bid-only development data, 2020–2024; no additional acquisition.
The final holdout begins 2025-01-01 UTC and remains inaccessible. The previously
discussed 2024 MVP test is now explicitly a development evaluation period, not
an untouched confirmation of choices made in this phase.

Closed 3-minute MVP features are unchanged. Labels use entry at prediction +1
minute, exit at entry +H, for H=3,6,9,12,15,30,60,180 minutes. Only uninterrupted
minute paths qualify. A 24-hour horizon is deferred: no authoritative closure
calendar exists to distinguish missing minutes from closed markets. This is
not permission to interpolate gaps. Record each horizon's dropped-path coverage.
Direction retains the symmetric +/-6 bps log-return neutral zone (4 bps base
cost plus 2 bps buffer). Continuous arithmetic return, range and realized
volatility are also recorded. Bid opens are observed; asks are explicitly
spread proxies, not observed executable quotes.

## Walk-forward schedule

Outer evaluations are calendar years 2022, 2023, 2024, with expanding history
starting in 2020. For outer year Y, the two inner validation quarters are
April–June and July–September of Y-1. Each inner fit uses only earlier records.
October–December of Y-1 is reserved for later calibration and is never used for
fitting, early stopping, hyperparameters or decision thresholds in this phase.
The outer model is refitted on all eligible history before that reserved block.
Training blocks end with a common 181-minute embargo (max horizon + latency),
independent of the horizon. All block labels must mature strictly before the
block end. Validation/test records are never subsampled or randomly shuffled.

## Fixed candidate and compute budget

All candidates share exact per-horizon/fold sample IDs, features, labels and
cost assumptions. Different horizons may have different eligible rows.

- Retain all six MVP naive references and add empirical training-class priors.
- Frozen reference: logistic C=0.1, balanced, refitted for each fold.
- Logistic challenger: C=0.1,1,10, balanced (three candidates).
- Ridge return regression: alpha=0.1,10,1000 (three candidates).
- XGBoost: max_depth=2,3,4 (three candidates), learning_rate=0.05,
  hist trees, max 120 rounds, early stopping patience 10, n_jobs=2.
- Every tuned family gets the same two inner folds and three configurations;
  fit counts and durations are recorded, not claimed to be equal FLOPs.
- Classifiers rank by mean inner macro-F1, then lower log loss; ridge ranks by
  lower arithmetic-return MAE. Fixed-order candidate tie breaking.
- Early stopping sees only inner validation; final refit uses the median
  selected inner best-round count and never sees calibration or outer labels.

Preprocessors, empirical priors, class-conditional return expectations and ridge
residual scale are fitted on training rows only. Ridge probabilities are a
normal-residual approximation, explicitly uncalibrated. Classifier expected
returns use training class-conditional arithmetic means. A rolling historical
volatility baseline scales the 20-return 3-minute root-sum-square by sqrt(H/60); a training median
range baseline and a zero-return baseline accompany continuous-error metrics.
These are research assumptions, not demonstrated economic edge.

## Decision policy and backtest

For each family choose among (confidence, minimum expected net bps)
=(0.5,0),(0.6,2),(0.7,4) using only its selected model's inner predictions.
Require >=20 trades in each inner fold and positive mean net fixed-notional
return. If none qualifies, freeze a cash policy (confidence 1, infinite edge
implemented as a finite prohibitive threshold). Never choose thresholds using
the outer evaluation. Record all policy attempts and no-trade outcomes.

Base cost is 3 bps proxy spread +0.5 bps slippage per side; stress spread is
4.5 bps with unchanged slippage. No commission/financing in this intraday
research proxy. Orders/signals at prediction time, fills at entry time;
positions cannot overlap per instrument within a strategy. Record gross,
execution costs, net, drawdown, turnover, exposure and UTC-session breakdowns.
Returns are summed arithmetic bps per fixed initial bid notional, not account
returns, compounded equity or a live execution claim.

## Reporting and promotion

Preserve each fit's sample hashes, configs, dependencies, model/scaler bundles,
inner predictions, policy attempts and outer predictions/trades. Report mean,
median and worst fold metrics, including probability quality and continuous
errors. Coverage and worst-fold performance remain visible even with no trades.
Outer results describe development performance, not causal profitability proof.

Register the frozen logistic reference as research champion per horizon. A
challenger is only eligible for a later explicit promotion when its mean macro-F1
improves, mean Brier does not worsen, worst macro-F1 does not worsen, every
outer fold has >=20 trades and positive base net, mean stress net is positive,
and worst base net is no worse than the reference. Otherwise keep champion.
No trading champion or live orders are activated by this research command.
Full OOF calibration remains phase 11, richer feature ablations phase 7.

Reduced horizon/year subsets, shortened budgets and smaller minimum trade
counts are permitted only for synthetic tests and exploratory smoke runs. Such
runs are explicitly marked reduced scope and do not establish phase completion.

XGBoost early stopping follows the [official estimator API](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html),
with our own strictly chronological splits instead of the documentation's
random-split examples. Protocol changes require a new version and decision log.
