# Phase 10 CPI preregistration

Protocol: `phase10-cpi-cuur0000sa0-modeled-v1`

This document freezes the fourth Phase-10 source experiment before any real CPI benchmark result is inspected.

## Hypothesis

Add one official U.S. inflation context stream to the frozen Phase-7 15-minute `mvp/logistic` price-only reference: BLS `CUUR0000SA0`, CPI-U All items in U.S. city average, all urban consumers, not seasonally adjusted.

## Source and availability

- source ID: `cpi`
- source value: published CPI index level
- reference timestamp: first day of the reference month at 00:00 UTC
- availability: official BLS 2020-2024 CPI calendar date at 08:30 `America/New_York`
- source snapshot: current BLS time-series flat file
- evidence class: `modeled_latency`
- revision policy for this exploratory snapshot: `append_only`
- stale threshold: 90 days
- missing/stale/feature-warm-up policy: exact Phase-7 price-only fallback

The release dates are historical BLS calendar evidence, but the current flat file does not prove historical value vintages. This run is therefore not strict PIT and cannot promote a champion or activate trading.

## Frozen model inputs

Exactly seven CPI model inputs are admitted:

1. `cpi_index`
2. `cpi_mom_pct`
3. `cpi_yoy_pct`
4. `cpi_mom_accel_pp`
5. `cpi_yoy_accel_pp`
6. `cpi_release_age_seconds`
7. `cpi_release_within_60m`

Audit/routing flags `cpi_is_missing` and `cpi_is_stale` are not model inputs. Monthly changes are calculated only across distinct CPI observations and then carried forward. The release-age clock starts at the CPI publication timestamp. The first 60 minutes are represented by a frozen half-open indicator `[release, release+60m)`.

Fourteen distinct monthly observations are required before year-over-year acceleration is finite. Earlier rows therefore use exact Phase-7 fallback.

No consensus, forecast, surprise, core-CPI, seasonally adjusted CPI or other macro source may be added to this experiment after seeing the result.

## Frozen benchmark methodology

- horizon: 15 minutes only
- outer test years: 2022, 2023, 2024
- purge gap: 181 minutes
- exact frozen Phase-7 gold sample universe and values
- exact frozen Phase-7 `mvp/logistic` reference probabilities
- logistic `C` grid: `{0.1, 1.0, 10.0}`
- `class_weight="balanced"` only
- preprocessing fitted on train rows only
- candidate selection on inner folds only
- calibration block remains reserved
- unsupported context rows copy the frozen Phase-7 predictions exactly
- no 2025+ holdout access.

## Admission gates

Predictive admission requires all three:

1. mean macro-F1 higher than Phase 7;
2. worst-fold macro-F1 no worse than Phase 7;
3. mean multiclass Brier no worse than Phase 7.

The economic gate remains frozen:

- worst-fold trades >= 20;
- worst-fold base net > 0;
- mean stress net > 0;
- worst-fold base net >= reference.

Because the source is modeled-latency, even a positive predictive gate can only produce `seek_strict_source`. It cannot promote the research champion or activate trading.

## Decision rule

- predictive gate fails -> `stop`; do not seek historical value-vintage evidence;
- predictive gate passes -> `seek_strict_source`; authenticate historical BLS value/release snapshots and rerun under a separately frozen strict protocol;
- never combine CPI with previously rejected sources on the basis of this result;
- final holdout remains closed in every case.
