# Phase 10 CPI implementation verification

**Protocol:** `phase10-cpi-cuur0000sa0-modeled-v1`  
**Scope:** software/data-contract readiness only; no real CPI market-result inspection  
**Status:** implementation verified and ready for authenticated local BLS source import + real-data preflight

## Verified branch state

Implementation hardening was completed on branch `codex/phase10-external-context` through commit `92119bc73f5f71457893e1c6b0856b1ba64aa1f0` (`Harden CPI invalid-value validation`).

GitHub Actions run `34552359488` completed successfully on the pull-request merge ref corresponding to that branch head.

Quality gate evidence:

- Ruff: all checks passed;
- Mypy: success, no issues found in 103 source files;
- Pytest: 1016 passed in 113.26 seconds.

## Verified CPI implementation surface

The following CPI-specific pieces are present and exercised by tests:

- frozen `CPI_PROFILE` and protocol routing;
- disabled strict-PIT placeholder, modeled-latency exploratory source config and ablation config;
- bounded BLS `CUUR0000SA0` parser/importer with exact source/schema/scope validation;
- authenticated raw-source provenance and annual Parquet bundle manifests;
- frozen 2020-2024 historical CPI release schedule at 08:30 `America/New_York`;
- December 2024 observation excluded because its release is outside the development period;
- distinct-month CPI features and warm-up handling;
- exact price-only fallback for missing, stale or insufficient-history rows;
- CPI-aware metadata-first preflight and authenticated raw-to-bundle replay;
- exact Phase-7 common-universe/fold/gap/reference checks reused unchanged;
- CPI-specific preregistration snapshot in sealed runs;
- synthetic nested-fold train-only selection, fallback parity, tamper rejection and artifact replay.

## Frozen modeling inputs

The CPI challenger adds exactly these seven model inputs to the frozen Phase-7 price basis:

1. `cpi_index`
2. `cpi_mom_pct`
3. `cpi_yoy_pct`
4. `cpi_mom_accel_pp`
5. `cpi_yoy_accel_pp`
6. `cpi_release_age_seconds`
7. `cpi_release_within_60m`

`cpi_is_missing`, `cpi_is_stale` and `cpi_usable` remain routing/audit fields rather than substitute predictive signals.

## Research-boundary verification

This readiness verification does **not** constitute CPI admission or a benchmark result.

The following remain unchanged:

- frozen Phase-7 15-minute `mvp/logistic` reference;
- outer test years 2022, 2023 and 2024;
- 181-minute gap;
- frozen logistic `C` grid and balanced class weighting;
- exact source-specific fallback to persisted Phase-7 predictions;
- final 2025+ holdout closed;
- champion unchanged;
- trading activation false.

The current BLS flat file is a current historical snapshot rather than historical value-vintage evidence. Therefore this CPI experiment remains `modeled_latency`; even a later positive exploratory result may only justify a separately preregistered strict-source follow-up.

## Exit decision

The previously incomplete CPI software/preflight/pipeline implementation step is now formally complete.

The next permitted gate is:

1. acquire the official BLS `cu.data.1.AllItems` source locally;
2. import it into authenticated CPI bundles;
3. inspect the import provenance;
4. run the real-data CPI preflight;
5. only if that preflight is fully green, run the CPI exploratory ablation.

No real CPI preflight or benchmark was opened as part of this verification.
