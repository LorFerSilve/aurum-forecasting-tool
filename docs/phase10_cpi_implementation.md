# Phase 10 CPI implementation

The CPI source is implemented as the fourth independent Phase-10 challenger under protocol `phase10-cpi-cuur0000sa0-modeled-v1`.

## Components

- `src/gold_forecasting/phase10/cpi_bls.py` authenticates and parses the official BLS `cu.data.1.AllItems` flat file, selects `CUUR0000SA0`, applies the frozen 2020-2024 CPI release calendar and writes annual authenticated Parquet bundles.
- `src/gold_forecasting/phase10/cpi_features.py` builds the frozen CPI level, monthly/yearly inflation, acceleration and post-release timing features over distinct monthly observations.
- `configs/phase10_cpi_exploratory.yaml` freezes the modeled-latency source contract.
- `configs/phase10_cpi.yaml` is a disabled strict-PIT placeholder and cannot be used until historical value/version evidence exists.
- `configs/phase10_cpi_ablation.yaml` binds the source to the exact Phase-7 reference, folds, gap and evaluation gates.
- `scripts/import_phase10_cpi.py` creates the authenticated local bundle set from a user-downloaded official BLS file.

## Safety properties

- exact filename and source identity checks;
- bounded regular-file reads; symlinks rejected;
- exact canonical BLS schema;
- exact monthly 2020-2024 coverage for the selected series;
- December 2024 value excluded because its release is outside development;
- official historical release dates converted from Eastern Time with DST awareness;
- raw source SHA-256 and frozen schedule digest recorded;
- annual Parquet partitions authenticated by manifests;
- raw-to-bundle replay required by real-data preflight;
- backward-only as-of joins;
- stale/missing/warm-up rows use exact frozen Phase-7 predictions;
- no 2025+ gold holdout access;
- modeled-latency evidence cannot promote or activate trading.

## Local source layout

The official file must be saved exactly as:

`data/raw/phase10/cpi/cu.data.1.AllItems`

Import command:

```powershell
.\.venv\Scripts\python.exe scripts\import_phase10_cpi.py
```

A real CPI preflight or benchmark must not be run until implementation CI is green and the imported source metadata has been inspected.
