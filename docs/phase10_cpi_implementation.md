# Phase 10 CPI implementation

The CPI source is implemented as the fourth independent Phase-10 challenger under protocol `phase10-cpi-cuur0000sa0-modeled-v1`.

## Components

- `src/gold_forecasting/phase10/cpi_acquisition.py` acquires only the fixed official BLS `cu.data.1.AllItems` URL with redirects disabled, a 16 MiB bound, lightweight transport-shape checks and atomic no-clobber persistence by default.
- `scripts/fetch_phase10_cpi.py` exposes that acquisition step without changing the frozen modeling protocol.
- `src/gold_forecasting/phase10/cpi_bls.py` authenticates and parses the official BLS `cu.data.1.AllItems` flat file, selects `CUUR0000SA0`, applies the frozen 2020-2024 CPI release calendar and writes annual authenticated Parquet bundles.
- `src/gold_forecasting/phase10/cpi_features.py` builds the frozen CPI level, monthly/yearly inflation, acceleration and post-release timing features over distinct monthly observations.
- `configs/phase10_cpi_exploratory.yaml` freezes the modeled-latency source contract.
- `configs/phase10_cpi.yaml` is a disabled strict-PIT placeholder and cannot be used until historical value/version evidence exists.
- `configs/phase10_cpi_ablation.yaml` binds the source to the exact Phase-7 reference, folds, gap and evaluation gates.
- `scripts/import_phase10_cpi.py` creates the authenticated local bundle set from the acquired official BLS file.

## Safety properties

- acquisition uses one hard-coded HTTPS BLS source URL; redirects are rejected;
- download size is bounded and the destination is written atomically;
- an existing source file is not replaced unless overwrite is explicitly requested;
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

The acquisition helper does **not** turn the latest BLS flat-file snapshot into historical value-vintage evidence. Full schema/content authentication remains in the importer and the experiment remains `modeled_latency`.

## Local source layout

The official file must be saved exactly as:

`data/raw/phase10/cpi/cu.data.1.AllItems`

Recommended acquisition command:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_phase10_cpi.py
```

The command prints the fixed source URL, local path, byte size and SHA-256 so the acquired snapshot can be inspected before import. It refuses to overwrite an existing file unless `--overwrite` is supplied explicitly.

Import command:

```powershell
.\.venv\Scripts\python.exe scripts\import_phase10_cpi.py
```

A manually downloaded official file remains acceptable because the importer performs the authoritative schema, series, scope and SHA-256 provenance checks.

A real CPI preflight or benchmark must not be run until implementation CI is green and the imported source metadata has been inspected.
