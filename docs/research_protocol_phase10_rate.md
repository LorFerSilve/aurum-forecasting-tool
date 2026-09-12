# Research protocol — Phase 10 rate source

**Protocol:** `phase10-rate-dfii10-modeled-v1`  
**Status:** frozen before any real DFII10 preflight or market-result inspection.  
**Source hypothesis:** FRED `DFII10`, 10-Year Treasury Inflation-Indexed Security, Constant Maturity.

## Purpose

This is the third independent Phase-10 source hypothesis. Silver and inverse EURUSD have
already ended with `stop` and are not combined with this source. The experiment asks only
whether a daily U.S. 10-year real-yield context adds robust information to the frozen
Phase-7 15-minute price-only research champion.

This protocol is exploratory because the ordinary FRED historical CSV is a current
snapshot and does not preserve the original historical vintages used for each past date.
Modeled H.15 availability therefore cannot prove strict point-in-time availability.
A positive result may only justify a follow-up using historical vintage evidence such as
ALFRED; it cannot promote a champion or activate trading.

## Frozen reference and development universe

The benchmark reference remains exactly:

- Phase-7 run `20260907T014255645674Z-b2afe281`;
- code `3f0a703568224fe9169b1e9f8d61dad131f0005b`;
- completion `sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f`;
- horizon `15min`;
- variant/family `mvp/logistic`.

The experiment retains:

- outer years 2022, 2023 and 2024;
- exact Phase-7 gold rows, values and ordered sample IDs;
- the same inner folds;
- 181-minute train-side gap/purge;
- the same labels, costs and execution assumptions;
- final holdout from 2025-01-01 UTC closed.

The rate source may never reduce the gold universe. Missing, stale or insufficient rate
history routes the affected gold sample to the exact persisted Phase-7 probabilities and
expected return.

## Source and availability contract

The scalar source is `DFII10`, quoted by FRED in percent. The local raw artifact must be
named exactly `DFII10_2020_2024.csv` and contain only development reference dates from
2020-01-01 through 2024-12-31. The source bytes are SHA-256 authenticated before derived
bundles are trusted.

The exploratory timing rule is frozen as:

1. `observed_at_utc` = the DFII10 reference date at 00:00 UTC;
2. `available_at_utc` = 16:15 `America/New_York` on the next regular U.S. federal
   business day after the reference date;
3. New York daylight-saving time is respected;
4. a `.` or blank FRED value means no new observation and is skipped;
5. the project ingestion timestamp is stored separately and is not historical release
   evidence;
6. `availability_basis=modeled_latency` throughout this experiment;
7. the downloaded FRED history is treated as a latest snapshot with no original vintages.

The stale threshold is 432000 seconds (five days). A rate observation older than this at
a gold prediction cutoff remains auditable but cannot be used by the challenger.

No configuration change may relabel this source as `provider_timestamp`. A later strict
PIT follow-up requires a new protocol and authenticated historical vintage evidence.

## Frozen rate feature contract

DFII10 is a yield, not a tradable price level, so changes are additive yield differences,
not log returns. The candidate adds exactly these five model inputs:

| Feature | Definition |
|---|---|
| `rate_level_pct` | latest fresh DFII10 level in percent |
| `rate_change_1obs_bps` | `(current - previous distinct observation) * 100` |
| `rate_change_5obs_bps` | `(current - observation 5 distinct releases earlier) * 100` |
| `rate_change_20obs_bps` | `(current - observation 20 distinct releases earlier) * 100` |
| `rate_age_seconds` | prediction cutoff minus reference observation time |

`rate_is_missing`, `rate_is_stale` and `rate_usable` are routing/audit fields and are not
substitute model signals.

The same daily DFII10 observation is repeated across many intraday gold rows. Repetition
must not manufacture zero rate changes. The 1/5/20 changes are therefore computed only
across distinct rate observations and then carried unchanged until a newer observation is
available. All five model features must be finite before `rate_usable=True`; consequently
the first 20 distinct usable observations form intentional feature warm-up/fallback.

No interpolation, backward fill from the future, inferred intraday rate path, nominal
Treasury proxy, curve reconstruction or interaction with the rejected silver/dollar
sources is allowed in this ablation.

## Frozen model isolation

The experiment tests the source, not a new learner. The challenger therefore uses:

- frozen Phase-7 MVP price features plus the five rate features above;
- logistic regression only;
- `C ∈ {0.1, 1.0, 10.0}` exactly;
- `class_weight="balanced"` exactly;
- validation macro-F1 with log-loss tie-break;
- train-only preprocessing;
- model and policy selection on inner history only;
- reserved calibration block not fitted;
- no outer-test influence on fitting, imputation, scaling, selection or policy.

If the context model cannot be fitted safely, all affected predictions remain exact
Phase-7 fallback.

## Predictive and economic gates

Against the frozen reference, the same Phase-7 predictive gate is retained:

1. mean macro-F1 must be higher;
2. worst-fold macro-F1 must not be worse;
3. mean Brier score must not be worse.

Accuracy, balanced accuracy, log loss, calibration error, return MAE, coverage and age are
reported but cannot replace the preregistered gate.

The existing economic gate is also replayed:

- worst-fold trade count at least 20;
- worst-fold base net bps > 0;
- mean stress net bps > 0;
- worst-fold base net bps at least the reference.

Because this is modeled-latency evidence, even a predictive/economic pass cannot itself
promote the champion or activate paper/live trading.

## Decision rule

The run-level decision is mechanically frozen:

- predictive gate fails -> `stop`;
- predictive gate passes -> `seek_strict_source`.

`seek_strict_source` means seek and preregister historical-vintage DFII10 evidence, with
ALFRED being a concrete candidate. It does not mean promotion.

## Mandatory preflight and artifact evidence

Before RunRegistry opens, preflight must verify:

- clean committed Git identity;
- authenticated local DFII10 raw bytes and metadata;
- exact annual 2020–2024 bundle manifests and hashes;
- exact reparsing parity from the raw CSV to every annual Parquet partition;
- frozen Phase-7 completion and reference predictions;
- exact gold data versions, feature configuration, values and ordered sample IDs;
- exact 2022/2023/2024 folds and 181-minute gap;
- causal backward as-of alignment;
- missing/stale/warm-up/fallback coverage;
- `formal_benchmark_ready=false`;
- `strict_pit_source_ready=false`;
- `holdout_opened=false`;
- `champion_changed=false`;
- `trading_activated=false`.

The completed run must persist and seal source/config/code/dependency snapshots, features,
inner/outer predictions, selection, training audits, base/stress backtests and the frozen
Phase-7 reference artifacts. Validation replays inventory hashes, source-derived features,
fallback, candidate selection, policies, metrics and run-level gates without deserializing
model checkpoints.

No real-data preflight or benchmark result may be inspected before this protocol and its
code/tests are committed and CI-green.
