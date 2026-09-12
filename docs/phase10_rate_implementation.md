# Phase 10 — rate source implementation readiness

**Source:** FRED `DFII10` — 10-year U.S. inflation-indexed Treasury constant-maturity yield  
**Protocol:** `phase10-rate-dfii10-modeled-v1`  
**Status:** implementation/hardening complete before real-data acquisition, import or preflight.  
**Strict-PIT status:** blocked; ordinary FRED history is treated as a latest snapshot without historical vintages.

## Implemented

- frozen rate source/profile/config contracts;
- bounded local FRED CSV parser with exact filename/schema/date-range checks;
- SHA-256 provenance for the original source bytes;
- modeled next-U.S.-business-day H.15 availability at 16:15 America/New_York;
- DST- and U.S.-federal-holiday-aware schedule handling;
- annual authenticated Parquet partitions plus bundle-set and source-artifact manifests;
- backward-only as-of alignment through the generic Phase-10 context layer;
- five frozen rate model features based on distinct daily observations;
- explicit missing/stale/20-observation-warmup fallback;
- exact Phase-7 prediction fallback for unusable context rows;
- the same frozen 2022/2023/2024 outer folds, 181-minute gap, three-C balanced logistic grid and inner-only selection;
- source-specific real-data metadata guard and authenticated raw-to-Parquet replay;
- generic runner support and semantic artifact validation;
- source-specific protocol snapshot copied into a real rate run;
- unit tests for source parsing, holidays/DST, provenance/tampering, feature causality and stale/warm-up semantics;
- synthetic rate integration tests for nested fitting, exact fallback, outer/calibration isolation and completion/validator replay.

## Safety properties

A real rate run cannot start unless the shared real-data preflight proves the frozen
Phase-7 reference, exact gold sample universe, local source provenance, rate bundle parity,
context coverage and clean Git identity. The preflight remains explicitly exploratory:

- `formal_benchmark_ready=false`;
- `strict_pit_source_ready=false`;
- `holdout_opened=false`;
- `champion_changed=false`;
- `trading_activated=false`.

The run decision remains mechanically `stop` when the predictive gate fails and
`seek_strict_source` when it passes. Neither outcome can promote the champion under the
modeled-latency protocol.

## Evidence boundary

No real DFII10 data has been imported or inspected as model evidence while the protocol is
being designed. No real rate preflight or market ablation has been executed yet. This
preserves the preregistration boundary.

The next step is deliberately external to implementation: acquire one development-bounded
`DFII10_2020_2024.csv`, authenticate/import it, then inspect a real-data preflight before
opening any exploratory benchmark. See `phase10_rate_source_research.md` for acquisition
constraints and `research_protocol_phase10_rate.md` for the frozen experiment contract.
