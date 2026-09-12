# Phase 10 CPI source research

## Decision

The fourth independent Phase-10 source hypothesis uses **BLS CPI-U All items, not seasonally adjusted**, series `CUUR0000SA0`.

Official source artifacts:

- BLS time-series data file: `https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems`
- BLS series dictionary: `https://download.bls.gov/pub/time.series/cu/cu.series`
- BLS annual release calendars: `https://www.bls.gov/schedule/<year>/` for 2020-2024.

The series dictionary identifies `CUUR0000SA0` as All items in U.S. city average, all urban consumers, not seasonally adjusted. The annual BLS calendars publish CPI at 08:30 Eastern Time and provide the historical dates used by the importer.

## Why this source

CPI is economically distinct from the rejected silver, inverse-EURUSD and DFII10 hypotheses. Inflation information can alter expected real rates, monetary-policy expectations and gold demand. The test remains deliberately compact: one official BLS series, evaluated independently against the frozen Phase-7 price-only champion.

The unadjusted All-items index was selected because it is a direct published price-level series and avoids adding a second seasonally adjusted source hypothesis. Monthly and year-over-year changes are derived causally from released index levels.

## Point-in-time limitation

The annual BLS schedules establish historical release dates, but the current BLS time-series flat file is a latest snapshot. It does not by itself prove the exact value/version visible at each historical prediction cutoff. Consequently the first CPI ablation is classified as `modeled_latency` and cannot promote the champion or activate trading.

A positive exploratory result may justify a separate search for authenticated historical BLS release tables or archived snapshots. A negative result stops the CPI source without that additional work.

## No consensus-surprise feature

This experiment does **not** use economist consensus, forecasts, or a constructed CPI surprise. Those require an independent historical expectations provider and would mix two source hypotheses. The registered CPI experiment uses only BLS-published index levels and their causal transforms.

## Development boundary

The source parser requires all 60 monthly `CUUR0000SA0` rows for reference months January 2020 through December 2024, but only values whose CPI release occurs before `2025-01-01T00:00:00Z` may enter the development experiment. December 2024 CPI is therefore excluded because it was released in January 2025.

The final 2025+ gold holdout remains inaccessible.
