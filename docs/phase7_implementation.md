# Fase 7 — implementatiestatus

**Protocol:** `phase7-v1`  
**Release:** nog steeds `v0.1.0`  
**Empirische status:** volledige 2020–2024 ablationbenchmark nog niet uitgevoerd

## Wat is geïmplementeerd

- Afzonderlijke phase-7 featurebuilder; de MVP-builder blijft ongewijzigd.
- Causale multi-timeframe alignment voor `1min`, `3min`, `5min`, `15min`, `30min`,
  `1h` en `3h`.
- Per-timeframe returns/lags, candle geometry, momentum, realized volatility,
  distance-to-SMA, price z-score en breakoutpositie.
- Vaste UTC-sessievelden, sessie-overlap, maand- en minute-of-hour-cycli.
- Causale volatility buckets, trend score en shock score.
- Stalenessguard: een oudere hogere-timeframe candle wordt niet onbeperkt over
  datagaten vooruit gedragen.
- Gemeenschappelijk volledig feature-universe voor alle ablations.
- Cumulatieve ablations van MVP tot volledige multi-timeframe variant.
- Hergebruik van de fase-6 nested walk-forward-, preprocessing-, policy- en
  backtestcode.
- Featuredistributies per outer fold.
- Catalogus met formule, timeframe, lookback en availability-regel.
- Future-mutation-, higher-timeframe-cutoff-, handformule- en batch/online-paritytests.
- CLI: `phase7 run` en `phase7 validate`.
- Artifact hashing, source snapshots en gesloten-holdoutverificatie.

## Bewust niet geïmplementeerd

Microstructure is niet uit bid-OHLC gereconstrueerd. Zonder echte point-in-time ask,
spread en tick-countdata zouden spread-z-scores, mid/ask returns en liquidity buckets
gefabriceerde informatie zijn. Een latere rijkere bron kan deze groep als afzonderlijke
challenger toevoegen.

Ook tijd-tot/van marktopening is uitgesteld zolang er geen vertrouwde markt-/holiday
calendar is. De huidige sessievelden zijn expliciet vaste UTC-vensters.

## Ablationvolgorde

1. `mvp`
2. `price_3min`
3. `price_session_3min`
4. `price_session_regime_3min`
5. `price_session_regime_multitimeframe`

Alle vijf varianten worden op exact dezelfde sample IDs per horizon/fold geëvalueerd.

## Lokale eindverificatie

De Git-repository bevat bewust niet de grote raw/curated marktbestanden. Daarom moet
de empirische gate op de machine met de geharde fase-5-data worden uitgevoerd:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src/gold_forecasting
.\.venv\Scripts\gold-forecast.exe data validate --config configs/phase5.yaml
.\.venv\Scripts\gold-forecast.exe phase7 run --config configs/phase7.yaml
```

Na voltooiing:

```powershell
.\.venv\Scripts\gold-forecast.exe phase7 validate reports/phase7_runs/<run-id>
```

Pas daarna mogen de phase-7 exitcriteria worden afgesloten en kan `v0.2` worden
bevroren. Een uitkomst `keep_mvp_features` is geldig; er hoeft geen complexere
featuregroep te winnen.

## Promotielogica

De pipeline scheidt twee gates:

- **predictive admission:** hogere gemiddelde macro-F1, niet slechtere worst-fold
  macro-F1 en niet slechtere gemiddelde Brier dan `mvp/reference`;
- **economic promotion:** daarnaast voldoende trades en positieve base/stress
  nettoresultaten volgens het bestaande fase-6-contract.

Zelfs een economic candidate wordt alleen als kandidaat gerapporteerd. Phase 7
activeert geen paper- of live-tradingmodel.
