# Gold Forecasting

Een reproduceerbare, leakage-bewuste onderzoekspipeline voor XAU/USD. Release `v0.2.0`
bevat de volledige research-MVP plus de geharde datalaag, het strengere fase-6
walk-forwardkader en de geverifieerde fase-7 price-only feature-ablation.

De huidige scope gebruikt HistData bid-only `1min`-candles uit 2020–2024 en causale
afgeleide timeframes tot `3h`. De betrouwbare researchbenchmark evalueert acht horizons
van 3 tot 180 minuten. Per horizon is een price-only researchchampion bevroren; de
15-minutenhorizon behoudt bewust het eenvoudige MVP-featureschema.

> Dit is researchsoftware. Fase 7 vond predictive verbeteringen op meerdere development
> folds, maar **geen economic promotion candidate**. Alle getrainde model-families selecteerden
> cash/no-trade. Probabilities zijn nog ongekalibreerd en `v0.2.0` is niet geschikt voor
> paper trading, echte orders of rendementsclaims.

## Ontwikkelomgeving

Vereist: Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## CLI

```powershell
.\.venv\Scripts\gold-forecast.exe --help
.\.venv\Scripts\gold-forecast.exe config validate --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe data import --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe data validate --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe dataset build --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe train --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe predict --config configs/mvp.yaml
.\.venv\Scripts\gold-forecast.exe mvp run --config configs/mvp.yaml --dry-run
.\.venv\Scripts\gold-forecast.exe mvp run --config configs/mvp.yaml
```

Voor het geharde dataprofiel:

```powershell
.\.venv\Scripts\gold-forecast.exe data update --config configs/phase5.yaml
.\.venv\Scripts\gold-forecast.exe data validate --config configs/phase5.yaml
.\.venv\Scripts\gold-forecast.exe mvp run --config configs/phase5.yaml
.\.venv\Scripts\python.exe scripts/verify_phase5.py --include-mvp
```

De updater hergebruikt reeds gedownloade, gehashte jaararchieven binnen de vaste
ontwikkelingsperiode 2020–2024. HistData levert hiermee geen actuele brokerfeed.
Het geharde profiel schrijft naar een eigen `phase5`-submap in de curated-laag.
Het verificatiescript vergelijkt twee herbouws en, met `--include-mvp`, de volledige
modeluitkomsten. Na die vergelijking herstelt het de gegenereerde MVP-modelartefacten.

Alle UTC-timeframes worden ondersteund: `1min`, `3min`, `5min`, `15min`, `30min`,
`1h`, `3h`, `1d` en `1mo`. Voor de huidige bron blijven `1d` en `1mo` leeg zolang
ontbrekende minuten niet aantoonbaar door een vertrouwde marktkalender worden verklaard.
Het kwaliteitsrapport markeert deze beperking expliciet. De modelinput blijft `3min`.
Zie het [fase-5-datacontract](docs/data_contract_phase5.md) voor de precieze regels.

De afzonderlijke fase-6-benchmark (afgerond) gebruikt dezelfde data en MVP-features:

```powershell
.\.venv\Scripts\gold-forecast.exe benchmark run --config configs/benchmark.yaml
.\.venv\Scripts\gold-forecast.exe benchmark validate reports/benchmark_runs/<run-id>
```

Dit commando vergelijkt acht horizons van 3 minuten tot 3 uur op drie walk-forwardjaren.
Logistische regressie, lineaire returnregressie, XGBoost en alle oorspronkelijke naïeve
referenties krijgen dezelfde records per horizon/fold. Modelkeuze en handelspolicy gebruiken
alleen eerdere inner folds; een apart kwartaal blijft gereserveerd voor latere kalibratie.
Uitkomsten, modellen, configuraties en broncode worden apart onder `reports/benchmark_runs`
opgeslagen; de bestaande MVP-artefacten blijven ongewijzigd. De benchmark downloadt niets.
Zie het [fase-6-protocol](docs/research_protocol_phase6.md) voor het vooraf vastgelegde budget,
de holdoutbeveiliging, kosten, no-tradefallback en promotiecriteria.
De interpretatie en reproduceerbaarheidsgrens staan in het [fase-6-verificatierapport](docs/phase6_verification.md)
en de [model card](docs/model_card_phase6.md).

### Fase 7 — richer price-only feature ablations

Fase 7 is als afzonderlijke challengerlaag geïmplementeerd. De bevroren MVP-featurebuilder
en fase-6-resultaten worden niet overschreven. De nieuwe laag gebruikt één
MVP-geankerd causaal sample-universe met rijkere 3min-pricefeatures, vaste
UTC-sessiecontext, volatility/trend/shock-regimes en gesloten candles uit `1min`,
`5min`, `15min`, `30min`, `1h` en `3h`. Sparse hogere-timeframewaarden blijven
expliciet missing en worden uitsluitend met train-only mediaanimputatie behandeld.

```powershell
.\.venv\Scripts\gold-forecast.exe phase7 run --config configs/phase7.yaml
.\.venv\Scripts\gold-forecast.exe phase7 validate reports/phase7_runs/<run-id>
```

De cumulatieve ablations zijn `mvp`, `price_3min`, `price_session_3min`,
`price_session_regime_3min` en `price_session_regime_multitimeframe`. Iedere variant
gebruikt dezelfde toegelaten timestamps, walk-forward-folds, modelselectie en kosten.
Per fold worden ook featuredistributies en missing coverage opgeslagen. De fase-6
feature-allowlist blijft frozen; fase 7 gebruikt uitsluitend zijn eigen gehashte catalogus.
De run downloadt niets en vereist
dat de geharde 2020–2024-data lokaal aanwezig en geldig is.

Microstructure-features worden bewust niet nagebootst: HistData levert in dit project
bid-only OHLC zonder betrouwbare historische ask, spread of tick count. De formele
`phase7-v2`-run `20260907T014255645674Z-b2afe281` is geslaagd en 11.311 artifacts zijn
daarna integraal gevalideerd. Er zijn predictive featureverbeteringen op meerdere horizons,
maar 0 economic promotion candidates. De finale holdout vanaf 2025-01-01 blijft gesloten
en probabilities blijven ongekalibreerd. Zie het
[fase-7-protocol](docs/research_protocol_phase7.md), het
[verificatierapport](docs/phase7_verification.md), de
[implementatiestatus](docs/phase7_implementation.md) en de
[v0.2-model card](docs/model_card_v0.2.md).

### Fase 8 — compact multi-timeframe neural challenger

De formele `phase8-v1` benchmark is afgerond en geverifieerd met run
`20260908T020044407464Z-24f4c0d4`. De compact GRU-core gebruikt
`1min/3min/15min` voor 3–15m en `3min/15min/1h` voor 30–180m, twee frozen seeds
en exact dezelfde outer samples als de Phase-7 champions.

De neural challenger verbetert Brier en log loss, maar **0/8 horizons** verbeteren
mean én worst-fold macro-F1. Geen horizon passeert predictive admission en alle
inner-selected policies blijven cash/no-trade. De Phase-7 champions blijven daarom
actief en de neural core gaat alleen als researchvariant mee naar Phase 9. De finale
2025+ holdout blijft gesloten.

Zie het [Phase-8 protocol](docs/research_protocol_phase8.md), het
[verificatierapport](docs/phase8_verification.md) en de
[implementatiestatus](docs/phase8_implementation.md).

### Fase 9 — direct future-candle path (afgerond)

Op `feature/phase-9-future-path` staat inmiddels de structurele Phase-9 keten:
vijf toekomstige 3min-candles, geordende 10/50/90%-quantielen, geldige OHLC-
reconstructie, een directe 15min-head en een free-running recursive one-step baseline.
Dezelfde 2022/2023/2024 nested walk-forwardstructuur en het 181-minuten-gapcontract zijn
voorbereid. Train-only pathscaling, common-sample referencepariteit, atomaire
checkpoints/artifacts en een cryptografische validator zijn aanwezig.

De integratie en canonical preflight kunnen afzonderlijk worden gecontroleerd met:

```powershell
.\.venv\Scripts\gold-forecast.exe phase9 dry-run --output reports/phase9_dry_run
.\.venv\Scripts\gold-forecast.exe phase9 validate reports/phase9_dry_run
.\.venv\Scripts\gold-forecast.exe phase9 preflight --config configs/phase9.yaml --report reports/phase9_preflight.json
```

Na de geslaagde canonical real-data preflight is het formele benchmarkcommando beschikbaar:

```powershell
.\.venv\Scripts\gold-forecast.exe phase9 run --config configs/phase9.yaml
```

`phase9 run` voert vóór het registreren van de formele run opnieuw de volledige
fail-closed preflight uit. Daarna moet de voltooide run afzonderlijk met
`phase9 validate reports/phase9_runs/<run-id>` worden geverifieerd.

De canonieke run `20260908T211301827616Z-ad6b573c` is daarna afgerond en integraal
gevalideerd (225 bestanden; completion
`sha256:a39fb334233a27c5fea65d1f4269ce3134f24bf9d89c2530b8998f5465a0a776`).

Phase-9 direct verbetert probability scores ten opzichte van Phase 7, maar de
directionele macro-F1 blijft duidelijk slechter. Tegen Phase 8 is de gemiddelde
macro-F1 slechts marginaal hoger en de slechtste fold slechter. Alle policies blijven
cash/no-trade. De Phase-7 15m champion blijft daarom actief. De direct path-head wordt
alleen als research/distributionele output behouden; recursive wordt niet als
standaardvariant meegenomen. De finale 2025+ holdout bleef gesloten.

De canonieke Phase-7/8 references zijn cryptografisch gepind in `phase9-v1`. Zie
[het fase-9-protocol](docs/research_protocol_phase9.md),
[het verificatierapport](docs/phase9_verification.md) en de
[implementatiestatus](docs/phase9_implementation.md).

`data import` downloadt uitsluitend de uit `configs/splits_mvp.yaml` afgeleide
ontwikkelingsjaren. Ruwe bronbestanden en afgeleide datasets worden lokaal gehouden en zijn
door `.gitignore` van versiebeheer uitgesloten. Zodra de data aanwezig en geldig is, hergebruikt
`mvp run` haar en bouwt het alle latere artifacts opnieuw op.

De volledige run schrijft:

- geversioneerde data-, feature-, label- en modelmanifests met hashes;
- een train-only preprocessor en opgeslagen modelbundle;
- één vergelijkbaar baseline-/modelrapport;
- reliability- en UTC-uurrapporten;
- base- en stressbacktests met beslissingen en trades;
- één voorbeeldvoorspelling en een atomisch runmanifest.

### Fase 10 — externe context (in ontwikkeling)

Phase 10 heeft nu zowel de generieke point-in-time contextlaag als de eerste
real-data silver-ablationketen. XAGUSD wordt via een afzonderlijke HistData-adapter
als jaarlijkse geauthenticeerde Parquetpartities geladen; de frozen XAUUSD-ingestie
blijft ongewijzigd. Backward as-of joins, stale/missing routing, causal silverfeatures
en exacte Phase-7 price-only fallback zijn afgedwongen.

De real-data silver runner gebruikt uitsluitend de frozen Phase-7 15m
`mvp/logistic` champion als baseline, exact dezelfde 2022/2023/2024 folds en
181-minuten gap, en dezelfde drie logistic `C`-waarden met balanced class weighting.
De volledige gold sample-ID/value parity wordt vóór training bewezen.

```powershell
\.\.venv\Scripts\gold-forecast.exe phase10 silver-import --archive-directory <XAGUSD-map> --output data/context/phase10/silver --source configs/phase10_silver_exploratory.yaml
\.\.venv\Scripts\gold-forecast.exe phase10 preflight --config configs/phase10_silver_ablation.yaml --report reports/phase10_preflight.json
\.\.venv\Scripts\gold-forecast.exe phase10 run --config configs/phase10_silver_ablation.yaml
\.\.venv\Scripts\gold-forecast.exe phase10 validate reports/phase10_runs/<run-id>
```

HistData levert geen historische per-row release timestamps. Daarom blijft
`available_at = candle close + 60s` expliciet **modeled latency**. De run is uitsluitend
exploratory: hij kan nooit de champion wijzigen of trading activeren. Een positief resultaat
kan alleen de beslissing `seek_strict_source` rechtvaardigen. De 2025+ holdout blijft
gesloten.

De runner/validator zijn geïmplementeerd en CI is groen; de echte lokale XAGUSD
2020–2024 preflight en markt-run zijn nog niet uitgevoerd. Zie
[het Phase-10 protocol](docs/research_protocol_phase10.md) en
[de implementatiestatus](docs/phase10_implementation.md).

## Kwaliteitscontroles

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
```

Zie [het onderzoeksprotocol](docs/research_protocol.md) voor de vooraf bevroren tijds-, label-,
kosten- en splitregels, [de v0.2-model card](docs/model_card_v0.2.md) voor resultaten en beperkingen,
en [de roadmap](roadmap.md) voor de gefaseerde vervolgstappen.
