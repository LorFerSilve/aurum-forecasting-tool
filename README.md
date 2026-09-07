# Gold Forecasting

Een reproduceerbare, leakage-bewuste onderzoekspipeline voor XAU/USD. Release `v0.1.0`
implementeert de volledige research-MVP: publieke historische data, validatie, resampling,
causale features, labels, chronologische splits, modelselectie, voorspelling, evaluatie en een
eenvoudige kostenbewuste backtest.

Fase 5 voegt een apart gehard dataprofiel toe met hervatbare acquisitie, geversioneerde
bronmetadata, acht afgeleide timeframes en dagelijkse kwaliteitsrapporten. De huidige
release blijft `v0.1.0`; `v0.2` volgt pas na fasen 6–7.

De huidige scope gebruikt HistData bid-only `1min`-candles uit 2020–2024, afgeleide `3min`-
en `15min`-candles, achttien eenvoudige price-only features en een 15-minutenrichting
(`down`, `neutral`, `up`). De broker-/databron zit achter een vervangbare provideradapter.

> Dit is researchsoftware. De eerste echte backtest is negatief en de probabilities zijn
> voorlopig en ongekalibreerd. De release is niet geschikt voor paper trading, echte orders
> of rendementsclaims.

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

De afzonderlijke fase-6-benchmark (afgerond; fase 7 volgt) gebruikt dezelfde data en features:

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

## Kwaliteitscontroles

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
```

Zie [het onderzoeksprotocol](docs/research_protocol.md) voor de vooraf bevroren tijds-, label-,
kosten- en splitregels, [de model card](docs/model_card_v0.1.md) voor resultaten en beperkingen,
en [de roadmap](roadmap.md) voor de gefaseerde vervolgstappen.
