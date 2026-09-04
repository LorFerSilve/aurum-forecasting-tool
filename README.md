# Gold Forecasting

Een reproduceerbare, leakage-bewuste onderzoekspipeline voor XAU/USD. Release `v0.1.0`
implementeert de volledige research-MVP: publieke historische data, validatie, resampling,
causale features, labels, chronologische splits, modelselectie, voorspelling, evaluatie en een
eenvoudige kostenbewuste backtest.

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
