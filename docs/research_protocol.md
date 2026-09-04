# Onderzoeksprotocol v0.0

Status: **bevroren voor de research-MVP**  
Bevriezingsdatum: 2026-09-04  
Beoogde release: `v0.1`  
Einddoel van de roadmap: professioneel paper trading; echte orders vallen buiten de MVP.

Dit protocol legt vooraf vast hoe de eerste verticale onderzoeksketen wordt gebouwd en gemeten. Het doel van `v0.1` is een correcte, reproduceerbare pipeline, niet een rendementsclaim. Wijzigingen aan dit contract vereisen een nieuwe protocolversie en een vermelding in het beslissingslog.

## 1. Instrument en bron

- Primair instrument: `XAU_USD`, een XAU/USD-spotreferentie.
- Rekeneenheid: Amerikaanse dollar per troy ounce goud.
- Historische MVP-bron: HistData.com, Generic ASCII M1, symbool `XAUUSD`.
- Prijszijde: bid-OHLC. De M1-bron bevat geen ask-prijzen en geen bruikbaar handelsvolume.
- Brontijdzone: vaste EST-offset `UTC-05:00`, uitdrukkelijk zonder zomertijdaanpassing.
- Normalisatietijdzone: UTC.
- Brontimestamp: wordt als candle-open geïnterpreteerd en vóór modellering met een golden sample gecontroleerd.
- Markturen: feitelijk aanwezige bronrecords zijn leidend. Weekenden, onderhoud, feestdagen en onverwachte gaten worden gerapporteerd en nooit geïnterpoleerd.
- Gebruik: uitsluitend lokale, persoonlijke en niet-commerciële research. Ruwe brondata wordt niet herverdeeld.

De provider zit achter een adaptercontract. Een latere brokerfeed moet dezelfde genormaliseerde candlekolommen leveren en mag worden toegevoegd zonder feature-, label- of modelcode te herschrijven.

## 2. Tijdcontract

Alle intervallen zijn links-gesloten en rechts-open: `[start, end)`.

- Raw timeframe: `1min`; `1mo` betekent steeds één kalendermaand.
- Afgeleide MVP-timeframes: `3min` en `15min`.
- Prediction cadence: iedere 3 minuten op het UTC-raster.
- Feature cutoff: `t`, direct na sluiting van de laatste volledig afgesloten 1min-candle `[t-1min, t)`.
- Prediction time: `t`.
- Toegelaten informatie: uitsluitend waarden met `available_at <= t`.
- Beslissingslatency: 1 minuut, omdat de bron geen uitvoerbare intraminutequote na `t` bevat.
- Entry time: bid-open van de 1min-candle op `t+1min`; deze ligt strikt na prediction time.
- Primaire horizon: 15 minuten vanaf entry.
- Exit time: bid-open op `t+16min`.
- Een sample vervalt wanneer de entry- of exitcandle ontbreekt of wanneer de horizon een niet-toegelaten datagat kruist.

## 3. Return en labels

De continue directionele referentiereturn wordt in basispunten berekend:

```text
gross_return_bps = 10_000 * ln(bid_open(t+16min) / bid_open(t+1min))
```

Het vaste MVP-kostenmodel is 4 basispunten round-trip. De afzonderlijke ruisbuffer is 2 basispunten. De totale labelbuffer bedraagt dus 6 basispunten:

```text
up       als gross_return_bps >  6
down     als gross_return_bps < -6
neutral  anders, inclusief exact -6 en +6
```

De continue return blijft naast het klasselabel bewaard. De buffer wordt niet achteraf aangepast om testresultaten te verbeteren.

## 4. Kosten- en uitvoeringsmodel

Omdat de M1-bron alleen bid-OHLC bevat, gebruikt de MVP een expliciete spreadproxy:

- spread: 3 basispunten round-trip;
- slippage: 0,5 basispunt per zijde, dus 1 basispunt round-trip;
- commissie: 0 basispunten;
- financiering: 0 voor een positie van 15 minuten;
- totale basiskosten: 4 basispunten round-trip;
- kostenstress: 5,5 basispunten round-trip, door de spreadproxy met 1,5 te vermenigvuldigen.

De backtest rapporteert bruto én netto basispunten. Positiegrootte is voor de MVP genormaliseerd op `1x` notional; leverage en geldbedragen worden nog niet gesimuleerd. Wanneer later bid/ask- of brokerdata beschikbaar is, vervangt een nieuwe protocolversie deze proxy.

## 5. Historische perioden

Alle grenzen zijn UTC en de eindgrens is exclusief.

| Blok | Begin | Einde | Gebruik |
|---|---|---|---|
| Train | 2020-01-01 | 2023-01-01 | Fit van model en preprocessors |
| Validation | 2023-01-01 | 2024-01-01 | Beperkte model- en drempelkeuzes |
| MVP-test | 2024-01-01 | 2025-01-01 | Eenmalige vergelijking voor `v0.1` |
| Finale holdout | 2025-01-01 | 2026-01-01 | Alleen fase 14, niet voor gewone ontwikkeling |

De normale ontwikkelconfig weigert records op of na `2025-01-01T00:00:00Z`. De finale holdout krijgt geen normaal CLI-pad en wordt tijdens de MVP niet gedownload. Data uit 2026 valt eveneens buiten de huidige MVP-dataset.

Rond iedere splitgrens geldt minimaal 16 minuten purging/embargo: 1 minuut entrylatency plus 15 minuten labelhorizon. Preprocessors worden uitsluitend op train gefit.

## 6. Meetcontract

Primaire metrics:

- richting: macro-F1;
- probability quality: multiclass log loss;
- economische simulatie: netto basispunten na de vastgelegde kosten.

Verplichte contextmetrics:

- accuracy en balanced accuracy;
- confusion matrix en klassedekking;
- multiclass Brier score en eenvoudige reliability-buckets;
- aantal signalen, hit rate, gemiddelde netto basispunten per trade;
- cumulatief nettoresultaat, turnover en maximale drawdown;
- dezelfde records en perioden voor alle baselines en het getrainde model.

Confidence- of handelsdrempels mogen alleen met validationdata worden gekozen. De testset beïnvloedt training, preprocessing, featureselectie of modelselectie niet.

## 7. Technische succesdefinitie voor `v0.1`

De MVP is technisch geslaagd wanneer:

1. één lokaal commando de keten data -> validatie -> features -> labels -> training -> evaluatie -> backtest uitvoert;
2. identieke data, configuratie en seed identieke kernresultaten geven;
3. ruwe bestanden immutable zijn en alle belangrijke artefacten hashes en versies hebben;
4. alle temporele, leakage-, resampling- en save/load-tests slagen;
5. naïeve baselines en multinomiale logistische regressie op exact dezelfde testrecords staan;
6. een modelartefact opnieuw geladen kan worden en dezelfde probabilities produceert;
7. de backtest nooit vóór prediction time uitvoert en bruto/nettoresultaten toont;
8. beperkingen, gaten en dataherkomst auditbaar zijn.

`v0.1` hoeft nog niet winstgevend te zijn en hoeft de baselines nog niet te verslaan. Vanaf `v0.2` zijn meerdere tijdsblokken en sterkere benchmarkregels verplicht. Vanaf `v1.0` gelden challenger/champion-promotie, stresstests en de eenmalige finale holdout. Paper trading vereist append-only voorspellingen, live datakwaliteit en operationele monitoring.

## 8. Handmatig narekenbaar tijdsvoorbeeld

Stel dat de laatste inputcandle in de bron opent op `2024-06-03 10:02:00` vaste EST. Na omzetting opent die op `15:02:00Z`, bestrijkt `[15:02, 15:03)` en is beschikbaar op `15:03:00Z`.

- feature cutoff en prediction time: `15:03:00Z`;
- entry: bid-open op `15:04:00Z`;
- exit: bid-open op `15:19:00Z`;
- bij entry `2350,00` en exit `2351,65` is de gross return ongeveer `7,02` basispunten;
- `7,02 > 6`, dus het label is `up`;
- na 4 basispunten basiskosten blijft ongeveer `3,02` basispunten netto over.

Geen prijs vanaf `15:03:00Z` of later mag in de features van deze voorspelling voorkomen.

## 9. Bewust uitgesteld

- echte brokerkeuze en live orders;
- bid/ask-tickdata en werkelijke brokerkosten;
- externe markten, macrodata en events;
- neurale modellen, future-path-output, calibration, ensemble en meta-labeling;
- financieel risicobudget, leverage en ordergrootte;
- commercieel gebruik of distributie van brondata.

