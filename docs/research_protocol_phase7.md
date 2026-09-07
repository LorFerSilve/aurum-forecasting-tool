# Researchprotocol — fase 7, richer price-only features

**Protocol:** `phase7-v2`

## Status

`phase7-v2` vervangt de niet-voltooide `phase7-v1`-implementatie nadat een pre-benchmark
coveragefout aantoonde dat de doorsnede van alle hogere-timeframefeatures een geldige
inner fold volledig kon leegtrekken. Er was geen voltooide phase-7-benchmark en er is
geen resultaat gebruikt om deze wijziging te kiezen.

Dit protocol is het bevroren ontwikkelingscontract dat voor de geslaagde fase-7-run is gebruikt. De finale holdout vanaf
2025-01-01 UTC blijft gesloten. Fase 7 mag uitsluitend de reeds toegelaten
ontwikkelingsperiode 2020–2024 gebruiken.

## Hypothese

De fase-6-benchmark gebruikt slechts de achttien causale MVP-features. Fase 7 test of
meer lokale marktstructuur uit dezelfde historische price-only bron de directionele
skill of robuustheid verbetert zonder de evaluatiemethodologie te versoepelen.

## Bevroren evaluatiecontract

Fase 7 hergebruikt de fase-6-methodologie:

- outer walk-forwardjaren 2022, 2023 en 2024;
- dezelfde inner validatieblokken;
- dezelfde purge en 181-minuten gap;
- calibrationblok blijft gereserveerd en ongebruikt;
- dezelfde horizons 3, 6, 9, 12, 15, 30, 60 en 180 minuten;
- dezelfde labels, neutral zone en execution timing;
- dezelfde logistische, ridge- en XGBoost-families;
- preprocessing uitsluitend op train;
- dezelfde policyselectie en base/stress execution costs;
- geen thresholdselectie op outer data.

Hierdoor is de featurelaag de primaire veranderende variabele.

## Gemeenschappelijk sample-universe

Alle cumulatieve feature-ablaties worden geëvalueerd op exact dezelfde causale
MVP/3min prediction rows. Beschikbaarheid van een rijkere of tragere timeframe mag
dus geen rijen uit een eenvoudiger variant verwijderen. Een ontbrekende phase-7
featurewaarde blijft expliciet missing en wordt pas in de modelpipeline met de
bestaande **train-only mediaan-imputer** behandeld. Validatie- of outer data bepalen
nooit de imputatiewaarde. Dit voorkomt dat coverage een featureverbetering nabootst
en voorkomt tegelijk dat een tijdelijk sparse 1h/3h-segment een volledig kwartaal
uit alle ablations verwijdert.

Voor iedere daadwerkelijk geobserveerde extern-timeframefeature geldt:

```text
source_candle_close <= prediction_time
```

en een laatst gesloten candle wordt niet onbeperkt over een datagat vooruit gedragen.
De toegestane age is maximaal één bron-timeframe onder de standaardconfiguratie.

## Featuregroepen en ablations

De vaste volgorde is:

1. `mvp` — exact de bestaande MVP-features;
2. `price_3min` — uitgebreidere returns, candles, momentum, volatility, SMA-distance,
   price-zscore en breakoutcontext op 3min;
3. `price_session_3min` — voegt vaste UTC-sessie- en kalendercontext toe;
4. `price_session_regime_3min` — voegt causale volatility-, trend- en shockregimes toe;
5. `price_session_regime_multitimeframe` — voegt 1min, 5min, 15min, 30min, 1h en 3h toe.

Lookbacks worden per timeframe begrensd. Vooral voor 1h/3h voorkomt dit dat de
observed-row kalender na ieder weekend vrijwel de hele volgende week als
onvoldoende historiek behandelt.

## Microstructure-beperking

De huidige HistData-bron is bid-only en levert geen betrouwbare historische ask,
spread of tick count. Daarom zijn de volgende roadmapfeatures expliciet niet
toegestaan in phase7-v2:

- absolute/historische spread;
- ask- en mid-return;
- tick count en range-per-tick;
- echte quote-staleness of liquidity bucket.

Deze waarden worden niet gesimuleerd of uit bid-OHLC teruggeraden. Ze vereisen een
nieuwe point-in-time databron en een apart challengerexperiment.

## Regimes

Volatility buckets gebruiken uitsluitend de trailing verdeling vóór de huidige
observatie. Trend score gebruikt momentum gedeeld door gerealiseerde volatiliteit.
Shock score vergelijkt de huidige absolute return met de mediaan van voorafgaande
absolute returns. Geen regime gebruikt toekomstige labels of outer-foldstatistieken.

## Verificatie

Voor promotie moet minimaal worden gecontroleerd:

- future mutation verandert oudere features niet;
- batch en single-timestamp/online berekening zijn exact gelijk;
- hogere timeframes gebruiken nooit een nog open candle;
- phase-7 features bevatten nooit +/-inf; missing waarden blijven expliciet tot
  train-only preprocessing;
- imputatie en scaling worden uitsluitend op de betreffende trainingfold gefit;
- iedere trainingfold heeft voor iedere gebruikte feature minimaal één geobserveerde
  waarde;
- featuredistributies inclusief missing coverage worden per outer fold opgeslagen;
- alle ablations gebruiken dezelfde sample IDs;
- de finale holdout blijft gesloten;
- de formele benchmark draait alleen vanaf een schone, gecommitte Git-state.

## Interpretatie en promotie

Een predictive challenger moet tegenover `mvp/reference` minimaal:

- betere gemiddelde macro-F1 hebben;
- geen slechtere worst-fold macro-F1 hebben;
- geen slechtere gemiddelde Brier-score hebben.

Economische promotie vereist daarnaast de bestaande fase-6 coverage- en
kostencriteria. Een predictive featureverbetering activeert nooit automatisch een
tradingmodel. Probabilities blijven ongekalibreerd tot fase 11.

Als geen featuregroep robuust verbetert, is `keep_mvp_features` een geldige
phase-7-uitkomst. Complexiteit is geen exitcriterium.
