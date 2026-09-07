# Model card — v0.2 price-only researchbenchmark

## Status

- Release: `v0.2.0`
- Fasen: 5–7 afgerond
- Protocol: `phase7-v2`
- Empirische run: `20260907T014255645674Z-b2afe281`
- Tradingchampion: **geen**
- Finale holdout vanaf 2025-01-01 UTC: **gesloten**

Deze release is een betrouwbare researchbenchmark en geen paper- of live-tradingmodel.

## Data en evaluatie

De benchmark gebruikt HistData XAU/USD bid-only minute data uit 2020–2024. Curated
timeframes lopen van 1min tot 3h; 1d en 1mo blijven buiten de modelinput zolang hun
volledigheid zonder vertrouwde markt-/holidaykalender niet kan worden bewezen.

Evaluatie gebruikt de fase-6 nested walk-forwardopzet met outer testjaren 2022, 2023
en 2024, een 181-minuten training-side gap, gereserveerde calibratieblokken, train-only
preprocessing en dezelfde base/stress execution assumptions.

## Bevroren researchchampions

| Horizon | Featurevariant | Model-family uit phase7 run |
|---:|---|---|
| 3m | `price_session_regime_multitimeframe` | `logistic` |
| 6m | `price_session_regime_3min` | `logistic` |
| 9m | `price_session_regime_multitimeframe` | `logistic` |
| 12m | `price_session_regime_multitimeframe` | `reference` |
| 15m | `mvp` | `logistic` |
| 30m | `price_session_regime_multitimeframe` | `logistic` |
| 60m | `price_3min` | `reference` |
| 180m | `price_3min` | `logistic` |

Een vaste-reference attributiecontrole kiest op alle horizons dezelfde featurevariant.
De featureconclusie is dus niet alleen het gevolg van model-familyselectie.

## Predictive resultaat

Het sterkste consistente bewijs zit op 3m, 30m en 60m: de geselecteerde featurevariant
verbetert macro-F1 in elk van de drie outer jaren en passeert tegelijk de vooraf
vastgelegde worst-fold- en Brier-poort. 6m, 9m, 12m en 180m passeren de formele poort
met kleinere en gemengde fold-delta's. 15m behoudt de eenvoudige MVP-features.

Zie `docs/phase7_verification.md` voor exacte delta's per horizon en fold.

## Economische status

Geen getrainde model-family selecteerde een actieve policy. Voor de getrainde
`reference`, `logistic`, `ridge` en `xgboost`-aggregaten zijn mean en worst-fold
trade counts 0. Er zijn 0 economic promotion candidates.

Daarom:

- geen paper-tradingpromotie;
- geen live-tradingpromotie;
- geen rendementsclaim;
- de price-only champions zijn uitsluitend research fallbacks voor volgende challengers.

## Featurecoverage

De fase-7-catalogus bevat 171 features op 534.698 MVP-geankerde prediction rows.
Ongeveer 19,15% van de volledige featurematrix is missing. Beschikbaarheid is 100%
voor 1min/3min, 96,2% voor 5min, 83,4% voor 15min, 72,1% voor 30min, 59,9% voor 1h
en 15,9% voor 3h.

Missing waarden worden niet geforwardfilld. De modelpipeline gebruikt uitsluitend
train-only mediaanimputatie.

## Belangrijkste beperkingen

- De historische bron is bid-only: echte ask, spread en betrouwbare tick count ontbreken.
- Microstructurefeatures zijn niet gesynthetiseerd.
- 1h en vooral 3h zijn sparse; de volledige multi-timeframevariant leunt dus sterk op
  train-only imputatie.
- Probabilities zijn niet gekalibreerd; volledige OOF-calibratie is fase 11.
- Phase7-v2 gebruikt vooraf vastgelegde mean/worst-fold/Brier-gates maar geen aparte
  formele significantietest.
- De finale 2025+ holdout mag pas in fase 14 worden geopend.

## Intended use

Gebruik v0.2 als:

- leakage-bewuste price-only benchmark;
- fallback waarmee fase-8+ challengers op dezelfde development folds worden vergeleken;
- reproduceerbaar referentiepunt voor feature- en modelonderzoek.

Gebruik v0.2 niet voor:

- echte orders;
- paper-tradingclaims;
- position sizing;
- calibrated probability claims;
- rendementsgaranties.
