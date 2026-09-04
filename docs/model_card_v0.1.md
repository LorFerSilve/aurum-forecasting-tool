# Model card — XAU/USD research-MVP v0.1.0

Status: **research benchmark, niet geschikt voor tradinginzet**  
Datum: 2026-09-04  
Modelversie: `v0.1.0+09e5c67a0f9d`

## Model en bedoeld gebruik

Dit is een multinomiale logistische regressie die na iedere afgesloten `3min`-candle de
XAU/USD-richting over een uitvoerbare horizon van 15 minuten classificeert als `down`,
`neutral` of `up`. Het doel is de reproduceerbare onderzoeks- en evaluatieketen als eerste
champion vast te leggen. Het model is niet bedoeld voor financieel advies, paper trading,
echte orders, leverage, positiegrootte of een rendementsclaim.

## Data en tijdcontract

- Publieke HistData Generic ASCII M1 bid-OHLC, 2020–2024.
- Vaste bronoffset `UTC-05:00` zonder DST, genormaliseerd naar UTC.
- Features beschikbaar op prediction time `t`.
- Entry: bid-open op `t+1min`; exit: bid-open op `t+16min`.
- Label: `down` onder `-6 bp`, `up` boven `+6 bp`, anders `neutral`.
- Train: 2020–2022; validation: 2023; test: 2024.
- 16 minuten purge/embargo; geen random shuffle.
- Finale holdout vanaf 2025 is niet gedownload en wordt door gewone ontwikkelconfig geweigerd.

De bron heeft geen ask-OHLC en geen bruikbaar volume. Het kostenmodel gebruikt daarom een
expliciete spreadproxy en kan echte uitvoeringskosten niet bewijzen.

## Inputs

Achttien causale price-only features op `3min`: recente log-return en drie lags, candle
body/range/wicks/closepositie, momentum 5/20, realized volatility 20, afstand tot SMA 5/20,
en cyclische UTC-uur- en weekdagvelden. Rolling windows lopen nooit over datagaten.

De train-only median-imputer en standaardisatie worden samen met de exacte featurevolgorde
in de modelbundle opgeslagen.

## Training en selectie

- Random seed: `20260904`.
- Kandidaten: `C={0.1, 1, 10}` en class weight `{none, balanced}`.
- Primaire selectiemetric: validation macro-F1; log loss is tiebreaker.
- Geselecteerd: `C=0.1`, `class_weight=balanced`.
- Final fit: uitsluitend 331.853 trainrecords.
- Validation: 83.054 records; test: 111.098 records.

De testset beïnvloedt preprocessing, fit of selectie niet.

## Testprestatie

| Metric | Waarde |
|---|---:|
| Dekking | 1,000000 |
| Accuracy | 0,571738 |
| Balanced accuracy | 0,422887 |
| Macro-F1 | 0,424344 |
| Multiclass log loss | 0,999740 |
| Multiclass Brier | 0,590026 |

De beste eenvoudige price-only baseline op macro-F1 was mean reversion met `0,387768`.
De probabilities zijn niet professioneel gekalibreerd. Reliability-buckets tonen duidelijke
afwijkingen tussen gemiddelde confidence en empirische accuracy; ze mogen niet als zekere
kansen worden geïnterpreteerd.

## Economische simulatie

Bij confidence `>=0,50`, één niet-overlappende positie en 4 bp round-tripkosten werden 527
trades uitgevoerd. Het resultaat was `-665,10 bp` bruto en `-2.773,10 bp` netto, met
`2.851,21 bp` maximale drawdown. Bij 5,5 bp kostenstress daalde netto naar `-3.563,60 bp`.

De strategie is dus niet economisch inzetbaar. De negatieve uitkomst is bewaard en niet
weggetuned op de testset.

## Versies en reproduceerbaarheid

- Modeltabel:
  `sha256:d5ad6e63676b19d9f80c84c31f2f643261b314962e0db11368a937cf7a5e6453`.
- Featuredefinitie:
  `sha256:122383f4ea2e0a1467e2c077a599b70de6aa2a7c279dbb44fae30b78bfe856b8`.
- Labeldefinitie:
  `sha256:694c864c76ae77584fa77f1c2ccb452bd31456104ce80dacdcc9df09fb157303`.
- Splitdefinitie:
  `sha256:086d143e56e77916c132eca5584ae1b3ba5c8eb3f8179901a8975ce673d5eac8`.
- Train-only preprocessor:
  `sha256:7462aead3a9f4faa096e210d131d29f085dfe43ae5d5dae884ac350add46025c`.
- Volledige rootconfig:
  `sha256:5038ed747d7fc20ccdeba86c81fa3153ef9b611824a2cdd557e4939efcd59e5e`.
- Modelconfiguratiesnapshot:
  `sha256:c55b1b2aa3fcf3b3f780b7ebfe45fa84e34ef2a1d86220593992e2741bc2e8fe`.
- Modelbundle SHA-256:
  `b2f098aac94f2ece083926b433e58258e30f571bb3f8e9782893737fe6be73c4`.
- Reproduceerbaarheidsruns:
  `20260904T154809220237Z-a2b9c042` en `20260904T155009507615Z-73592cb0`.
- Releasecode: Git-tag `v0.1.0`; het exacte commit-ID is opvraagbaar met
  `git rev-list -n 1 v0.1.0`.

Twee volledige echte runs gaven negentien byte-identieke inhoudelijke runartefacten en
dezelfde voorbeeldkansen. Alleen het runmanifest verschilt door run-ID en UTC-tijden.

## Belangrijkste beperkingen en risico's

- Eén gratis bid-only databron; bronfouten, ontbrekende candles en providerbias zijn mogelijk.
- Geen werkelijke bid/askspread, tickdata, brokerlatency, commissie, financiering of slippage.
- Eén vaste split; nog geen walk-forwardvalidatie of onafhankelijke finale holdout.
- Alleen eenvoudige prijsfeatures; geen robuuste regime-, multi-timeframe- of externe context.
- Confidence-threshold `0,50` is configureerbaar maar niet als winstoptimum op test getuned.
- Geen calibrated probabilities, OOD-detectie, risicomotor, monitoring of kill switch.
- Historische simulatie garandeert geen toekomstige prestatie.

## Besluit en volgende poort

De softwareketen is als `v0.1.0` geslaagd; het model blijft de eerste researchchampion en
fallbackbenchmark. Tradinginzet is gestopt. Een challenger mag dit model pas vervangen onder
dezelfde labels, splits, kosten en vergelijkbare tuningbudgetten, met later walk-forward- en
holdoutbewijs volgens de roadmap.
