# Verificatie fase 3 — features, labels en datasplits

Datum: 2026-09-04  
Uitkomst: **geslaagd**

## Gebouwde artifacts

| Artifact | Rijen | Versie of eigenschap |
|---|---:|---|
| Causale `3min`-features | 534.698 | 18 features |
| Volledige `15min`-labels | 526.005 | 8.693 featuretijden vervielen door een ontbrekend toekomstpad |
| Modeltabel | 526.005 | `sha256:d5ad6e63676b19d9f80c84c31f2f643261b314962e0db11368a937cf7a5e6453` |
| Train | 331.853 | 2020–2022 |
| Validation | 83.054 | 2023 |
| Test | 111.098 | 2024 |

De achttien features omvatten return en drie lags, candle body/range/wicks/closepositie,
momentum over 5 en 20 candles, realized volatility over 20 candles, afstand tot SMA 5 en 20,
en cyclische UTC-uur- en weekdagvelden. Tick count is bewust niet gebruikt omdat de bron die
niet betrouwbaar levert.

## Temporele en leakagecontroles

- [x] Features gebruiken uitsluitend complete `3min`-candles en resetten rolling windows na
  ieder datagat.
- [x] `feature_available_at_utc <= prediction_time_utc` voor ieder sample; in de MVP zijn ze
  exact gelijk.
- [x] Entry is exact `prediction_time + 1min`; label-einde is exact
  `prediction_time + 16min` en het volledige `1min`-pad moet aaneengesloten zijn.
- [x] Een mutatietest verandert toekomstige candles en bewijst dat oudere features gelijk
  blijven.
- [x] De splits zijn chronologisch, zonder random shuffle, met 16 minuten purge/embargo rond
  de grenzen.
- [x] De median-imputer en standaardisatie zijn uitsluitend op 331.853 trainrecords gefit en
  ongewijzigd op validation/test toegepast.
- [x] Sample-index, splitdefinitie, featureschema, labeldefinitie, preprocessor en upstream
  datasets zijn alle gehasht en streng tegen hun manifests gevalideerd.
- [x] Verouderde of gemuteerde config, lineage, artifacts of preprocessor worden geweigerd.

De train-only preprocessor heeft versie
`sha256:7462aead3a9f4faa096e210d131d29f085dfe43ae5d5dae884ac350add46025c`.
Twee echte herbouwen leverden dezelfde modeltabelversie en dezelfde sample-index op.

## Handmatige trace

Sample `XAU_USD|histdata|20210215T135700.000000Z` is van de raw 2021-ZIP via de afgesloten
`3min`-candle en alle achttien features naar entry `13:58Z`, exit `14:13Z`, return
`-1,815311 bp` en klasse `neutral` teruggevolgd. De raw hash, bronversies en alle relevante
tijden staan in de modeltabel.

## Besluit

`promote`: fase 3 levert een volledig herbouwbare, leakage-bewuste modeltabel; fase 4 mag
zonder handmatige data-aanpassingen trainen.
