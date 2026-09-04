# Verificatie fase 2 — minimale XAU/USD-data

Datum: 2026-09-04  
Uitkomst: **geslaagd**

## Bron en contract

- Provider: HistData Generic ASCII M1, symbool `XAUUSD`, jaren 2020–2024.
- Adapter: het providerneutrale `M1Provider`-contract en een registry/factory houden ingestie
  los van features, labels en modellen.
- Prijszijde: bid-OHLC; het volumeveld is aanwezig maar onbetrouwbaar en wordt niet als feature
  gebruikt.
- Brontijd: vaste `UTC-05:00` zonder DST; alle modeldata is genormaliseerd naar UTC.
- Raw ZIP-bestanden blijven ongewijzigd en staan met grootte en SHA-256 in het raw manifest.

## Werkelijke datadekking

| Timeframe | Rijen | Eerste candle | Laatste candle |
|---|---:|---|---|
| `1min` | 1.726.589 | 2020-01-01 23:00Z | 2024-12-31 21:58Z |
| `3min` | 575.254 | 2020-01-01 23:00Z | 2024-12-31 21:57Z |
| `15min` | 114.762 | 2020-01-01 23:00Z | 2024-12-31 21:45Z |

De `1min`-validator vond 2.220 gaten, samen goed voor 902.789 ontbrekende minuten ten
opzichte van een onafgebroken kalender. Deze omvatten normale marktsluitingen en mogelijke
brongaten. Niets is geïnterpoleerd.

Belangrijkste versies:

- curated `1min`: `sha256:b5f7463b066e3c1d86408441c8cbce62f165a4a99867154daa2f5baa84d2963f`;
- curated `3min`: `sha256:2110d9e4fdcdaca867881cdba7a9ec4ffaf802b898c3aecde91cabd1b593102b`;
- curated `15min`: `sha256:2b9e1ec052b0ad92207adbfa37f4c4ab839416972221f3e45a53c4b759e9a1d0`.

## Uitgevoerde controles

- [x] Positieve, finite OHLC-waarden en canonieke high/low-relaties.
- [x] Unieke candlekeys en een expliciet `drop_identical`-beleid voor exacte duplicaten.
- [x] Strikt stijgende timestamps per instrument, bron en timeframe.
- [x] Alleen volledig afgesloten candles in curated data.
- [x] Hogere candles ontstaan alleen uit volledige, aaneengesloten, UTC-uitgelijnde vensters.
- [x] Vijf geldige `3min`-candles vormen exact dezelfde `15min`-OHLC als de vijftien
  onderliggende `1min`-candles.
- [x] Partiële, gapped en incomplete vensters worden verwijderd en niet opgevuld.
- [x] Een herbouw met dezelfde raw inputs behoudt dezelfde contentversies en creëert geen
  duplicaten.
- [x] Alle manifestpaden, hashes, groottes, perioden, jaarpartities en lineage zijn gevalideerd.

## Handmatige brontrace

Voor prediction time `2021-02-15 13:57:00Z` zijn de bronregels `08:54`, `08:55` en `08:56`
vaste EST samengevoegd tot de afgesloten `3min`-candle die opent om `13:54Z`. De entry is de
raw bid-open op `08:58 EST` (`13:58Z`) van `1818,035`; de exit is de raw bid-open op
`09:13 EST` (`14:13Z`) van `1817,705`.

```text
10_000 * ln(1817,705 / 1818,035) = -1,815311 bp
```

Dat valt binnen de vaste grens `[-6, +6]` en krijgt dus label `neutral`. Zowel feature als
label verwijzen terug naar raw SHA-256
`312229c9db5b39138c41b1e488d00ed32e158a0d1fc3f23a225b223c3271c0a4`.

## Besluit

`promote`: de minimale prijsdataset is reproduceerbaar, geversioneerd en geschikt als input
voor fase 3. De datakwaliteitsbeperkingen blijven expliciet modelrisico.
