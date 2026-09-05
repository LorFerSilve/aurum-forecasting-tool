# Verificatie fase 5 — geharde primaire data

Datum: 2026-09-05. Status: **technisch geverifieerd; dag- en maanddata nog niet bruikbaar**.

De publieke HistData-historiek 2020–2024 is via het nieuwe incrementele contract tweemaal
opgebouwd. Het profiel blijft bij dezelfde ontwikkelingsperiode en opent de finale holdout
niet. Een volledige update duurde op deze machine ongeveer anderhalve minuut; de exacte
tijden van de laatste twee herbouws staan in het JSON-verificatierapport. Dit zijn
waarnemingen, geen prestatiegaranties.

## Reproduceerbaarheid en dekking

Commando:

```powershell
.\.venv\Scripts\python.exe scripts/verify_phase5.py --include-mvp
```

Alle 45 curated Parquet-bestanden en de zeven kwaliteits-/dekkingsrapporten hebben bij
beide herbouws dezelfde inhoudshashes. Het als laatste gepubliceerde `phase5_build.json`
is byte-identiek. De oorspronkelijke vier MVP-datamanifests zijn ongewijzigd gebleven.
De validatie controleert iedere bijbehorende input/outputhash en de volledige lineage.

| Timeframe | Volledige candles |
|---|---:|
| 1min | 1.726.589 |
| 3min | 575.254 |
| 5min | 345.000 |
| 15min | 114.762 |
| 30min | 57.215 |
| 1h | 28.463 |
| 3h | 8.402 |
| 1d | 0 |
| 1mo | 0 |

De kwaliteitslaag bevat 1.827 dagen per timeframe, inclusief geheel lege datasets.
Bronloze intervallen zijn gemarkeerd en niet ingevuld. De oorspronkelijke 1min-gapdetectie
ziet 2.220 interne gaps; de nieuwe dagelijkse audit maakt ook dagranden zichtbaar en
onderscheidt 2.766 onbekende gapsegmenten. Deze tellingen hebben verschillende definities.
Binnen de geobserveerde M1-dagdekking ontbreken 101.230 minuten. Zonder betrouwbare
marktkalender kan niet worden beweerd dat dit allemaal providerstoringen zijn.

Alleen aantoonbaar volledige UTC-vensters worden modelinput. HistData M1 heeft geen in dit
project geverifieerde markturen-/feestdagenkalender. Daarom blijven dag- en maanddatasets
leeg: de huidige bron kan hun volledigheid niet bewijzen. De code ondersteunt beide
timeframes, inclusief schrikkelmaanden en expliciet bewezen marktsluitingen. Dit is geen
claim dat reeds bruikbare historische dag- en maandfeatures beschikbaar zijn.

Geharde M1-versie:
`sha256:ae899269c2f19fe38267bcd069ed951ba08aa1979354eab694cbee9be0e408cd`.
Alle datasetversies en meetwaarden staan in
[phase5_data_verification.json](phase5_data_verification.json).

## Volledige MVP blijft gelijk

Geslaagde runs:

- Gehard profiel: `20260905T215108619982Z-79ae0e0e`.
- Oorspronkelijke MVP: `20260905T215234405218Z-7c0c7334`.

Zeven inhoudelijke artefacten zijn byte-identiek tussen beide runs: modelvergelijking in
JSON en CSV, metrics per UTC-uur, reliability-buckets, backtestsamenvatting, alle
handelsbeslissingen en alle trades. De 111.098 testrecords leiden nog steeds tot 527 trades
en -2.773,10 bp cumulatief netto in het basisscenario. De extra timeframes veranderen
de featureselectie niet.

Het geharde profiel heeft een eigen data-/modelversie vanwege de nieuwe manifests en
configuratie. De oorspronkelijke modeltabelversie blijft
`sha256:d5ad6e63676b19d9f80c84c31f2f643261b314962e0db11368a937cf7a5e6453`.
Na de vergelijking staan de gedeelde gegenereerde feature-/modelpaden weer op de
oorspronkelijke MVP. De uitvoerrapporten van beide runs blijven beschikbaar.

Zie [phase5_mvp_verification.json](phase5_mvp_verification.json) voor de individuele hashes.
De ontwikkelruns registreerden de fase-5-werkboom als `c9c09c7...+dirty`; dit is een
geverifieerde ontwikkelstap op de fase-5-branch, geen nieuwe `v0.2`-release.

## Softwareverificatie

- 254 tests geslaagd; Ruff en strict mypy groen (42 bronbestanden).
- Dependencycontrole geslaagd; wheel en source distribution gebouwd.
- Nieuwe CLI-route `data update` en configuratie `configs/phase5.yaml` beschikbaar.
- Offline regressies voor retries, rate limiting, paginering, hervatten en holdoutgrenzen.
- Bestaande raw acquisitietijd wordt met microsecondeprecisie overgenomen; conflicterende
  hashes, bronidentiteit of metadata worden geweigerd zonder bestaande bytes te vervangen.
- UTC-jaargrenzen, gedeeltelijke bronjaarimports, maandgrenzen, schrikkeljaren, ontbrekende
  minuten en onvoltooide vensters zijn getest.
- Reuse weigert een gewijzigde kalenderpolicy, verouderde logicaversies, een onvolledige
  completionrecord of gewijzigde kwaliteitsrapporten.

## Besluit en vervolg

**Datalaag accepteren; huidige modelchampion behouden.** De data-infrastructuur is uitgebreid
en de gebruikte intraday-timeframes zijn reproduceerbaar. Dag-/maandinput blijft uitgesloten
tot kalenderkennis of een toekomstige feed voldoende bewijs levert. De economische uitkomst
blijft negatief; er is geen promotie naar paper trading.

HistData ondersteunt hier gesloten jaararchieven en hervatten op archiefniveau. Het levert
geen live feed, gedeeltelijke HTTP-range-resume of automatische upstream-revisiedetectie.
Ask, mid, spread en betrouwbare tick count blijven afwezig. Het generieke contract maakt
latere adapters mogelijk zonder deze gegevens nu te fabriceren.

De volgende roadmapfase is 6: walk-forwardvalidatie, meerdere betrouwbare intradayhorizons,
sterkere klassieke baselines en een uitgebreidere backtester. `v0.2` volgt pas na fasen 6–7.
Het volledige datacontract staat in [data_contract_phase5.md](../docs/data_contract_phase5.md).
