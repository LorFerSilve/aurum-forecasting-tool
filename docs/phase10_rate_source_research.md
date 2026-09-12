# Phase 10 bron 3 — 10-year TIPS real yield (DFII10)

**Status:** bronkeuze en exploratory contract bevroren vóór een echte rate-benchmark.  
**Exploratory protocol:** `phase10-rate-dfii10-modeled-v1`  
**Series:** FRED `DFII10`  
**Strict vervolg indien nuttig:** ALFRED DFII10-vintages.

## Waarom deze renteproxy

De derde Phase-10 bron test één rentehypothese bovenop de frozen Phase-7 15m
`mvp/logistic` reference. We kiezen de **10-jarige Amerikaanse TIPS real yield** in plaats
van een nominale Treasury yield. Voor goud is de reële rente economisch de directere
opportunity-cost proxy: zij combineert de nominale rentestand met de door de markt
ingeprijsde inflatiecomponent zonder in deze fase meerdere rentebronnen tegelijk toe te voegen.

De concrete serie is:

- FRED: `DFII10` — *Market Yield on U.S. Treasury Securities at 10-Year Constant
  Maturity, Quoted on an Investment Basis, Inflation-Indexed*;
- source: Board of Governors of the Federal Reserve System;
- release: H.15 Selected Interest Rates;
- units: percent, daily, not seasonally adjusted;
- source page: https://fred.stlouisfed.org/series/DFII10

De Federal Reserve H.15-pagina vermeldt dat de release maandag t/m vrijdag om **16:15 ET**
wordt geplaatst, behalve op holidays/Board closures:
https://www.federalreserve.gov/releases/h15/

Een actuele release toont bovendien de gebruikelijke één-businessdag-lag: de H.15-release
van 2026-09-09 bevat observaties tot 2026-09-08. Dat ondersteunt de exploratory
availability-regel, maar is geen bewijs dat iedere historische rij exact volgens hetzelfde
schema beschikbaar kwam.

## Waarom de eerste run modeled-latency blijft

Een gewone FRED-historie-download geeft de **huidige** historische waarden. FRED vermeldt
dat series aan revisies onderhevig kunnen zijn. Zo'n snapshot bevat niet per rij het exacte
historische release- of revisiemoment. Daarom mag de gewone FRED-CSV geen strict-PIT claim
maken.

Voor `phase10-rate-dfii10-modeled-v1` geldt daarom:

- `availability_basis=modeled_latency`;
- observation/reference time = de DFII10 referentiedatum om `00:00 UTC`;
- modeled availability = de eerstvolgende reguliere U.S. federal business day om
  `16:15 America/New_York`;
- timezone is DST-aware;
- een `.`/lege FRED-waarde betekent **geen nieuwe rate-observation** en wordt overgeslagen;
- de vorige geldige observatie mag uitsluitend via de gewone backward as-of join blijven
  gelden;
- stale threshold = 5 dagen (`432000` seconden), zodat normale weekends/holiday weekends
  kunnen worden overbrugd maar langere gaten fail-closed naar price-only gaan;
- de actuele project-downloadtijd wordt apart gehasht/opgeslagen en bewijst geen historische
  availability;
- geen observatie uit 2025+ wordt toegelaten.

Deze availability is bewust conservatief en benadert een H.15-publicatiekalender. Ad-hoc
Board closures, vertraagde publicaties en historische correcties zijn niet per rij bewezen.
Daarom kan deze run nooit champion promotion of trading activation veroorzaken.

## Strict-PIT vervolg: ALFRED

Voor DFII10 is een sterkere bron beschikbaar als de exploratory screening werkelijk waarde
toont. ALFRED biedt historische vintages voor DFII10:
https://alfred.stlouisfed.org/series/downloaddata?seid=DFII10

ALFRED documenteert het outputformaat **Observations by Real-Time Period** met:

- `observation_date`;
- de seriewaarde;
- `realtime_start_date` — eerste vintage/publicatiedatum van die versie;
- `realtime_end_date` — laatste datum waarop die versie current was.

Zie https://alfred.stlouisfed.org/help/downloaddata.

Een eventuele strict follow-up moet die vintage-evidence afzonderlijk importeren en
cryptografisch authenticeren. Om intraday look-ahead uit alleen een datumveld te vermijden,
kan een strict adapter een revision conservatief pas na het einde van haar
`realtime_start_date` toelaten, tenzij een historische provider timestamp met hogere
precisie beschikbaar is. Dit wordt **niet** stilzwijgend in de exploratory run gemengd.

## Frozen rate-featurehypothese

DFII10 is een yield in procenten, geen prijs. Daarom gebruiken we **additieve yieldchanges**
en geen logreturns. De modelinputs zijn vóór de echte marktuitkomst vastgelegd:

| Feature | Definitie |
|---|---|
| `rate_level_pct` | laatst beschikbare geldige DFII10-level in procent |
| `rate_change_1obs_bps` | verschil tegenover vorige distincte beschikbare DFII10-observatie × 100 |
| `rate_change_5obs_bps` | verschil over vijf distincte beschikbare observaties × 100 |
| `rate_change_20obs_bps` | verschil over twintig distincte beschikbare observaties × 100 |
| `rate_age_seconds` | prediction cutoff minus referentietijd van de gebruikte observatie |

`rate_is_missing` en `rate_is_stale` zijn uitsluitend routing/auditvelden. Alle vijf
modelkolommen moeten eindig zijn; anders gebruikt die gold row exact de persisted Phase-7
price-only probabilities. Een dagelijks DFII10-level wordt op vele 3min gold rows herhaald;
dat mag **geen kunstmatige nul-yieldchange** produceren. De laatst berekende 1/5/20-observation
change blijft onveranderd tot een nieuw rate-observation beschikbaar komt.

## Zelfde admissioncontract als eerdere Phase-10 bronnen

De ratebron verandert niets aan:

- de frozen Phase-7 15m `mvp/logistic` reference;
- exact hetzelfde gold sample universe;
- outer folds 2022/2023/2024;
- 181 minuten train-side gap/purge;
- logistic `C={0.1,1.0,10.0}` en `class_weight="balanced"`;
- train-only preprocessing;
- inner-only model- en policyselectie;
- predictive gate: hogere mean macro-F1, worst-fold macro-F1 niet slechter en mean Brier
  niet slechter;
- dezelfde economic gate;
- exacte Phase-7 fallback bij missing/stale/onvoldoende history;
- `holdout_opened=false`.

Silver en inverse EURUSD zijn beide reeds met `stop` afgesloten en worden **niet** met de
rateproxy gecombineerd. Dit is opnieuw een onafhankelijke bronablation.

## Lokale acquisitie — nog geen benchmark openen

Download uitsluitend de developmentperiode **2020-01-01 t/m 2024-12-31** van FRED DFII10.
Een eenvoudige FRED graph-CSV kan via de DFII10-pagina worden geëxporteerd; controleer dat
het bestand slechts twee kolommen bevat (`observation_date` of `DATE`, en `DFII10`) en geen
2025+ rij bevat.

Bewaar/hernom het lokale bestand exact als:

```text
data/raw/phase10/rate/DFII10_2020_2024.csv
```

De importer downloadt niets en weigert een andere bestandsnaam, out-of-development dates,
onverwachte kolommen, niet-numerieke values, ongeldige yields of een sourceconfig die
strict-PIT pretendeert.

Na acquisitie wordt eerst alleen geïmporteerd en gepreflight. De real-data benchmark blijft
gesloten totdat de preflight volledig is gereviewd.
