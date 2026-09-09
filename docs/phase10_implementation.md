# Fase 10 — externe context en eventinformatie

**Status:** technische ontwikkelbasis; nog geen formele marktbenchmark of `v0.3`-release.

**Branch:** `codex/phase10-external-context`

**Proefprotocol:** `phase10-context-smoke-v1` (uitsluitend synthetisch).

Phase 9 is op `main` gemerged in `9a15e04`. De onafhankelijke hercontrole op
2026-09-09 verifieerde de 225 canonieke artefacten en reproduceerde de opgeslagen
path-returnaudit exact. De Phase-7 researchchampions blijven bevroren, inclusief
`mvp/logistic` voor 15m. Er is geen tradingchampion. De finale 2025+ holdout blijft gesloten.

## Uitvoerbare eerste keten

```text
broncontract + CSV met beschikbaarheid/vintages + manifest
-> metadata-only ontwikkelingsguard
-> SHA-256-verificatie van dezelfde bytes die worden ingelezen
-> backward as-of contextkoppeling
-> prijs-/momentum-/correlationfeatures met ontbrekendheid en leeftijd
-> chronologische synthetische price-only/contextvergelijking
-> exacte price-only fallback bij uitval
-> controleerbare artefacten
```

```powershell
.\.venv\Scripts\gold-forecast.exe phase10 preflight
.\.venv\Scripts\gold-forecast.exe phase10 dry-run --output reports/phase10_dry_run
.\.venv\Scripts\gold-forecast.exe phase10 validate reports/phase10_dry_run
```

`preflight` retourneert momenteel bewust exitcode **1**: zilver staat uit en de
historische beschikbaarheid is niet bewezen. De JSON beschrijft de blokkades.
Dit commando downloadt niets. Een uitgeschakelde bron wordt niet geopend, ook
niet wanneer een optioneel bundlepad is meegegeven.

`dry-run` vereist een nieuwe uitvoermap. De run maakt gesimuleerde goud- en
zilverprijzen, publicatiemomenten en bronuitval. Beide modelvarianten krijgen
dezelfde train-/test-sample IDs; preprocessing wordt alleen op train gefit.
Vijftien minuten aan labels wordt aan beide splitgrenzen gepurged. De twee
kleine logistische modellen zijn integratiefixtures; de run vergelijkt niet
met de echte Phase-7 champion en is geen economische ablation.

De run bewaart bronrecords, features, testvoorspellingen, volledig bronloze
fallbackvoorspellingen, metrics en een completionmanifest. `validate` controleert
de vaste bestandsinventaris, hashes en expliciete synthetische scope. Dit is
integriteitsverificatie, geen bewijs dat een databron betrouwbaar is of een
model voorspellende waarde heeft.

## Tijdreekscontract

`ContextSource` beschrijft bronidentiteit, markturen/tijdzone, publicatievertraging,
stalegrens, revisiebeleid, onderbouwing van beschikbaarheid en expliciete opt-in.
Alle bronnen vallen bij afwezigheid terug op de price-only keten.

Een waarneming bevat:

| Veld | Betekenis |
|---|---|
| `source_id`, `observation_id` | Bron en stabiele identiteit van de waarneming |
| `observed_at_utc` | Gesloten marktbar of referentietijd van een macro-observatie |
| `available_at_utc` | Eerste beschikbaarheid van deze specifieke versie |
| `ingested_at_utc` | Werkelijke acquisitietijd van het gebruikte bronbestand |
| `value` | Getal; null is een expliciet ontbrekende waarneming |
| `revision_id` | Identiteit van deze versie |
| `source_uri`, `raw_sha256` | Herkomst van het oorspronkelijke bronartefact |

Corefuncties accepteren uitsluitend expliciet UTC-aware timestamps. Een
CSV-bundle verlangt ISO-tijden met `T` en `Z` of `+00:00`, een exacte kolomvolgorde,
row count, SHA-256 en halfopen observatiegrenzen binnen 2020–2024. Deze grenzen
worden vóór het lezen van de CSV gecontroleerd. De bundlehash authenticeert
de lokale CSV; de upstream hashes in de rijen moeten bij brontoelating nog
tegen de originele bronbestanden worden gecontroleerd.

Per cutoff wordt de recentste waarneming genomen uit de versies die toen
beschikbaar waren. Een late revisie van een oudere periode vervangt geen
nieuwere periode. Een revisie verjongt de waarneming niet:
`age_seconds = prediction_time - observed_at`. Stale waarden blijven zichtbaar
in de auditkolommen, maar worden niet als bruikbare zilverfeatures gebruikt.

Rendementen en momentum zijn logveranderingen in basispunten. De correlation
gebruikt twintig gepaarde rendementen uit 21 opeenvolgende waarnemingen. Niet
alleen prediction time, maar ook de waarnemingstijd van zilver moet vooruitgaan.
Een doorgedragen quote levert dus geen verzonnen nulrendement; na een gat moet
het betreffende venster opnieuw gevuld worden. De prijsratio mag een nog verse
as-of waarde gebruiken, met zichtbare leeftijd.

## Eventkalender

`join_events` verwerkt CPI, Amerikaanse arbeidsmarktdata en FOMC als geplande
events met `scheduled_at_utc`, een afzonderlijke `available_at_utc`, versienummer
en bronhash. Een wijziging of annulering werkt uitsluitend vanaf haar eigen
beschikbaarheidsmoment. Next/previous-eventvelden bewaren de gebruikte versie
en herkomst; een latere wijziging herschrijft geen eerdere prediction rows.

De standaardvensters zijn 60 minuten vóór het event, de eerste 5 minuten vanaf
het geplande tijdstip en daarna tot en met 60 minuten na het event. Bij overlap
geldt de prioriteit `during > before > after > normal`. De afstand tot een event gebruikt de geplande
tijd; deze features bewijzen geen werkelijke publicatie of surprise.

Een ontbrekende kalender of een kalender die nog niet bekend was, geeft
onbekende fase/flags en `event_is_missing=True`. Beschikbare schedule-evidence
betekent niet dat de kalender compleet is. `normal` betekent alleen dat geen
op dat moment bekende schedule binnen het venster valt. Een latere bronadapter
moet daarnaast historische kalenderdekking onderbouwen.

`local_event_time` converteert expliciet lokale tijden naar UTC en weigert
onbestaande/ambigue tijden rond DST. Winter-/zomervoorbeelden en beide
DST-overgangen zijn getest. De synthetische CLI-run bewaart een kalender met
wijziging en annulering en voegt de eventvelden aan de audittabel toe. Deze
eventvelden zijn nog geen modelinputs; bronablation blijft één bron tegelijk.
De eventtabel accepteert geen extra consensus-/surprisevelden.

## Zilverbron: beschikbaar, maar nog niet toegelaten

De officiële HistData-downloadpagina's voor XAGUSD M1 2020–2024 vermelden de
verwachte jaarlijkse ZIP-bestanden. Die pagina's zijn op 2026-09-09 gecontroleerd;
de ZIP-inhoud en historische dekking zijn nog niet gedownload of gevalideerd.
Zie de [2020-pagina](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xagusd/2020)
en [2024-pagina](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xagusd/2024).

De [officiële specificatie](https://www.histdata.com/f-a-q/data-files-detailed-specification/)
beschrijft vaste EST zonder DST. De [FAQ](https://www.histdata.com/f-a-q/) beschrijft
bid-only OHLC en onbruikbaar volume. De interpretatie als candle-open timestamp
komt uit de bestaande projectadapter; de geraadpleegde specificatie benoemt dat
niet expliciet. Dit moet bij een XAGUSD-adapter als aanname worden geregistreerd.

De geraadpleegde bronvelden leveren geen historische publicatie- of revisietijden.
Daaruit volgt voor deze fase: een `available_at` berekend uit candle close plus
60 seconden is een **onderzoeksaanname**. De huidige downloadtijd bewijst geen
historische beschikbaarheid. Daarom staat `configs/phase10_silver.yaml` op
`availability_basis: modeled_latency` en `enabled: false`. `require_strict_pit`
weigert zowel deze basis als synthetische beschikbaarheid. Ook een gedeclareerde
provider timestamp vereist nog inhoudelijke controle van de brononderbouwing.

De bestaande goudadapter blijft ongewijzigd: zilver mag niet als XAUUSD worden
geparseerd en daarna worden hernoemd. Bij uitbreiding moeten URL, formulier,
ZIP-member, bronidentiteit en metadata consequent XAGUSD zijn. Data blijft lokaal
volgens [het databronbeleid](data_licenses.md).

## Resterend werk vóór afronding van fase 10

1. Kwalificeer de eerste zilverbron en haar historische beschikbaarheidsbewijs.
   Als alleen modeled latency haalbaar is, documenteer dat als apart exploratief
   protocol; presenteer het niet als de strikte point-in-time benchmark.
2. Bouw de bronadapter, geverifieerde lokale bronmanifesten en real-data preflight.
3. Bevries het empirische ablationprotocol: dezelfde ontwikkelingsfolds en
   sample IDs, train-only transformaties, inner-only selectie, expliciete
   toelatingscriteria en vergelijking met de gepinde price-only champion.
4. Voer de marktbenchmark uit met coverage, stale/missing, worst-foldkwaliteit
   en base-/stresskosten. Alleen de bron met aangetoonde meerwaarde kan worden
   toegelaten; de bronloze champion blijft beschikbaar.
5. Herhaal dit achtereenvolgens voor dollar, rente en de eventbronnen. Betrouwbare
   historische schedule-vintages zijn vereist; consensus/surprise blijft uit
   totdat daarvoor afzonderlijk point-in-time bewijs bestaat.

De technische proefresultaten mogen geen Phase-9 artefacten, championconfiguratie,
holdoutbesluit, probability-calibratie of paper/live trading activeren.

## Verificatie — 2026-09-09

- Volledige projectsuite: **804 passed, 1 skipped** in 90,98 seconden. Alleen de
  symlinktest is overgeslagen omdat deze Windows-host geen symlinks mag maken.
- Ruff, strikte mypy (88 bronbestanden), `pip check` en sdist/wheel-build geslaagd.
- CLI-proefrun `reports/phase10_dry_run_20260909`: **8 artefacten** geverifieerd,
  895 synthetische prediction rows en 178 testrows per variant.
- Bij tijdelijke uitval/vensterherstel gebruiken 55 testrows exact de price-only
  probabilities; zonder bron vallen alle 178 testrows exact daarop terug.
- Bronpreflight geeft de verwachte blokkades voor de uitgeschakelde, nog niet
  point-in-time bewezen zilverbron. Dit is geen geslaagde real-data preflight.

Het compacte [verificatiebewijs](../reports/phase10_verification_20260909.json)
bevat ook SHA-256-hashes van de nieuwe implementatie en het completionmanifest.
De proefrun is uitgevoerd op een lokale gewijzigde checkout; het is geen
benchmark vanaf een bevroren releasecommit.
