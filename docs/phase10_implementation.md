# Fase 10 — externe context en eventinformatie

**Status:** exploratory real-data runner en validator geïmplementeerd; nog geen echte XAGUSD markt-run of `v0.3`-release.

**Branch:** `codex/phase10-external-context`

**Proefprotocol:** `phase10-context-smoke-v1` (synthetisch).  
**Exploratory protocol:** `phase10-silver-modeled-v1` (real-data XAGUSD met modeled latency).

Phase 9 is op `main` gemerged in `9a15e04`. De onafhankelijke hercontrole op
2026-09-09 verifieerde de 225 canonieke artefacten en reproduceerde de opgeslagen
path-returnaudit exact. De Phase-7 researchchampions blijven bevroren, inclusief
`mvp/logistic` voor 15m. Er is geen tradingchampion. De finale 2025+ holdout blijft gesloten.

## Uitvoerbare eerste keten

```text
broncontract + geauthenticeerde CSV/Parquet-partities + bundle-set
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

Corefuncties accepteren uitsluitend expliciet UTC-aware timestamps. Kleine/synthetische
bundles kunnen CSV gebruiken; real-data context gebruikt gecomprimeerde Parquet-partities.
Iedere partitie heeft een exacte kolomvolgorde, row count, SHA-256 en halfopen
observatiegrenzen binnen 2020–2024. Een bundle-set mag uitsluitend sibling manifests
bevatten en de volledige samengevoegde release stream wordt opnieuw gevalideerd.
De XAGUSD-import bewaart daarnaast per jaarlijks bronarchief SHA-256, grootte,
bronpagina, gebruikte ZIP-members en project-ingestiontijd.

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

## Zilverbron: adapter gereed, strict-PIT evidence nog geblokkeerd

De officiële HistData XAGUSD M1-bron blijft methodologisch beperkt doordat historische
per-row release timestamps ontbreken. Daarom zijn nu twee expliciet gescheiden configs
aanwezig:

- `configs/phase10_silver.yaml`: disabled, strict gate blijft gesloten;
- `configs/phase10_silver_exploratory.yaml`: enabled voor uitsluitend
  `phase10-silver-modeled-v1`.

De aparte `phase10/silver_histdata.py` adapter verandert de frozen XAUUSD-ingestie niet.
Hij accepteert alleen XAGUSD ZIP-members, controleert archive SHA-256/CRC, vaste UTC−05,
source-year en OHLC-invarianten, en converteert de bid-close naar het contextcontract.
`observed_at` is candle close; `available_at` is candle close + 60 seconden en blijft
expliciet een modeled-latency aanname.

Real-data wordt per jaar als geauthenticeerde Parquet-partitie opgeslagen, met één
bundle-setmanifest voor de volledige 2020–2024 stream. De lokale bronarchieven worden
niet in Git opgenomen. Een modeled-latency bundle kan de status
`ready_for_exploratory_ablation` krijgen, maar `strict_pit_source_ready` en
`formal_benchmark_ready` blijven false.

Zie [het Phase-10 onderzoeksprotocol](research_protocol_phase10.md) voor de vooraf
bevroren admissionregels.

## Resterend werk vóór afronding van fase 10

De code voor de eerste silver ablation is nu gereed. De frozen Phase-7 15m
`mvp/logistic` reference wordt cryptografisch geauthenticeerd; het volledige gold
sample-universe, inner folds, outer folds en sample-digests moeten exact paritair zijn.
De silver challenger gebruikt uitsluitend dezelfde drie logistic `C`-waarden als
Phase 7 en behoudt `class_weight="balanced"`. Onbruikbare of fail-closed silverrows
vallen exact terug op de persisted Phase-7 probabilities.

De real-data preflight verifieert vóór RunRegistry-creatie:

1. de vijf lokale XAGUSD 2020–2024 ZIP-archieven;
2. archive/partition/bundle hashes en provenance;
3. de frozen Phase-7 completion en alle gebruikte reference predictions;
4. dezelfde gold curated datasets en Phase-7 featureconfig;
5. exact hetzelfde 15m sample-universe en 181-minuten foldcontract;
6. causal silver alignment, coverage, stale/missing en feature-warmup;
7. een clean committed Git identity;
8. `holdout_opened=false`.

De exploratory runner bewaart config/source/source-code snapshots, features,
reference- en challengerpredictions, inner selectie, trainingaudits, base/stress
backtests en een completionmanifest. De validator replayt artifact inventory,
reference hashes, gold alignment, causal silverfeatures, fallback, selected
three-C logistic grid, policyselectie, metrics en run-level admission gates.

Wat nog daadwerkelijk uitgevoerd moet worden:

1. zorg dat de echte XAGUSD 2020–2024 jaarlijkse ZIPs lokaal aanwezig zijn;
2. importeer ze met `phase10 silver-import`;
3. draai de real-data `phase10 preflight --config ...`;
4. inspecteer coverage/alignment en los alleen echte preflightproblemen op;
5. alleen na een volledig groene preflight: draai de modeled-latency exploratory run;
6. valideer de run met `phase10 validate`;
7. neem daarna een expliciet `stop` of `seek_strict_source` besluit.

Ook een positieve exploratory uitkomst kan geen championpromotie activeren.
Een strict `phase10-context-v1` benchmark blijft geblokkeerd totdat echte
historische provider-release-evidence beschikbaar is. Dollar, rente en eventbronnen
komen pas na de silverbeslissing aan bod.

## Verificatie — 2026-09-09

- GitHub CI op de actuele Phase-10 branch: **911 passed** in 101,83 seconden.
- Ruff volledig groen; strikte mypy: **95 bronbestanden, 0 issues**.
- De nieuwe tests dekken reference-authenticatie, exact gold-universe, archive/bundle
  provenance, causal silverfeatures, nested train-only selectie, exact fallbackgedrag,
  run-sealing lifecycle en artifact inventory.
- CLI-proefrun `reports/phase10_dry_run_20260909`: **8 artefacten** geverifieerd,
  895 synthetische prediction rows en 178 testrows per variant.
- Bij tijdelijke uitval/vensterherstel gebruiken 55 testrows exact de price-only
  probabilities; zonder bron vallen alle 178 testrows exact daarop terug.
- De disabled strict bronpreflight blijft terecht geblokkeerd. De nieuwe real-data
  preflight/runner is geïmplementeerd maar nog niet tegen de lokale 2020–2024 XAGUSD
  archive-set uitgevoerd; er is dus nog geen empirisch silverresultaat.

Het compacte [verificatiebewijs](../reports/phase10_verification_20260909.json)
bevat ook SHA-256-hashes van de nieuwe implementatie en het completionmanifest.
De proefrun is uitgevoerd op een lokale gewijzigde checkout; het is geen
benchmark vanaf een bevroren releasecommit.
