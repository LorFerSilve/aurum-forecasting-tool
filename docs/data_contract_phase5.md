# Datacontract fase 5

Dit document beschrijft het geïmplementeerde datacontract van `configs/phase5.yaml`. Het profiel breidt de bestaande onderzoeksdata uit met extra timeframes, incrementele acquisitie en dagelijkse kwaliteitsrapportage. Het gebruikt hetzelfde XAU/USD-instrument, dezelfde publieke HistData-bron en dezelfde ontwikkelingsperiode als de MVP: `[2020-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`.

De feature-, label-, model- en kostenconfiguraties blijven die van de MVP. Extra datasets toevoegen betekent niet dat het model die timeframes automatisch als features gebruikt. De finale holdout blijft buiten gewone import-, update- en modelruns. Het brongebruik volgt het bestaande lokale onderzoeksbeleid in [data_licenses.md](data_licenses.md); fase 5 verandert dat beleid niet.

## Bron en providercontract

De huidige adapter is `histdata_ascii_m1`, voor bron `histdata` en symbool `XAUUSD`. M1 levert bid-open, bid-high, bid-low en bid-close. Ask, mid, spread en betrouwbare tick count zijn niet beschikbaar in dit adaptercontract. Het oorspronkelijke volumeveld wordt als `volume_unreliable` beschreven en niet als betrouwbare modelinput gebruikt.

Een bronrecord bevat een candle-open in vaste `UTC-05:00`, zonder DST. De adapter normaliseert dit naar UTC en bepaalt de sluiting één minuut later. Bronregels worden niet behandeld alsof hun lokale jaartal noodzakelijk gelijk is aan hun UTC-jaartal.

Het basiscontract `M1Provider.ingest_year(...)` blijft beschikbaar. `IncrementalM1Provider` voegt resourcepagina's, resource-acquisitie en machineleesbare capabilities toe. Een andere geregistreerde adapter kan dezelfde canonieke candlekolommen leveren. De actuele HistData-capabilities zijn:

| Capability | Waarde | Betekenis in deze implementatie |
|---|---|---|
| `annual_immutable_resources` | `True` | Acquisitie per afgesloten bronjaar; lokaal bewaarde ZIP-bestanden worden niet overschreven. |
| `pagination_mode` | `none` | De HistData-adapter levert één resourcecatalogus zonder vervolgcursor. |
| `resume_mode` | `completed_resource_cache` | Voltooide, gecontroleerde archieven worden hergebruikt. |
| `revision_mode` | `content_sha256` | De lokale inhoudshash identificeert de bronrevisie. |

De generieke updater kan vervolgpagina's doorlopen en weigert herhaalde cursors, dubbele resources, overlappende resourceperioden en resources buiten het toegestane ontwikkelingsvenster. De HistData-adapter inventariseert alleen volledig afgesloten jaren binnen het verzoek. De normale download bevestigt vervolgens of een archief beschikbaar is.

`data update` haalt ontbrekende jaararchieven op en bouwt daarna de geconfigureerde curated datasets opnieuw. Met dezelfde reeds opgeslagen archieven vindt geen nieuwe download plaats. Hervatten gebeurt op archiefniveau: HTTP-range-resume van een gedeeltelijke download, een lopend jaar en een live minuutfeed zijn niet geïmplementeerd. Een bestaande lokale momentopname wordt ook niet periodiek met een mogelijke gewijzigde remote versie vergeleken.

Iedere HTTP-poging doorloopt dezelfde begrenzing. De standaardpolicy gebruikt maximaal drie pogingen, minimaal 0,25 seconde tussen pogingen en een oplopende wachttijd van 0,5 tot maximaal 4 seconden. Tijdelijke transportfouten en de geconfigureerde tijdelijke HTTP-fouten kunnen worden herhaald. Een numerieke `Retry-After` wordt binnen die bovengrens verwerkt.

## UTC-grid en volledige candles

Alle intervallen zijn links gesloten en rechts open: `[timestamp_open_utc, timestamp_close_utc)`. Alle lagere input voor resampling is canonieke `1min`-data. Iedere hogere candle wordt rechtstreeks uit deze bronresolutie opgebouwd.

| Code | UTC-grid | Exclusieve sluiting |
|---|---|---|
| `1min` | Iedere volle minuut | Open + 1 minuut |
| `3min`, `5min`, `15min`, `30min` | Respectieve veelvouden vanaf 00:00 UTC | Open + de opgegeven minuten |
| `1h` | 00:00, 01:00, …, 23:00 UTC | Open + 1 uur |
| `3h` | 00:00, 03:00, …, 21:00 UTC | Open + 3 uur |
| `1d` | Iedere dag 00:00 UTC | Volgende dag 00:00 UTC |
| `1mo` | Eerste dag van de kalendermaand, 00:00 UTC | Eerste dag van de volgende kalendermaand, 00:00 UTC |

`1mo` is geen vaste periode van dertig dagen. Februari 2024 loopt bijvoorbeeld van 1 februari tot 1 maart en omvat 29 dagen. De validator controleert de exacte maandgrens, ook wanneer een timestamp slechts een nanoseconde daarvan afwijkt.

De aggregatie neemt de eerste open, hoogste high, laagste low en laatste close van het venster. Ontbrekende minuten worden niet ingevuld. Een onvoltooide lagere candle diskwalificeert het volledige hogere venster; dubbele of ongeldige broncandles worden door validatie geweigerd. Een onvolledig begin- of eindfragment levert evenmin een volledige hogere candle op.

Standaard moet iedere minuut in het UTC-venster aantoonbaar aanwezig en volledig zijn. Deze eis geldt ook voor dagen en maanden. Daarnaast mag de sluiting niet na `closed_through_utc` liggen. Het fase-5-profiel gebruikt daarvoor het vaste einde van de ontwikkelingsperiode. Bij een directe resamplingaanroep zonder deze parameter wordt de laatste ingestietimestamp per bron/instrument als deterministische grens gebruikt.

De Python-API `resample_candles(...)` en `resample_timeframes(...)` accepteert een expliciete `MarketCalendarPolicy`. Een ontbrekende minuut is dan uitsluitend toegestaan wanneer de kalender die minuut expliciet als niet verwacht classificeert (`expected=False`). `expected=None` is onvoldoende bewijs. Ook ontbrekende minuten aan de randen van het venster worden gecontroleerd. `require_dense=True` eist altijd alle minuten; `require_dense=False` zonder kalenderpolicy wordt geweigerd.

De huidige HistData-configuratie bevat geen geverifieerde sessiekalender en blijft daarom bij de strikte minutencontrole. Lege `1d`- en `1mo`-datasets zijn een expliciet resultaat wanneer geen venster als volledig kan worden bewezen. In de gecontroleerde bronpartitie 2024 zijn beide datasets leeg. Ze worden met hun canonieke schema en een manifest met nul rijen opgeslagen en blijven zichtbaar in de kwaliteitsrapporten. Twee aanwezige minuten maken dus nooit een volledige historische maand, ook niet als de maand inmiddels verstreken is.

## Bronjaar, UTC-jaar en compatibiliteit

De bestaande MVP bewaart curated bestanden onder `source_year=<jaar>`. Dat jaartal verwijst naar het bronarchief. Dit pad en de oorspronkelijke `1min`-, `3min`- en `15min`-datasets blijven beschikbaar voor `configs/mvp.yaml`.

Het fase-5-profiel combineert de gekozen bronarchieven vóór resampling en partitioneert daarna op het UTC-openjaar. Dit voorkomt dat een bronbestandsgrens een verder volledig UTC-venster in twee fragmenten breekt. Bijvoorbeeld: een bronrecord met lokale open `2023-12-31 23:59` hoort na normalisatie bij UTC-jaar 2024.

De relevante paden zijn:

```text
data/raw/histdata/XAUUSD/1min/manifest.json
data/raw/histdata/XAUUSD/1min/manifest_phase5.json
data/curated/histdata/XAU_USD/<timeframe>/source_year=<jaar>/candles.parquet
data/curated/histdata/XAU_USD/phase5/<timeframe>/utc_year=<jaar>/candles.parquet
data/curated/histdata/XAU_USD/phase5/<timeframe>/manifest.json
data/curated/histdata/XAU_USD/phase5/phase5_build.json
```

Beide profielen delen dezelfde immutable raw archieven. De fase-5-build schrijft eigen raw/curated manifests en eigen datarapporten. `--years` selecteert bronarchieven en begrenst de gedeeltelijke build tot diezelfde UTC-jaren; eventuele bronregels in een volgend UTC-jaar blijven in raw maar vallen buiten opslag, aggregatie en audit van die partial build. Een gedeeltelijk manifest geldt nooit als een volledige dataset voor een normale modelrun. Aan de rand van de gekozen bronhistoriek kan UTC-dekking ontbreken; die wordt niet door aanvulling of extrapolatie verzonnen.

## Kalenderkennis en dagelijkse kwaliteit

Het geconfigureerde markturenbeleid is `observed_source_rows`, met kwaliteitskalender `observed-source-v1`. Dit beleid weet dat een record aanwezig is, maar leidt uit een afwezig record geen weekend, feestdag, onderhoud of daadwerkelijke providerstoring af. Die oorzaak blijft `unknown_market_status`. Een gewijzigd, niet-ondersteund kalenderbeleid wordt zowel bij bouwen als bij hergebruik geweigerd.

De kwaliteitsmodule ondersteunt expliciete wekelijkse sessies, weekenddagen, UTC-feestdag-/onderhoudsvensters en bekende gaten als Python-contracten. Deze worden niet automatisch voor HistData ingevuld. Een verklaring van een datagat is op zichzelf geen bewijs dat de markt gesloten was en certificeert geen hogere candle.

Het rapport bevat één record per UTC-dag, bron, instrument en timeframe, ook voor een geheel lege dataset. De dagelijkse records meten geobserveerde en verwachte candles, ontbrekende en stale intervallen, onvoltooide candles, duplicaten en ongeldige OHLC-waarden. Afzonderlijke gaprecords beschrijven aaneengesloten ontbrekende intervallen en hun verklaringstoestand.

Bij de huidige kalender telt `missing_count` gaten tussen de eerste en laatste geobserveerde candle van een dag. `unknown_gap_count` omvat ook ontbrekende intervallen buiten die geobserveerde dekking; deze tellers kunnen overlappen en mogen niet zonder meer worden opgeteld. Op een volledig lege dag betekent nul `missing_count` daarom niet dat volledige marktdekking is bewezen. Voor `1mo` ligt alleen op de eerste dag van iedere maand een mogelijke maandopen op het grid.

De audit gebruikt het vaste einde van het ontwikkelingsvenster als `as_of_utc`, met een stale-drempel van vijf minuten. Stale is daardoor reproduceerbare historische gapmetadata, geen actuele meting van live feed-latency. Een candle waarvan de sluiting na het auditmoment ligt telt als onvoltooid, ook als de bron haar als volledig markeert. De pipeline valideert de te bewaren candles vóór de audit; de auditmodule zelf kan ook defecte input tellen zonder die te repareren.

De zeven datarapporten zijn:

```text
reports/data/phase5_quality_daily.parquet
reports/data/phase5_quality_gaps.parquet
reports/data/phase5_quality.json
reports/data/phase5_quality.md
reports/data/phase5_coverage.json
reports/data/phase5_coverage.md
reports/data/phase5_gaps_1min.parquet
```

## Versies, lineage en voltooiing

| Onderdeel | Logica-/beleidsversie |
|---|---|
| Raw manifestlogica | `1.0.0` |
| Curated normalisatie/partitionering | `2.0.0` |
| Resampling | `2.0.0` |
| Kwaliteitsrapportage | `1.0.0` |
| Geobserveerde bronkalender | `observed-source-v1` |

Deze versies staan los van de inhoudsversie `sha256:...` van iedere dataset. De bronrevisie gebruikt de archiefhash. De eerste ingestietijd wordt bewaard en bij hergebruik gecontroleerd; herhaald verwerken geeft de bronregels geen nieuwe ingestietijd. Bestaande MVP-acquisitiemetadata blijft bruikbaar. Beschadigde of conflicterende lokale raw bestanden worden niet stilzwijgend vervangen.

Het raw manifest identificeert de archieven en acquisitiemetadata. Curated `1min` verwijst exact naar deze raw outputs; alle hogere curated manifests verwijzen exact naar de `1min`-outputs. Validatie controleert deze keten, bestandsomvang en hashes, schema, identiteit, rowcounts, tijdgrenzen, broncontract en de relevante logicaversies. Contentversies laten de niet-deterministische aanmaaktijd van een manifest buiten de hash; manifestbestanden zelf hoeven daardoor niet bytegelijk te zijn bij een rebuild.

Bestanden worden afzonderlijk atomair gepubliceerd. Pas na de data en alle kwaliteitsrapporten verschijnt `phase5_build.json`, met raw/curated contentversies, de kwaliteitslogicaversie en de hashes van de volledige rapportenset. Hergebruik vereist een passende completionrecord. Een fout halverwege kan eerder geschreven bestanden achterlaten; die worden niet als een nieuwe volledige build geaccepteerd wanneer versies of rapporthashes niet overeenkomen. Dit is geen rollback van een hele directory.

## Commando's en reproduceerbaarheidscontrole

Voer deze commando's uit vanuit de repositoryroot met de geïnstalleerde projectomgeving. Geef de fase-5-configuratie expliciet mee: `data import` en `data validate` gebruiken anders het MVP-profiel.

```powershell
.venv\Scripts\gold-forecast.exe data import --config configs/phase5.yaml
.venv\Scripts\gold-forecast.exe data update --config configs/phase5.yaml
.venv\Scripts\gold-forecast.exe data validate --config configs/phase5.yaml
```

`data import` bouwt de gekozen historie; bestaande raw archieven kunnen daarbij worden hergebruikt. `data update` doorloopt eerst het incrementele resourcecontract en bouwt daarna dezelfde datalaag. `data validate` controleert bestaande artefacten en downloadt niets.

Het verificatiescript verwacht een bestaande MVP-databuild als referentie:

```powershell
.venv\Scripts\python.exe scripts/verify_phase5.py
.venv\Scripts\python.exe scripts/verify_phase5.py --include-mvp
```

De standaardcontrole voert twee fase-5-updates uit, vergelijkt hun completionrecord, valideert de gehashte outputs en controleert dat de oorspronkelijke MVP-manifests behouden zijn. Bij succes schrijft zij `reports/phase5_data_verification.json`.

`--include-mvp` draait bovendien de volledige MVP eerst op de geharde data en vervolgens op de oorspronkelijke data. Het script vergelijkt vastgelegde evaluatie- en backtestartefacten en schrijft bij succes `reports/phase5_mvp_verification.json`. Deze optie bouwt ook de gedeelde feature- en modelartefacten; wanneer beide runs slagen eindigen die paden weer bij het oorspronkelijke MVP-profiel.

De aanwezigheid van dit contract of het verificatiescript is geen testuitslag. De daadwerkelijke runresultaten en eventuele resterende beperkingen horen in de gegenereerde verificatiebestanden en het faseverslag.
