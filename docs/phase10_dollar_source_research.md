# Phase 10 bron 2: keuze van de dollarproxy

**Onderzocht:** 2026-09-10, uitsluitend publieke bronbeschrijvingen en downloadpagina's.  
**Keuze:** HistData EURUSD M1 bid-close, met inverse returnrichting als dollarproxy.  
**Protocol:** `phase10-dollar-eurusd-modeled-v1`.  
**Evidence:** uitsluitend exploratory modeled latency; geen strict-PIT admission.  
**Marktbenchmark:** niet uitgevoerd tijdens het bronnenonderzoek; ZIP-inhoud en dekking
moeten lokaal worden geverifieerd.

## Besluit vóór de eerste dollarbenchmark

De gekozen hypothese is **dollarsterkte tegenover de euro**, gemeten als het negatieve
log-rendement van EURUSD. Dit is een bilaterale proxy, geen DXY-reconstructie, geen brede
trade-weighted dollarindex en geen nieuw verhandelbaar instrument. Een positief resultaat
geldt alleen voor deze expliciete hypothese; een negatief resultaat sluit andere
dollarbronnen niet uit.

Echte ICE DXY heeft de voorkeur voor een latere brede-dollarhypothese zodra een bruikbare
historische levering met voldoende rechten en release/correctieprovenance beschikbaar is.
De publieke documentatie biedt nu geen direct te verifiëren 2020–2024 bestandenset met
historische per-row availability-evidence. Daarom is het conservatiever nu een helder
geïdentificeerde bilaterale bron te implementeren dan een ongedocumenteerde indexfeed
stilzwijgend als ICE DXY te behandelen.

Deze keuze verandert niets aan de afgeronde silverbeslissing `stop` of aan de frozen
Phase-7 15m `mvp/logistic` baseline, outer jaren 2022/2023/2024, 181 minuten gap, gold
samplecontract, inner-only selectie of train-only preprocessing. De 2025+ holdout blijft
gesloten en paper/live trading blijft uit.

## Beoordeelde bronnen

| Bron | Geschiktheid en bewijs | Besluit |
|---|---|---|
| ICE cash U.S. Dollar Index, symbool DXY | Directe basketproxy; ICE noemt intraday/daily historie sinds 1996 en historische levering via zijn dataoplossingen. Exacte dekking en historische availabilityvelden vereisen een concrete levering. | Voorkeursbron wanneer praktisch verifieerbaar; nu niet toegelaten. |
| HistData UDXUSD | De provider noemt dit een US Dollar Index en toont jaarlijkse M1-downloadpagina's voor 2020–2024. In de geraadpleegde documentatie ontbreken de exacte index-/venue-identiteit, constructie, eventuele futuresrollen, upstream provenance en historische releasevelden. | Niet gelijkstellen aan geauthenticeerde ICE cash DXY; nu niet gekozen. |
| HistData inverse EURUSD | Duidelijke bilaterale identiteit, M1 bid-OHLC, vaste bronzone en dezelfde importvorm als de bestaande broninfrastructuur. Geen historische releases; euro-specifieke bewegingen blijven een beperking. | Gekozen voor één afzonderlijke exploratory ablation. |
| Federal Reserve broad nominal dollar index | Brede dollarmaatstaf met dagelijkse observaties en wekelijkse publicatie. Vereist historische vintages, release/correctie-audit en een apart contract voor tragere context. | Verdedigbaar voor later macro-onderzoek, minder passend bij deze intraday hypothese. |

ICE bevestigt DXY als officiële indexnaam en symbool. Zijn catalogus beschrijft een
geometrische zesvalutabasket met 57,6% eurogewicht, berekend uit spot midquotes.
Dat grote eurogewicht motiveert de bilaterale proxy als onderzoekshypothese; het bewijst
geen equivalentie of voorspellende relatie met gold.
[ICE Currency Indices](https://www.ice.com/fixed-income-data-services/index-solutions/currency-indices),
[ICE officiële datacatalogus](https://developer.ice.com/fixed-income-data-services/catalog/ice-data-indices-currency-indices).

ICE Consolidated History beschrijft opgeslagen feeddata en historische bars vanaf één
minuut. Dit is algemene productdekking, geen bevestiging dat iedere gewenste DXY-rij over
2020–2024 alle originele publicatie- en correctietimestamps bevat. Een historische bar
timestamp of een actueel publicatieschema is geen bewijs van historische beschikbaarheid.
[ICE Consolidated History](https://www.ice.com/fixed-income-data-services/access-and-delivery/connectivity-and-feeds/ice-consolidated-history).

ICE reserveert rechten op de index en verlangt toestemming voor gebruik. Er is hier geen
prijs, entitlement of research-/redistributierecht verondersteld; dat moet bij een concrete
datalevering worden vastgelegd. Er is geen brokerlogin, aankoop of licentie aangevraagd.
[ICE USDX brochure, rechtenverklaring op laatste pagina](https://www.theice.com/publicdocs/ICE_USDX_Brochure.pdf).

HistData biedt gratis downloads voor backtesting en noemt geen garantie of certificering.
De geraadpleegde pagina's bewijzen geen algemene open-data- of redistributielicentie.
Bewaar lokaal verkregen bronbestanden buiten Git en leg de daadwerkelijk meegeleverde
voorwaarden vast; de projectcode verleent geen rechten op marktdata. De FAQ identificeert
UDX/USD slechts algemeen en documenteert bid-OHLC, ontbrekende bar-ask en vaste EST zonder
DST. [HistData FAQ](https://www.histdata.com/f-a-q/).

De Fed publiceert dagelijkse nominale indexwaarden via de wekelijkse H.10-release, met
voorgaande-weekdata. De actuele historische reeksen kunnen na nieuwe gewichten worden
herzien. Een download van de huidige reeks met de observatiedatum als availability zou
daardoor toekomstinformatie kunnen gebruiken.
[Fed H.10 releasebeschrijving](https://www.federalreserve.gov/releases/h10/about.htm),
[Fed indexen en revisies](https://www.federalreserve.gov/releases/h10/Summary/).
Zelfs historische release-URL's moeten op correcties worden gecontroleerd: de pagina
`20200106` toont nu een correctie met 30 januari 2020 als releasedatum. Een oorspronkelijke
vintage kan dus niet alleen uit de URL worden afgeleid.
[Concreet H.10 correctievoorbeeld uit 2020](https://www.federalreserve.gov/releases/h10/20200106/).

## Semantiek en causale featurehypothese

De ruwe contextwaarde blijft EURUSD `bid_close`, oftewel USD per EUR. Voor prijs `E` is
de directionele dollarreturn over `k` opeenvolgende beschikbare observaties:

```text
dollar_return_k_bps = -10_000 * log(E_t / E_(t-k))
```

Een EURUSD-daling geeft zo een positieve dollarreturn. De omgekeerde prijs `1 / E` is
hoogstens een wiskundige directionele maat; de inverse van een bid is geen uitvoerbare
USDEUR-bid. Er wordt geen ask, spread, volume of microstructure gereconstrueerd.

Korte returns/momentum en trailing gold/dollar-returncorrelatie toetsen of de betekenis
van een goldbeweging afhangt van de actuele dollarbeweging. Een returnproduct kan die
interactie aan logistic regression aanbieden zonder een prijsschaalratio tussen
verschillende eenheden te introduceren. Alleen verleden tot en met de prediction cutoff
mag worden gebruikt; precieze vensters en featurekolommen worden vóór de marktbenchmark
in het dollarprotocol/config bevroren. Geen vensterkeuze op outer-testresultaten.

Voor de import geldt dezelfde expliciete onderzoeksaanname als bij silver:

- Generic ASCII M1 timestamp geïnterpreteerd als candle-open; de provider bewijst deze
  open/close-conventie niet expliciet.
- Bronzone vaste UTC−05:00, zonder DST; `observed_at_utc = open + 1 minuut`.
- `available_at_utc = observed_at_utc + 60 seconden`; dit is modeled latency.
- Project-download/importtijd blijft apart in `ingested_at_utc`; geen historisch bewijs.
- Backward as-of gebruikt alleen reeds beschikbare versies. Age telt vanaf observatietijd.
- Gaps, ontbrekende rijen, stale waarden en onvoldoende featurehistorie leiden tot exacte
  frozen Phase-7 probabilities. Herhaalde as-of waarden zijn geen nieuwe candles.

De ASCII-specificatie documenteert het zesveldenformaat, `YYYYMMDD HHMMSS`, bid-OHLC en
de vaste bronzone; zij bevat geen oorspronkelijke release- of revisietimestamps.
[HistData bestandspecificatie](https://www.histdata.com/f-a-q/data-files-detailed-specification/).

## Eerst lokaal verkrijgen, daarna preflight

Gebruik op HistData **Generic ASCII → M1 → EUR/USD → jaar**. Verkrijg deze vijf originele
jaarlijkse ZIPs en bewaar ze onder `data/raw/phase10/dollar/`:

| Jaar en officiële bronpagina | Lokaal ZIP-bestand | Vereist data-member |
|---|---|---|
| [2020](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2020) | `HISTDATA_COM_ASCII_EURUSD_M1_2020.zip` | `DAT_ASCII_EURUSD_M1_2020.csv` |
| [2021](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2021) | `HISTDATA_COM_ASCII_EURUSD_M1_2021.zip` | `DAT_ASCII_EURUSD_M1_2021.csv` |
| [2022](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2022) | `HISTDATA_COM_ASCII_EURUSD_M1_2022.zip` | `DAT_ASCII_EURUSD_M1_2022.csv` |
| [2023](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2023) | `HISTDATA_COM_ASCII_EURUSD_M1_2023.zip` | `DAT_ASCII_EURUSD_M1_2023.csv` |
| [2024](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2024) | `HISTDATA_COM_ASCII_EURUSD_M1_2024.zip` | `DAT_ASCII_EURUSD_M1_2024.csv` |

Tijdens dit onderzoek retourneerden alle vijf selectorpagina's HTTP 200 en toonden zij
de genoemde ZIP-naam. Alleen kleine HTML-pagina's zijn opgehaald, geen ZIPs of marktbestanden.
Dit bewijst de aangeboden bestandsnamen, niet dat ZIP-inhoud, volledige jaardekking,
CRC, hashes of historische availability al geverifieerd zijn. Dezelfde pagina-controle
voor UDXUSD 2020–2024 bewijst evenmin de precieze onderliggende indexidentiteit.

Bewaar originele namen en eventuele statusbestanden; hernoem geen ander instrument naar
EURUSD. De adapter moet instrument/periodenaam, archive SHA-256, CRC, OHLC, source-year,
UTC-ontwikkelinggrenzen en duplicaatconflicten controleren en provenance vastleggen.
Gebruik geen bestand uit 2025+. Haal geen extra data op om de holdoutgrens op te vullen.

Na lokale acquisitie: importeer en hash de jaarpartities, voer de dollar real-data
preflight uit met dezelfde gold data en frozen reference, inspecteer coverage/fallback
en gold/reference parity en voer pas bij volledig groene exploratory preflight de
nested ablation uit. Strict-PIT readiness blijft false. Een exploratory succes kan
uitsluitend beter brononderzoek rechtvaardigen; champion promotion en trading activation
blijven technisch false.

De configuraties scheiden de uitgeschakelde strict bron (`configs/phase10_dollar.yaml`),
de exploratory bron (`configs/phase10_dollar_exploratory.yaml`) en de bevroren ablation
(`configs/phase10_dollar_ablation.yaml`). De lokale commando's zijn:

```powershell
gold-forecast phase10 dollar-import --archive-directory data/raw/phase10/dollar --output data/context/phase10/dollar
gold-forecast phase10 preflight --config configs/phase10_dollar_ablation.yaml
```

De import levert `data/context/phase10/dollar/dollar.bundle-set.json`. Pas nadat de
preflight volledig groen is mag `gold-forecast phase10 run --config
configs/phase10_dollar_ablation.yaml` worden uitgevoerd. Controleer de definitieve
commando-opties en uitvoerlocaties in de implementatiedocumentatie.
