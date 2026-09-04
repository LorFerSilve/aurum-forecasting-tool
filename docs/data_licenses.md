# Dataherkomst en gebruiksbeperkingen

Laatst gecontroleerd: 2026-09-04.

## HistData.com — MVP-bron

Gebruikt product:

- XAU/USD (`XAUUSD`);
- Generic ASCII, M1-candles;
- publieke website-download zonder brokeraccount;
- lokale research- en backtestdoeleinden.

De [HistData FAQ](https://www.histdata.com/f-a-q/) noemt de bestanden gratis, bedoeld voor het testen/backtesten van strategieën en bruikbaar in externe applicaties. De [bestandspecificatie](https://www.histdata.com/f-a-q/data-files-detailed-specification/) beschrijft de M1-layout en tijdzone.

Belangrijke eigenschappen en beperkingen:

- M1-prijzen zijn bid-OHLC;
- ask staat alleen in de aparte tickbestanden;
- volume in de gratis bestanden is niet bruikbaar;
- timestamps staan in vaste EST (`UTC-05:00`) zonder DST;
- de leverancier geeft geen garantie of certificering voor juistheid;
- gaten kunnen zowel marktsluitingen als ontbrekende waarnemingen voorstellen;
- er is geen duidelijke open-data- of SPDX-licentie gepubliceerd.

Voor dit project geldt daarom het conservatieve beleid:

1. gebruik uitsluitend voor persoonlijke, lokale en niet-commerciële research;
2. commit of publiceer geen ruwe of afgeleide bronrecords;
3. bewaar lokaal de oorspronkelijke ZIP ongewijzigd met SHA-256-hash;
4. vermeld HistData als bron in interne manifests en rapporten;
5. hercontroleer de voorwaarden en vraag zo nodig schriftelijke toestemming vóór commercieel gebruik, publieke distributie of live inzet;
6. beschouw datafouten en ontbrekende records als expliciet modelrisico.

## Niet gekozen alternatieven

### Dukascopy

Dukascopy biedt technisch rijkere historische bid/ask-data, maar is niet gekozen. De actuele websitevoorwaarden beperken gebruik en geautomatiseerde toegang sterk en bevatten bovendien een waarschuwing voor Belgische inwoners. Zonder afzonderlijke schriftelijke toestemming wordt hiervoor geen downloader of database gebouwd.

### Twelve Data

Twelve Data blijft een mogelijke adapter met API-sleutel. De gratis-planbeschikbaarheid van XAU/USD is niet eenduidig en de intradayhistorie en output per request zijn begrensd. De MVP mag er niet stilzwijgend van afhankelijk worden.

### Latere brokerfeed

Een broker- of practicefeed wordt pas geselecteerd wanneer de research-MVP technisch werkt. Bij migratie worden bronidentiteit, bid/ask-definitie, markturen, timestamps, kosten en licentie opnieuw als protocolversie vastgelegd. Resultaten uit verschillende feeds worden niet behandeld alsof ze identiek zijn.

