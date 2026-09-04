# Verificatie fase 0 — onderzoekscontract v0.0

Datum: 2026-09-04  
Uitkomst: **geslaagd**

## Controles

- [x] `XAU_USD` en HistData Generic ASCII M1 zijn expliciet vastgelegd.
- [x] Bronoffset `UTC-05:00` zonder DST en normalisatie naar UTC zijn vastgelegd.
- [x] `1min` en `1mo` zijn ondubbelzinnig.
- [x] Feature cutoff, prediction, entry en exit hebben afzonderlijke tijden.
- [x] Voorbeeldreturn is handmatig nagerekend: `7,0188` bp bruto, `3,0188` bp netto, label `up`.
- [x] Train, validation en test zijn chronologisch en niet-overlappend.
- [x] De purge-gap is 16 minuten en dekt latency plus labelhorizon.
- [x] Normale ontwikkelruns weigeren data vanaf de holdoutgrens `2025-01-01T00:00:00Z`.
- [x] MVP-succes is technisch gedefinieerd en eist geen nog onbekende winst.
- [x] Gebruiks- en licentiebeperkingen van de publieke data zijn gedocumenteerd.

## Bevroren artefacten

| Bestand | SHA-256 |
|---|---|
| `docs/research_protocol.md` | `A12A46A9FB84C35DCEF147F20927543755A210857172C1A87260C6F1FF970B12` |
| `docs/data_licenses.md` | `E9E74F060374A42BC5A386C6772B794DD5059A06379D2CF2DAE61A3289C0245D` |
| `configs/instrument.yaml` | `EC33944CB60B032AD01D117016EF4782E5ED95A77F3E241AD7B9B600548710E5` |
| `configs/labels.yaml` | `5EAA18C021ACE501BD9D1D58E80E8819FFC37AF26C9132DA9C194D9855D94EEE` |
| `configs/costs.yaml` | `5EFC1B8E40788FCA6EE1AEDE39D9AFC29F2F0BCA65A0CC7DED37BE171E693211` |
| `configs/splits_mvp.yaml` | `764805D6E0A4567AB3474936456BA428E209221EBCB5A015ECA7D7137038E776` |

## Besluit

`promote`: fase 0 is bevroren als releasecontract `v0.0`; fase 1 mag starten.

