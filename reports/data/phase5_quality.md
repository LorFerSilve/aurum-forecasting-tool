# Fase 5 — dagelijkse datakwaliteit

Venster: 2020-01-01T00:00:00+00:00 tot 2025-01-01T00:00:00+00:00 (exclusief).
Historisch auditmoment: 2025-01-01T00:00:00+00:00; kalender: `observed-source-v1`.

HistData M1 levert bid-OHLC. Ask, mid, spread en betrouwbare tick count ontbreken.
Er is geen geverifieerde sessie-/feestdagenkalender geconfigureerd. Weekends,
feestdagen en onderhoud worden daarom niet afgeleid uit ontbrekende prijzen.
Onbekende sluitingen blijven gemarkeerd. Er wordt niets geïnterpoleerd.

`Missing` telt gaten binnen geobserveerde dagdekking; `onbekende gaten`
omvat ook ontbrekende intervallen buiten die dekking. Deze tellingen overlappen.
`Stale` gaat over ontbrekende intervallen op het vaste historische auditmoment,
niet over de actuele bruikbaarheid van deze historische feed voor live trading.
Een lege hogere dataset betekent dat geen venster aantoonbaar volledig is.

| Timeframe | Candles | Dagen | Missing | Stale | Incomplete | Onbekende gaten |
|---|---:|---:|---:|---:|---:|---:|
| 1min | 1726589 | 1827 | 101230 | 101230 | 0 | 2766 |
| 3min | 575254 | 1827 | 33995 | 33995 | 0 | 2732 |
| 5min | 345000 | 1827 | 20517 | 20517 | 0 | 2708 |
| 15min | 114762 | 1827 | 6998 | 6998 | 0 | 2644 |
| 30min | 57215 | 1827 | 3543 | 3543 | 0 | 2572 |
| 1h | 28463 | 1827 | 1701 | 1701 | 0 | 2488 |
| 3h | 8402 | 1827 | 68 | 68 | 0 | 1984 |
| 1d | 0 | 1827 | 0 | 0 | 0 | 1827 |
| 1mo | 0 | 1827 | 0 | 0 | 0 | 60 |
