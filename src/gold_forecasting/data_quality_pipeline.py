"""Persist daily quality evidence for a complete hardened data build."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gold_forecasting.artifacts import write_json_atomic, write_parquet_atomic, write_text_atomic
from gold_forecasting.config import ProjectConfig
from gold_forecasting.validation.quality import (
    ObservedSourceCalendarPolicy,
    build_daily_data_quality,
)

QUALITY_LOGIC_VERSION = "1.0.0"


def write_quality_artifacts(
    config: ProjectConfig,
    frames_by_timeframe: dict[str, list[pd.DataFrame]],
    *,
    years: tuple[int, ...],
) -> tuple[Path, list[Path]]:
    """Audit a fixed historical window; absent source-calendar knowledge stays explicit."""
    root = config.config_path.parent.parent
    directory = root / "reports" / "data"
    cutoff = pd.Timestamp(config.splits.splits.test.end)
    start = pd.Timestamp(year=min(years), month=1, day=1, tz="UTC")
    end = min(pd.Timestamp(year=max(years) + 1, month=1, day=1, tz="UTC"), cutoff)
    calendar = ObservedSourceCalendarPolicy()
    daily_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, object]] = {}
    lines = [
        "# Fase 5 — dagelijkse datakwaliteit",
        "",
        f"Venster: {start.isoformat()} tot {end.isoformat()} (exclusief).",
        f"Historisch auditmoment: {cutoff.isoformat()}; kalender: `{calendar.version}`.",
        "",
        "HistData M1 levert bid-OHLC. Ask, mid, spread en betrouwbare tick count ontbreken.",
        "Er is geen geverifieerde sessie-/feestdagenkalender geconfigureerd. Weekends,",
        "feestdagen en onderhoud worden daarom niet afgeleid uit ontbrekende prijzen.",
        "Onbekende sluitingen blijven gemarkeerd. Er wordt niets geïnterpoleerd.",
        "",
        "`Missing` telt gaten binnen geobserveerde dagdekking; `onbekende gaten`",
        "omvat ook ontbrekende intervallen buiten die dekking. Deze tellingen overlappen.",
        "`Stale` gaat over ontbrekende intervallen op het vaste historische auditmoment,",
        "niet over de actuele bruikbaarheid van deze historische feed voor live trading.",
        "Een lege hogere dataset betekent dat geen venster aantoonbaar volledig is.",
        "",
        "| Timeframe | Candles | Dagen | Missing | Stale | Incomplete | Onbekende gaten |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for timeframe, partitions in frames_by_timeframe.items():
        frame = pd.concat(partitions, ignore_index=True)
        report = build_daily_data_quality(
            frame,
            calendar=calendar,
            as_of_utc=cutoff,
            stale_after=pd.Timedelta(minutes=5),
            start_day_utc=start.date(),
            end_day_utc=(end - pd.Timedelta(days=1)).date(),
            identities=[
                (
                    config.instrument.provider.id,
                    config.instrument.instrument.id,
                    timeframe,
                )
            ],
        )
        daily = report.to_frame()
        daily_frames.append(daily)
        gap_frames.append(report.gaps_to_frame())
        totals = {
            column: int(daily[column].sum())
            for column in (
                "missing_count",
                "stale_count",
                "incomplete_count",
                "duplicate_count",
                "ohlc_invalid_count",
                "unknown_gap_count",
                "unexplained_gap_count",
            )
        }
        summaries[timeframe] = {
            "rows": len(frame),
            "days": len(daily),
            "usable_complete_candles": not frame.empty,
            **totals,
        }
        lines.append(
            f"| {timeframe} | {len(frame)} | {len(daily)} | {totals['missing_count']} | "
            f"{totals['stale_count']} | {totals['incomplete_count']} | "
            f"{totals['unknown_gap_count']} |"
        )
    daily_path = directory / "phase5_quality_daily.parquet"
    gaps_path = directory / "phase5_quality_gaps.parquet"
    summary_path = directory / "phase5_quality.json"
    markdown_path = directory / "phase5_quality.md"
    write_parquet_atomic(daily_path, pd.concat(daily_frames, ignore_index=True))
    write_parquet_atomic(gaps_path, pd.concat(gap_frames, ignore_index=True))
    write_json_atomic(
        summary_path,
        {
            "schema_version": 1,
            "quality_logic_version": QUALITY_LOGIC_VERSION,
            "calendar_version": calendar.version,
            "as_of_utc": cutoff.isoformat(),
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "stale_after_minutes": 5,
            "timeframes": summaries,
        },
    )
    write_text_atomic(markdown_path, "\n".join([*lines, ""]))
    return markdown_path, [daily_path, gaps_path, summary_path, markdown_path]
