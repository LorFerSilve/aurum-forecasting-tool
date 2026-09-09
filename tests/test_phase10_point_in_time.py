from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.phase10.contracts import (
    ContextSource,
    require_strict_pit,
    validate_observations,
)
from gold_forecasting.phase10.point_in_time import context_coverage, join_context


def _source(**updates: Any) -> ContextSource:
    return ContextSource.model_validate(
        {
            "source_id": "context",
            "description": "Synthetic release stream for contract tests",
            "source_url": "https://example.test/context",
            "market_hours": "Scheduled releases",
            "source_timezone": "America/New_York",
            "publication_delay_seconds": 0,
            "stale_after_seconds": 120,
            "availability_basis": "synthetic",
            "revision_policy": "vintages",
            "availability_evidence": "Explicit synthetic release timestamps",
            "enabled": True,
            **updates,
        }
    )


def _predictions(*times: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"prediction_time_utc": pd.to_datetime(list(times), utc=True, format="mixed")}
    )


def _observations(*releases: tuple[str, str, str, str, float]) -> pd.DataFrame:
    """Make explicit observation ID, vintage ID, observation, release and value rows."""
    frame = pd.DataFrame(
        [
            {
                "source_id": "context",
                "observation_id": observation_id,
                "revision_id": revision_id,
                "observed_at_utc": observed,
                "available_at_utc": released,
                "ingested_at_utc": released,
                "value": value,
                "source_uri": "https://example.test/context/raw.csv",
                "raw_sha256": "a" * 64,
            }
            for observation_id, revision_id, observed, released, value in releases
        ]
    )
    for name in ("observed_at_utc", "available_at_utc", "ingested_at_utc"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def test_release_cutoff_is_inclusive_to_the_nanosecond_without_backfill() -> None:
    predictions = _predictions(
        "2024-01-02 08:59:00",
        "2024-01-02 09:00:59.999999999",
        "2024-01-02 09:01:00",
    )
    observations = _observations(
        ("first", "v1", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 12.0),
    )

    result = join_context(predictions, observations, _source())

    assert result["context_is_missing"].tolist() == [True, True, False]
    assert result["context_value"].iloc[:2].isna().all()
    assert result["context_available_at_utc"].iloc[:2].isna().all()
    assert result.loc[2, "context_value"] == 12.0
    assert result.loc[2, "context_age_seconds"] == 60.0


def test_late_old_revision_never_replaces_newest_observation() -> None:
    observations = _observations(
        ("new", "v2", "2024-01-02 09:03:00", "2024-01-02 09:06:00", 21.0),
        ("old", "v2", "2024-01-02 09:00:00", "2024-01-02 09:05:00", 11.0),
        ("new", "v1", "2024-01-02 09:03:00", "2024-01-02 09:04:00", 20.0),
        ("old", "v1", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 10.0),
    )
    predictions = _predictions(
        "2024-01-02 09:01:00",
        "2024-01-02 09:04:00",
        "2024-01-02 09:05:00",
        "2024-01-02 09:06:00",
    )

    result = join_context(predictions, observations, _source())

    assert result["context_value"].tolist() == [10.0, 20.0, 20.0, 21.0]
    assert result["context_revision_id"].tolist() == ["v1", "v1", "v1", "v2"]
    assert result.loc[2, "context_available_at_utc"] == pd.Timestamp(
        "2024-01-02 09:04:00", tz="UTC"
    )


def test_revision_does_not_rejuvenate_observation_age() -> None:
    observations = _observations(
        ("old", "v1", "2024-01-02 09:00:00", "2024-01-02 09:00:00", 10.0),
        ("old", "v2", "2024-01-02 09:00:00", "2024-01-02 09:10:00", 12.0),
    )
    result = join_context(
        _predictions("2024-01-02 09:10:00"), observations, _source()
    )

    assert result.loc[0, "context_value"] == 12.0
    assert result.loc[0, "context_age_seconds"] == 600.0
    assert result.loc[0, "context_is_stale"]
    assert not result.loc[0, "context_is_missing"]


def test_staleness_boundary_and_explicit_missing_are_distinct() -> None:
    observations = _observations(
        ("first", "v1", "2024-01-02 09:00:00", "2024-01-02 09:00:00", 10.0),
        ("missing", "v1", "2024-01-02 09:03:00", "2024-01-02 09:03:00", np.nan),
    )
    result = join_context(
        _predictions(
            "2024-01-02 08:59:00",
            "2024-01-02 09:02:00",
            "2024-01-02 09:02:00.000000001",
            "2024-01-02 09:03:00",
        ),
        observations,
        _source(),
    )

    assert result["context_is_missing"].tolist() == [True, False, False, True]
    assert result["context_is_stale"].tolist() == [False, False, True, False]
    assert result.loc[2, "context_value"] == 10.0
    assert pd.isna(result.loc[3, "context_value"])
    assert result.loc[3, "context_observation_id"] == "missing"
    coverage = context_coverage(result, "context")
    assert coverage["rows"] == 4
    assert coverage["missing_rows"] == 2
    assert coverage["stale_rows"] == 1
    assert coverage["usable_rows"] == 1
    assert coverage["usable_fraction"] == 0.25


def test_mutating_future_releases_preserves_all_past_joined_fields() -> None:
    observations = _observations(
        ("first", "v1", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 10.0),
        ("future", "v1", "2024-01-02 09:04:00", "2024-01-02 09:05:00", 20.0),
        ("first", "v2", "2024-01-02 09:00:00", "2024-01-02 09:06:00", 11.0),
    )
    predictions = _predictions("2024-01-02 09:00:00", "2024-01-02 09:03:00")
    altered = observations.copy()
    altered.loc[1:, "value"] = [9000.0, -9000.0]
    altered.loc[1:, "raw_sha256"] = "b" * 64
    altered.loc[1:, "revision_id"] = "mutated"

    baseline = join_context(predictions, observations, _source())
    mutated = join_context(predictions, altered, _source())
    only_released = join_context(predictions, observations.iloc[:1], _source())

    assert_frame_equal(baseline, mutated)
    assert_frame_equal(baseline, only_released)


def test_ambiguous_same_time_vintages_fail_closed() -> None:
    observations = _observations(
        ("first", "v1", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 10.0),
        ("first", "v2", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 11.0),
    )
    with pytest.raises(ValueError, match="ambiguous versions"):
        validate_observations(observations, _source())


@pytest.mark.parametrize("zone", [None, "America/New_York"])
@pytest.mark.parametrize(
    "column", ["observed_at_utc", "available_at_utc", "ingested_at_utc"]
)
def test_core_rejects_naive_or_nonutc_observation_times(
    column: str, zone: str | None
) -> None:
    observations = _observations(
        ("first", "v1", "2024-01-02 09:00:00", "2024-01-02 09:01:00", 10.0),
    )
    observations[column] = observations[column].dt.tz_convert(zone)

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        validate_observations(observations, _source())


@pytest.mark.parametrize("zone", [None, "America/New_York"])
def test_core_rejects_naive_or_nonutc_prediction_times(zone: str | None) -> None:
    predictions = _predictions("2024-01-02 09:01:00")
    predictions["prediction_time_utc"] = predictions["prediction_time_utc"].dt.tz_convert(
        zone
    )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        join_context(predictions, None, _source())


@pytest.mark.parametrize(
    ("local_release", "utc_release"),
    [
        ("2024-01-02 08:30:00", "2024-01-02 13:30:00"),
        ("2024-07-02 08:30:00", "2024-07-02 12:30:00"),
    ],
)
def test_source_local_release_is_accepted_after_explicit_seasonal_utc_conversion(
    local_release: str, utc_release: str
) -> None:
    released = pd.Timestamp(local_release, tz="America/New_York").tz_convert("UTC")
    assert released == pd.Timestamp(utc_release, tz="UTC")
    observations = _observations(
        ("release", "v1", str(released), str(released), 10.0),
    )
    predictions = pd.DataFrame(
        {"prediction_time_utc": [released - pd.Timedelta(1, unit="ns"), released]}
    )

    result = join_context(predictions, observations, _source())

    assert result["context_is_missing"].tolist() == [True, False]


def test_disabled_optional_source_skips_invalid_observation_validation() -> None:
    predictions = _predictions("2024-01-02 09:01:00")
    result = join_context(
        predictions, pd.DataFrame({"invalid": [object()]}), _source(enabled=False)
    )

    assert_frame_equal(result[predictions.columns], predictions)
    assert result["context_is_missing"].all()


def test_missing_optional_source_preserves_nondefault_prediction_index() -> None:
    predictions = _predictions("2024-01-02 09:01:00", "2024-01-02 09:02:00")
    predictions.index = pd.Index([42, 17], name="original_row")
    result = join_context(predictions, None, _source())

    assert_frame_equal(result[predictions.columns], predictions)
    assert result["context_is_missing"].all()
    assert not result["context_is_stale"].any()
    assert result["context_age_seconds"].isna().all()
    assert str(result["context_observed_at_utc"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("timestamp", ["2019-12-31 23:59:59", "2025-01-01 00:00:00"])
def test_prediction_development_boundaries_fail_closed(timestamp: str) -> None:
    with pytest.raises(ValueError, match="development 2020-2024"):
        join_context(_predictions(timestamp), None, _source())


@pytest.mark.parametrize("timestamp", ["2019-12-31 23:59:59", "2025-01-01 00:00:00"])
def test_observation_development_boundaries_fail_closed(timestamp: str) -> None:
    observations = _observations(("outside", "v1", timestamp, timestamp, 10.0))
    with pytest.raises(ValueError, match="development 2020-2024"):
        validate_observations(observations, _source())


@pytest.mark.parametrize("basis", ["modeled_latency", "synthetic"])
def test_strict_pit_admission_rejects_hypothetical_availability(basis: str) -> None:
    with pytest.raises(ValueError, match="provider release evidence"):
        require_strict_pit(_source(availability_basis=basis))


def test_strict_pit_admission_accepts_provider_release_basis() -> None:
    require_strict_pit(_source(availability_basis="provider_timestamp"))
