"""Offline acquisition regressions for retry, resume, scope, and provenance."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from gold_forecasting.ingestion import (
    BoundedHttpRequester,
    HistDataArchive,
    HistDataArchiveError,
    HistDataDownloadError,
    HistDataIngestionResult,
    HistDataM1Provider,
    HistDataParseError,
    HttpRequestPolicy,
    IncrementalM1Provider,
    IncrementalUpdateError,
    M1Resource,
    M1ResourcePage,
    ProviderCapabilities,
    update_m1_history,
)

START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
OBSERVED = datetime(2026, 9, 5, tzinfo=UTC)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _resource(year: int) -> M1Resource:
    return M1Resource(
        f"histdata:XAUUSD:M1:{year}",
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year + 1, 1, 1, tzinfo=UTC),
        True,
        f"source_year={year}",
    )


def _archive_bytes(year: int, *, price: int = 2000, row_year: int | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"DAT_ASCII_XAUUSD_M1_{year}.csv",
            f"{row_year or year}0115 100000;{price};{price + 1};{price - 1};{price};0\n",
        )
    return buffer.getvalue()


def _form(year: int) -> str:
    fields = {
        "tk": "one-use-token",
        "date": str(year),
        "datemonth": str(year),
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": "XAUUSD",
    }
    inputs = "".join(f'<input name="{key}" value="{value}">' for key, value in fields.items())
    return f'<form id="file_down" action="/get.php">{inputs}</form>'


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> HistDataM1Provider:
    return HistDataM1Provider(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: OBSERVED,
        request_policy=HttpRequestPolicy(
            min_interval_seconds=0, initial_backoff_seconds=0, max_backoff_seconds=0
        ),
    )


def test_http_retries_are_bounded_and_every_attempt_is_rate_limited() -> None:
    clock = FakeClock()
    request_times: list[float] = []
    statuses = iter((429, 503, 200))

    def handler(request: httpx.Request) -> httpx.Response:
        request_times.append(clock.now)
        assert request.extensions["timeout"]["read"] == 7.0
        return httpx.Response(next(statuses), headers={"Retry-After": "99999"})

    requester = BoundedHttpRequester(
        HttpRequestPolicy(
            min_interval_seconds=2, initial_backoff_seconds=0.1, max_backoff_seconds=1
        ),
        sleeper=clock.sleep,
        monotonic=lambda: clock.now,
    )
    with httpx.Client(transport=httpx.MockTransport(handler), timeout=7) as client:
        assert requester.request(client, "GET", "https://example.test/data").status_code == 200
    assert request_times == [0, 2, 4]
    assert max(clock.sleeps) == 1


@pytest.mark.parametrize("failure", ["permanent", "transient", "transport"])
def test_http_errors_stop_at_the_correct_attempt(failure: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "transport":
            raise httpx.ReadTimeout("interrupted", request=request)
        return httpx.Response(404 if failure == "permanent" else 503)

    requester = BoundedHttpRequester(
        HttpRequestPolicy(min_interval_seconds=0, initial_backoff_seconds=0), sleeper=lambda _: None
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPError),
    ):
        requester.request(client, "GET", "https://example.test/data")
    assert len(requests) == (1 if failure == "permanent" else 3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_http_policy_rejects_nonfinite_or_negative_delays(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        HttpRequestPolicy(min_interval_seconds=value)


def test_histdata_catalogue_only_lists_closed_full_years() -> None:
    provider = HistDataM1Provider(clock=lambda: datetime(2024, 9, 1, tzinfo=UTC))
    assert isinstance(provider, IncrementalM1Provider)
    assert provider.capabilities.pagination_mode == "none"
    assert provider.capabilities.resume_mode == "completed_resource_cache"
    assert provider.capabilities.available_fields == (
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "volume_unreliable",
    )
    page = provider.list_resource_page(period_start_utc=START, period_end_utc=END)
    assert page.resources == (_resource(2023),)
    with pytest.raises(ValueError, match="pagination"):
        provider.list_resource_page(period_start_utc=START, period_end_utc=END, cursor="next")


def test_incremental_histdata_resumes_completed_resources_and_preserves_revision(
    tmp_path: Path,
) -> None:
    posts: list[int] = []
    fail_2024 = True

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            year = int(str(request.url).rsplit("/", 1)[1])
            return httpx.Response(200, text=_form(year))
        year = int(request.content.decode().split("date=")[1].split("&")[0])
        posts.append(year)
        if year == 2024 and fail_2024:
            return httpx.Response(503)
        return httpx.Response(200, content=_archive_bytes(year))

    provider = _provider(handler)
    with pytest.raises(HistDataDownloadError):
        update_m1_history(
            provider,
            tmp_path,
            period_start_utc=START,
            period_end_utc=END,
            reject_at_or_after_utc=END,
        )
    assert posts == [2023, 2024, 2024, 2024]
    first_state_path = provider.raw_archive_path(tmp_path, 2023).with_suffix(".archive.json")
    state_before = first_state_path.read_bytes()
    fail_2024 = False
    result = update_m1_history(
        provider,
        tmp_path,
        period_start_utc=START,
        period_end_utc=END,
        reject_at_or_after_utc=END,
    )
    assert result.downloaded_count == 1
    assert result.reused_count == 1
    assert posts == [2023, 2024, 2024, 2024, 2024]
    assert first_state_path.read_bytes() == state_before
    repeated = update_m1_history(
        provider,
        tmp_path,
        period_start_utc=START,
        period_end_utc=END,
        reject_at_or_after_utc=END,
    )
    assert repeated.downloaded_count == 0
    assert repeated.reused_count == 2
    for first, second in zip(result.batches, repeated.batches, strict=True):
        pd.testing.assert_frame_equal(first.candles, second.candles)
        assert first.archive.provider_revision == f"sha256:{first.archive.sha256}"
        assert first.archive.ingested_at_utc == second.archive.ingested_at_utc == OBSERVED


@pytest.mark.parametrize("mutation", ["content", "revision", "missing_archive"])
def test_cached_raw_provenance_is_immutable(tmp_path: Path, mutation: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, text=_form(2024))
            if request.method == "GET"
            else httpx.Response(200, content=_archive_bytes(2024))
        )

    provider = _provider(handler)
    archive = provider.download_year(2024, tmp_path)
    if mutation == "content":
        archive.path.write_bytes(_archive_bytes(2024, price=1000))
    elif mutation == "missing_archive":
        archive.path.unlink()
    else:
        state_path = archive.path.with_suffix(".archive.json")
        state = json.loads(state_path.read_text())
        state["provider_revision"] = "revised"
        state_path.write_text(json.dumps(state))
    with pytest.raises(HistDataArchiveError, match="immutable"):
        provider.download_year(2024, tmp_path)


def _seed_legacy_archive(tmp_path: Path) -> tuple[HistDataM1Provider, Path, dict[str, object]]:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail("migrating a cached archive must not contact the provider")

    provider = _provider(unexpected_request)
    archive_path = provider.raw_archive_path(tmp_path, 2024)
    archive_path.parent.mkdir(parents=True)
    content = _archive_bytes(2024)
    archive_path.write_bytes(content)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": "histdata",
        "source_symbol": "XAUUSD",
        "timeframe": "1min",
        "year": 2024,
        "archive": archive_path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_page_url": provider.source_page_url(2024),
        "first_ingested_at_utc": "2026-09-04T00:37:04.222601Z",
    }
    return provider, archive_path, metadata


def test_legacy_metadata_migration_preserves_original_timestamp_and_bytes(tmp_path: Path) -> None:
    provider, archive_path, metadata = _seed_legacy_archive(tmp_path)
    legacy_path = archive_path.with_suffix(".metadata.json")
    legacy_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
    legacy_path.write_bytes(legacy_bytes)
    archive_before = archive_path.read_bytes()
    expected_timestamp = datetime(2026, 9, 4, 0, 37, 4, 222601, tzinfo=UTC)

    first = provider.ingest_year(2024, tmp_path)
    second = provider.ingest_year(2024, tmp_path)

    assert first.archive.reused is True
    assert first.archive.observed_at_utc == OBSERVED
    assert first.archive.ingested_at_utc == second.archive.ingested_at_utc == expected_timestamp
    assert first.candles["ingested_at_utc"].eq(pd.Timestamp(expected_timestamp)).all()
    assert legacy_path.read_bytes() == legacy_bytes
    assert archive_path.read_bytes() == archive_before
    state = json.loads(archive_path.with_suffix(".archive.json").read_text())
    assert state["first_ingested_at_utc"] == metadata["first_ingested_at_utc"]
    assert state["provider_revision"] == f"sha256:{metadata['sha256']}"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("source", "other"),
        ("source_symbol", "XAGUSD"),
        ("timeframe", "15min"),
        ("year", 2023),
        ("archive", "other.zip"),
        ("sha256", "0" * 64),
        ("size_bytes", 0),
        ("source_page_url", "https://example.test/other"),
        ("first_ingested_at_utc", "2026-09-04T00:37:04"),
        ("first_ingested_at_utc", "2026-09-04T00:37:04+02:00"),
        ("first_ingested_at_utc", "not a timestamp"),
        ("first_ingested_at_utc", None),
    ],
)
def test_invalid_legacy_metadata_is_rejected_without_publishing_state(
    tmp_path: Path, field: str, value: object
) -> None:
    provider, archive_path, metadata = _seed_legacy_archive(tmp_path)
    metadata[field] = value
    legacy_path = archive_path.with_suffix(".metadata.json")
    legacy_bytes = json.dumps(metadata).encode()
    legacy_path.write_bytes(legacy_bytes)

    with pytest.raises(HistDataArchiveError, match="legacy raw metadata"):
        provider.download_year(2024, tmp_path)

    assert not archive_path.with_suffix(".archive.json").exists()
    assert legacy_path.read_bytes() == legacy_bytes


def test_current_and_legacy_raw_ingestion_timestamps_must_agree(tmp_path: Path) -> None:
    provider, archive_path, metadata = _seed_legacy_archive(tmp_path)
    legacy_path = archive_path.with_suffix(".metadata.json")
    legacy_path.write_text(json.dumps(metadata))
    provider.download_year(2024, tmp_path)
    state_path = archive_path.with_suffix(".archive.json")
    state_before = state_path.read_bytes()
    metadata["first_ingested_at_utc"] = "2026-09-03T00:37:04.222601Z"
    legacy_path.write_text(json.dumps(metadata))

    with pytest.raises(HistDataArchiveError, match="conflicts with legacy metadata"):
        provider.download_year(2024, tmp_path)

    assert state_path.read_bytes() == state_before


def test_interrupted_raw_publication_never_leaves_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_form(2024))
        return httpx.Response(200, content=_archive_bytes(2024))

    def fail_link(source: object, target: object) -> None:
        raise OSError("simulated crash before publication")

    provider = _provider(handler)
    with monkeypatch.context() as patch:
        patch.setattr("gold_forecasting.ingestion.histdata.os.link", fail_link)
        with pytest.raises(OSError, match="before publication"):
            provider.download_year(2024, tmp_path)
    assert list(tmp_path.rglob("*.zip")) == []
    assert list(tmp_path.rglob("*.tmp")) == []
    assert provider.download_year(2024, tmp_path).reused is False


def test_histdata_rejects_foreign_year_before_asof_filter(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_form(2024))
        return httpx.Response(200, content=_archive_bytes(2024, row_year=2025))

    with pytest.raises(HistDataParseError, match="declared source year"):
        _provider(handler).ingest_year(2024, tmp_path, as_of_utc=END)


class PagedProvider:
    capabilities = ProviderCapabilities(
        True, "cursor", "completed_resource_cache", "content_sha256", ("bid_close",)
    )

    def __init__(self, pages: tuple[M1ResourcePage, ...]) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []
        self.ingested: list[int] = []
        self.mutate: Callable[[pd.DataFrame], pd.DataFrame] = lambda frame: frame

    def list_resource_page(
        self, *, period_start_utc: datetime, period_end_utc: datetime, cursor: str | None = None
    ) -> M1ResourcePage:
        self.cursors.append(cursor)
        return self.pages[min(len(self.cursors) - 1, len(self.pages) - 1)]

    def ingest_resource(
        self,
        resource: M1Resource,
        raw_directory: str | Path,
        *,
        ingested_at_utc: datetime | None = None,
        as_of_utc: datetime | None = None,
    ) -> HistDataIngestionResult:
        return self.ingest_year(resource.period_start_utc.year, raw_directory)

    def ingest_year(
        self,
        year: int,
        raw_directory: str | Path,
        *,
        ingested_at_utc: datetime | None = None,
        as_of_utc: datetime | None = None,
    ) -> HistDataIngestionResult:
        self.ingested.append(year)
        path = Path(raw_directory) / f"{year}.zip"
        payload = _archive_bytes(year)
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        archive = HistDataArchive(
            path,
            digest,
            len(payload),
            year,
            "https://example.test/data",
            "https://example.test/download",
            OBSERVED,
            OBSERVED,
            f"sha256:{digest}",
            False,
        )
        candles = HistDataM1Provider().parse_archive(archive)
        return HistDataIngestionResult(archive, self.mutate(candles))


def test_updater_sorts_pages_before_acquiring_resources(tmp_path: Path) -> None:
    provider = PagedProvider(
        (
            M1ResourcePage((_resource(2024),), "second"),
            M1ResourcePage((_resource(2023),)),
        )
    )
    result = update_m1_history(
        provider,
        tmp_path,
        period_start_utc=START,
        period_end_utc=END,
        reject_at_or_after_utc=END,
    )
    assert provider.cursors == [None, "second"]
    assert provider.ingested == [2023, 2024]
    assert len(result.batches) == 2


@pytest.mark.parametrize(
    "problem", ["loop", "bound", "duplicate", "overlap", "scope", "capability"]
)
def test_bad_catalogues_fail_before_acquisition(tmp_path: Path, problem: str) -> None:
    resource = _resource(2023)
    pages = (M1ResourcePage((resource,), "again"),)
    expected = "repeated a cursor"
    if problem == "duplicate":
        pages = (M1ResourcePage((resource,), "next"), M1ResourcePage((resource,)))
        expected = "duplicate ids"
    elif problem == "overlap":
        pages = (M1ResourcePage((resource, replace(resource, resource_id="different"))),)
        expected = "overlapping periods"
    elif problem == "scope":
        pages = (M1ResourcePage((_resource(2025),)),)
        expected = "outside the requested interval"
    elif problem == "bound":
        expected = "exceeded 1 pages"
    elif problem == "capability":
        expected = "no pagination"
    provider = PagedProvider(pages)
    if problem == "capability":
        provider.capabilities = replace(provider.capabilities, pagination_mode="none")
    with pytest.raises(IncrementalUpdateError, match=expected):
        update_m1_history(
            provider,
            tmp_path,
            period_start_utc=START,
            period_end_utc=END,
            reject_at_or_after_utc=END,
            max_pages=1 if problem == "bound" else 10,
        )
    assert provider.ingested == []


@pytest.mark.parametrize("override,end", [(True, END), (False, datetime(2026, 1, 1, tzinfo=UTC))])
def test_holdout_requests_fail_before_listing_or_acquisition(
    tmp_path: Path, override: bool, end: datetime
) -> None:
    provider = PagedProvider((M1ResourcePage((_resource(2023),)),))
    with pytest.raises(IncrementalUpdateError, match="holdout"):
        update_m1_history(
            provider,
            tmp_path,
            period_start_utc=START,
            period_end_utc=end,
            reject_at_or_after_utc=END,
            allow_holdout_override=override,
        )
    assert provider.cursors == []
    assert provider.ingested == []


@pytest.mark.parametrize(
    "mutation", ["nat", "naive", "future", "old", "other_resource", "incomplete", "source"]
)
def test_provider_batch_cannot_bypass_scope_and_completion_guards(
    tmp_path: Path, mutation: str
) -> None:
    provider = PagedProvider((M1ResourcePage((_resource(2023),)),))

    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        if mutation == "nat":
            frame["timestamp_close_utc"] = pd.NaT
        elif mutation == "naive":
            frame["timestamp_close_utc"] = frame["timestamp_close_utc"].dt.tz_localize(None)
        elif mutation == "future":
            frame["timestamp_close_utc"] = pd.Timestamp("2025-01-01T00:01Z")
        elif mutation == "old":
            frame["timestamp_open_utc"] = pd.Timestamp("2022-12-31T00:01Z")
        elif mutation == "other_resource":
            frame["timestamp_open_utc"] = pd.Timestamp("2024-01-15T00:00Z")
            frame["timestamp_close_utc"] = pd.Timestamp("2024-01-15T00:01Z")
        elif mutation == "incomplete":
            frame["is_complete"] = False
        else:
            frame["source"] = "different-provider"
        return frame

    provider.mutate = mutate
    with pytest.raises(IncrementalUpdateError):
        update_m1_history(
            provider,
            tmp_path,
            period_start_utc=START,
            period_end_utc=END,
            reject_at_or_after_utc=END,
        )
