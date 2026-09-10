"""Development-only gold/context integrity and common-universe preflight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import content_version, sha256_file, write_json_atomic
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds, select_block
from gold_forecasting.features.phase7 import build_phase7_features, load_phase7_feature_config
from gold_forecasting.labels.multihorizon import build_horizon_labels
from gold_forecasting.phase7.pipeline import _TIMEFRAMES, _validate_fold_coverage
from gold_forecasting.phase10.bundle import (
    ContextBundleManifest,
    ContextBundleSetManifest,
    _read_bounded,
    load_context_bundle,
)
from gold_forecasting.phase10.config import Phase10Config, load_phase10_config
from gold_forecasting.phase10.contracts import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    ContextSource,
    load_source,
    validate_observations,
)
from gold_forecasting.phase10.point_in_time import context_coverage, join_context
from gold_forecasting.phase10.profiles import (
    SILVER_PROFILE,
    ContextProfile,
    profile_for_source,
)
from gold_forecasting.phase10.reference import (
    Phase10Reference,
    load_phase10_reference,
    verify_phase10_universe,
)
from gold_forecasting.phase10.silver_histdata import parse_histdata_xagusd_archive
from gold_forecasting.registry import get_git_code_version

SILVER_MODEL_FEATURES = SILVER_PROFILE.model_features
_YEARS = (2020, 2021, 2022, 2023, 2024)
_MAX_METADATA = 1024 * 1024


class Phase10PreflightError(ValueError):
    """A required development-only integrity gate is not satisfied."""


@dataclass(frozen=True, slots=True)
class Phase10PreparedInputs:
    root: Path
    config: Phase10Config
    project: ProjectConfig
    source: ContextSource
    references: Phase10Reference
    table: pd.DataFrame
    feature_columns: tuple[str, ...]
    folds: tuple[Any, ...]
    silver_features: tuple[str, ...]
    report: dict[str, Any]


def _unredirected(path: Path, root: Path) -> Path:
    absolute = path.absolute()
    if not absolute.is_relative_to(root) or absolute.resolve() != absolute:
        raise Phase10PreflightError(f"resource escapes project or is redirected: {path}")
    return absolute


def _json_metadata(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_bounded(path, _MAX_METADATA, "silver provenance metadata")
    try:
        decoded = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise Phase10PreflightError(f"invalid silver metadata: {path}") from exc
    if not isinstance(decoded, dict):
        raise Phase10PreflightError("silver metadata must contain a JSON object")
    return decoded, hashlib.sha256(payload).hexdigest()


def validate_modeled_context_source(source: ContextSource, profile: ContextProfile) -> None:
    """The separately frozen modeled protocol cannot be weakened by source config."""
    expected = {
        "source_id": profile.source_id,
        "enabled": True,
        "availability_basis": "modeled_latency",
        "publication_delay_seconds": 60,
        "stale_after_seconds": 600,
        "revision_policy": "append_only",
        "missing_policy": "price_only_fallback",
        "source_timezone": "Fixed UTC-05:00 without daylight saving time",
        "source_url": "https://www.histdata.com/f-a-q/data-files-detailed-specification/",
    }
    for key, value in expected.items():
        if getattr(source, key) != value:
            raise Phase10PreflightError(f"frozen modeled silver source mismatch: {key}")


def inspect_context_metadata(
    root: Path,
    config: Phase10Config,
    source: ContextSource,
) -> dict[str, Any]:
    """Inspect *all* annual metadata before opening any ZIP or observation payload."""
    profile = config.profile
    validate_modeled_context_source(source, profile)
    root = root.resolve()
    set_path = _unredirected(root / config.bundle_path, root)
    archive_root = _unredirected(root / config.archive_directory, root)
    bundle_set, set_hash = _json_metadata(set_path)
    manifest = ContextBundleSetManifest.model_validate_json(json.dumps(bundle_set))
    expected_names = tuple(f"{profile.source_id}-{year}.manifest.json" for year in _YEARS)
    if (
        manifest.source_id != profile.source_id
        or manifest.availability_basis != "modeled_latency"
        or manifest.bundles != expected_names
    ):
        raise Phase10PreflightError(
            "silver bundle-set must contain exactly annual 2020-2024 bundles"
        )
    archive_metadata_path = _unredirected(set_path.parent / "source_archives.json", root)
    provenance, provenance_hash = _json_metadata(archive_metadata_path)
    expected_provenance = {
        "schema_version": 1,
        "source_id": profile.source_id,
        "source_symbol": profile.symbol,
        "availability_basis": "modeled_latency",
        "availability_is_historical_evidence": False,
        "years": list(_YEARS),
    }
    if any(
        provenance.get(key) != value or type(provenance.get(key)) is not type(value)
        for key, value in expected_provenance.items()
    ):
        raise Phase10PreflightError("silver source archive metadata exposes unexpected source/year")
    records = provenance.get("archives")
    if not isinstance(records, list) or len(records) != len(_YEARS):
        raise Phase10PreflightError("silver metadata requires five annual archive records")
    bundles: list[dict[str, Any]] = []
    for year, name, record in zip(_YEARS, expected_names, records, strict=True):
        if not isinstance(record, dict):
            raise Phase10PreflightError("silver archive record must be a JSON object")
        expected = {
            "year": year,
            "source_symbol": profile.symbol,
            "source_timezone_offset": "-05:00",
            "source_observes_dst": False,
            "timestamp_semantics": "candle_open",
            "observation_semantics": "candle_close",
            "availability_basis": "modeled_latency",
            "publication_delay_seconds": 60,
            "source_page_url": (
                "https://www.histdata.com/download-free-forex-historical-data/"
                f"?/ascii/1-minute-bar-quotes/{profile.symbol.lower()}/{year}"
            ),
        }
        if any(
            record.get(key) != value or type(record.get(key)) is not type(value)
            for key, value in expected.items()
        ):
            raise Phase10PreflightError(
                "silver annual metadata has unexpected source/year or latency"
            )
        if type(record.get("year")) is not int or type(record.get("rows")) is not int:
            raise Phase10PreflightError("silver annual year and row count must be integers")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase10PreflightError("silver archive SHA-256 is invalid")
        archive_path = _unredirected(
            archive_root / f"HISTDATA_COM_ASCII_{profile.symbol}_M1_{year}.zip",
            root,
        )
        if Path(str(record.get("path"))).absolute() != archive_path:
            raise Phase10PreflightError(
                "silver source archive path differs from configured archive"
            )
        if not archive_path.is_file():
            raise Phase10PreflightError(
                f"required local {profile.symbol} archive is missing: {archive_path}"
            )
        if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
            raise Phase10PreflightError("silver source archive size must be a positive integer")
        member_names = record.get("members")
        if not isinstance(member_names, list) or not member_names:
            raise Phase10PreflightError("silver source archive member evidence is missing")
        # Do not hash a source archive until every annual manifest passes its scope guard.
        partition_path = _unredirected(set_path.parent / name, root)
        partition, partition_hash = _json_metadata(partition_path)
        item = ContextBundleManifest.model_validate(partition)
        start = pd.Timestamp(item.observation_start_utc)
        end = pd.Timestamp(item.observation_end_utc)
        if not DEVELOPMENT_START <= start < end <= DEVELOPMENT_END:
            raise Phase10PreflightError("silver bundle span must remain in development 2020-2024")
        if (
            item.source_id != profile.source_id
            or item.availability_basis != "modeled_latency"
            or item.format != "parquet"
            or item.file != f"{profile.source_id}-{year}.parquet"
            or item.row_count <= 0
            or item.row_count != record["rows"]
        ):
            raise Phase10PreflightError("silver annual bundle identity/rows differ from provenance")
        _unredirected(set_path.parent / item.file, root)
        bundles.append(
            {
                "year": year,
                "manifest_path": str(partition_path),
                "manifest_sha256": partition_hash,
                "data_file": item.file,
                "data_sha256": item.sha256,
                "archive": record,
            }
        )
    return {
        "bundle_set": str(set_path),
        "bundle_set_sha256": set_hash,
        "source_archives": str(archive_metadata_path),
        "source_archives_sha256": provenance_hash,
        "bundles": bundles,
    }


def load_verified_context(
    root: Path,
    config: Phase10Config,
    source: ContextSource,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reparse authenticated XAGUSD ZIP bytes and require exact bundle/provenance parity."""
    profile = config.profile
    evidence = inspect_context_metadata(root, config, source)
    if metadata is not None and evidence != metadata:
        raise Phase10PreflightError("silver metadata changed after the metadata-only guard")
    frames: list[pd.DataFrame] = []
    for bundle in evidence["bundles"]:
        record = bundle["archive"]
        if profile.source_id == "silver":
            # Keep the legacy parser entry point for reproducible silver evidence.
            reparsed, actual = parse_histdata_xagusd_archive(
                record["path"],
                year=bundle["year"],
                source=source,
                ingested_at_utc=pd.Timestamp(record["ingested_at_utc"]),
            )
        else:
            from gold_forecasting.phase10.histdata_context import parse_histdata_context_archive

            reparsed, actual = parse_histdata_context_archive(
                record["path"],
                year=bundle["year"],
                source=source,
                symbol=profile.symbol,
                ingested_at_utc=pd.Timestamp(record["ingested_at_utc"]),
            )
        if json.dumps(actual, sort_keys=True) != json.dumps(record, sort_keys=True):
            raise Phase10PreflightError(
                "silver source archive hash/metadata differs from original bytes"
            )
        loaded = load_context_bundle(bundle["manifest_path"], source)
        try:
            pd.testing.assert_frame_equal(loaded, reparsed, check_exact=True)
        except AssertionError as exc:
            raise Phase10PreflightError(
                "silver bundle observations differ from authenticated XAGUSD source archive"
            ) from exc
        frames.append(loaded)
    result = validate_observations(pd.concat(frames, ignore_index=True), source)
    if inspect_context_metadata(root, config, source) != evidence:
        raise Phase10PreflightError("silver metadata changed during verification")
    return result, evidence


def augment_context_table(
    table: pd.DataFrame,
    gold_candles: pd.DataFrame,
    observations: pd.DataFrame,
    source: ContextSource,
) -> pd.DataFrame:
    """Join closed-candle gold prices and existing PIT silver features without row loss."""
    profile = profile_for_source(source.source_id)
    validate_modeled_context_source(source, profile)
    keys = ["instrument", "source", "source_candle_open_utc"]
    prices = gold_candles.rename(
        columns={
            "timestamp_open_utc": "source_candle_open_utc",
            "bid_close": "gold_close",
            "timestamp_close_utc": "gold_close_available_at_utc",
        }
    ).loc[:, [*keys, "gold_close", "gold_close_available_at_utc"]]
    augmented = table.merge(prices, on=keys, how="left", sort=False, validate="many_to_one")
    if (
        len(augmented) != len(table)
        or not augmented["sample_id"]
        .reset_index(drop=True)
        .equals(table["sample_id"].reset_index(drop=True))
        or augmented["gold_close"].isna().any()
        or augmented["gold_close_available_at_utc"].isna().any()
        or augmented["gold_close_available_at_utc"].gt(augmented["prediction_time_utc"]).any()
    ):
        raise Phase10PreflightError("gold close alignment must use the exact closed source candle")
    result = profile.build_features(join_context(augmented, observations, source))
    numeric = result.loc[:, list(profile.model_features)].to_numpy(dtype=np.float64)
    result[f"{profile.source_id}_usable"] = (
        ~result[f"{profile.source_id}_is_missing"]
        & ~result[f"{profile.source_id}_is_stale"]
        & np.isfinite(numeric).all(axis=1)
    )
    return result


def context_feature_coverage(
    frame: pd.DataFrame, *, profile: ContextProfile = SILVER_PROFILE
) -> dict[str, Any]:
    result: dict[str, Any] = dict(context_coverage(frame, profile.source_id))
    age = frame[f"{profile.source_id}_age_seconds"].dropna()
    result.update(
        {
            "context_usable_rows": int(frame[f"{profile.source_id}_usable"].sum()),
            "context_usable_fraction": float(frame[f"{profile.source_id}_usable"].mean())
            if len(frame)
            else None,
            "fallback_rows": int((~frame[f"{profile.source_id}_usable"]).sum()),
            "fallback_fraction": float((~frame[f"{profile.source_id}_usable"]).mean())
            if len(frame)
            else None,
            "insufficient_history_rows": int(
                (
                    ~frame[f"{profile.source_id}_usable"]
                    & ~frame[f"{profile.source_id}_is_missing"]
                    & ~frame[f"{profile.source_id}_is_stale"]
                ).sum()
            ),
            "age_quantiles_seconds": {
                str(quantile): float(age.quantile(quantile)) if len(age) else None
                for quantile in (0.0, 0.05, 0.5, 0.95, 1.0)
            },
        }
    )
    return result


def prepare_phase10_inputs(
    config_path: str | Path = "configs/phase10_silver_ablation.yaml",
    *,
    require_clean: bool = True,
) -> Phase10PreparedInputs:
    """Prove real-data gates; no model fitting or market benchmark occurs here."""
    path = Path(config_path).resolve(strict=True)
    config = load_phase10_config(path)
    project = load_project_config(path.parent / config.data_config)
    root = project.config_path.parent.parent.resolve()
    source = load_source(_unredirected(path.parent / config.source_config, root))
    profile = config.profile
    source_metadata = inspect_context_metadata(root, config, source)
    preflight_development_inputs(project)
    code_version = get_git_code_version(root)
    if require_clean and (
        code_version in {"unavailable", "uncommitted"} or code_version.endswith("+dirty")
    ):
        raise Phase10PreflightError(
            "real-data preflight requires a clean committed Git working tree"
        )
    references = load_phase10_reference(root, path.parent / config.phase7_champion_config)
    observations, source_evidence = load_verified_context(
        root,
        config,
        source,
        metadata=source_metadata,
    )
    validate_existing_mvp_data(project.config_path)
    candles: dict[str, pd.DataFrame] = {}
    manifests: dict[str, Any] = {}
    frozen_manifests = references.run.read_json("source_manifests.json")
    for timeframe in _TIMEFRAMES:
        candles[timeframe], manifests[timeframe] = _load_curated(root, project, timeframe)
        expected = frozen_manifests.get(timeframe)
        if not isinstance(expected, dict) or (
            expected.get("dataset_version") != manifests[timeframe].dataset_version
        ):
            raise Phase10PreflightError(
                f"gold data differs from frozen Phase-7 source: {timeframe}"
            )
    feature_config = load_phase7_feature_config(path.parent / config.features_config)
    if feature_config.model_dump(mode="json") != references.run.read_json(
        "resolved_feature_config.json"
    ):
        raise Phase10PreflightError("price feature configuration differs from frozen Phase-7")
    built = build_phase7_features(candles, project.features, feature_config)
    labels = build_horizon_labels(candles["1min"], built.features, horizon_minutes=15)
    table = built.features.merge(
        labels.labels,
        on=["instrument", "source", "prediction_time_utc"],
        validate="one_to_one",
    )
    table["sample_id"] = [
        hashlib.sha256(f"{instrument}|{provider}|{stamp.isoformat()}|15".encode()).hexdigest()
        for instrument, provider, stamp in table[
            ["instrument", "source", "prediction_time_utc"]
        ].itertuples(index=False, name=None)
    ]
    feature_columns = tuple(built.catalog.variants["mvp"])
    parity = verify_phase10_universe(references, table, feature_columns)
    folds = make_walk_forward_folds(test_years=config.test_years)
    coverage = _validate_fold_coverage(table, folds, feature_columns)
    table = augment_context_table(table, candles["3min"], observations, source)
    fold_context: dict[str, Any] = {}
    for fold in folds:
        blocks = {
            "train": select_block(table, fold.train, gap_minutes=181),
            "calibration": select_block(table, fold.calibration, purge=False),
            "test": select_block(table, fold.test, purge=False),
        }
        fold_context[fold.name] = {
            name: context_feature_coverage(block, profile=profile) for name, block in blocks.items()
        }
    versions = {name: manifest.dataset_version for name, manifest in manifests.items()}
    report: dict[str, Any] = {
        "protocol": config.protocol_version,
        "status": "passed",
        "exploratory_ablation_ready": True,
        "formal_benchmark_ready": False,
        "strict_pit_source_ready": False,
        "formal_run_opened": False,
        "holdout_opened": False,
        "champion_changed": False,
        "trading_activated": False,
        "code_version": code_version,
        "test_years": list(config.test_years),
        "gap_minutes": 181,
        "phase7_reference": {
            "run_id": config.phase7_reference_run,
            "code_version": config.phase7_reference_code,
            "completion_version": config.phase7_reference_completion,
            "horizon_minutes": 15,
            "variant": "mvp",
            "family": "logistic",
        },
        "curated_versions": versions,
        "source": source.model_dump(mode="json"),
        f"{profile.source_id}_evidence": source_evidence,
        "reference_parity": parity,
        "common_sample_digest": sample_id_digest(table["sample_id"]),
        "common_rows": len(table),
        "folds": coverage,
        "context_coverage": context_feature_coverage(table, profile=profile),
        "fold_context": fold_context,
        "config_sha256": sha256_file(path),
        "source_config_sha256": sha256_file(path.parent / config.source_config),
        "data_version": content_version({"gold": versions, profile.source_id: source_evidence}),
    }
    return Phase10PreparedInputs(
        root,
        config,
        project,
        source,
        references,
        table,
        feature_columns,
        folds,
        profile.model_features,
        report,
    )


def run_phase10_preflight(
    config_path: str | Path = "configs/phase10_silver_ablation.yaml",
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    prepared = prepare_phase10_inputs(config_path)
    if report_path is not None:
        destination = Path(report_path)
        if not destination.is_absolute():
            destination = prepared.root / destination
        destination = _unredirected(destination, prepared.root)
        if (
            not destination.is_relative_to(prepared.root / "reports")
            or destination.suffix != ".json"
        ):
            raise Phase10PreflightError("preflight report must be a JSON file under reports/")
        write_json_atomic(destination, prepared.report)
    return prepared.report


__all__ = [
    "SILVER_MODEL_FEATURES",
    "Phase10PreflightError",
    "Phase10PreparedInputs",
    "augment_silver_table",
    "prepare_phase10_inputs",
    "run_phase10_preflight",
]


# Existing silver consumers retain their entry points and serialized evidence keys.
def validate_modeled_silver_source(source: ContextSource) -> None:
    validate_modeled_context_source(source, SILVER_PROFILE)


inspect_silver_metadata = inspect_context_metadata
load_verified_silver = load_verified_context
augment_silver_table = augment_context_table
silver_coverage = context_feature_coverage
