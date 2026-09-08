"""Cryptographically guarded loaders for frozen Phase-7/8 Phase-9 references."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from gold_forecasting.artifacts import content_version, sha256_file
from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.config import _load_yaml_mapping
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.phase8.reference import Phase7ChampionConfig, ResearchChampion
from gold_forecasting.phase9.config import Phase9Config

_REFERENCE_HORIZON = 15


class Phase9ReferenceError(ValueError):
    """Raised when a frozen Phase-7/8 prediction reference cannot be proven intact."""


@dataclass(frozen=True, slots=True)
class VerifiedReferenceRun:
    """Manifest-authenticated view of one immutable benchmark run."""

    root: Path
    run_id: str
    protocol: str
    code_version: str
    completion_version: str
    summary: dict[str, Any]
    completion: dict[str, Any]
    files: dict[str, dict[str, Any]]

    def verified_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative_path == "":
            raise Phase9ReferenceError("reference artifact path must stay inside its run")
        normalized = relative.as_posix()
        entry = self.files.get(normalized)
        if entry is None:
            raise Phase9ReferenceError(
                f"completion manifest does not cover required artifact: {normalized}"
            )
        try:
            target = (self.root / relative).resolve(strict=True)
        except FileNotFoundError as exc:
            raise Phase9ReferenceError(
                f"required frozen reference artifact is missing: {normalized}"
            ) from exc
        if not target.is_relative_to(self.root) or not target.is_file():
            raise Phase9ReferenceError(
                f"reference artifact escapes or is not a file: {normalized}"
            )
        if (
            entry.get("size_bytes") != target.stat().st_size
            or entry.get("sha256") != sha256_file(target)
        ):
            raise Phase9ReferenceError(
                f"frozen reference artifact digest mismatch: {normalized}"
            )
        return target

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.verified_path(relative_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase9ReferenceError(
                f"invalid frozen reference JSON: {relative_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise Phase9ReferenceError(
                f"frozen reference JSON must contain an object: {relative_path}"
            )
        return cast(dict[str, Any], payload)

    def read_parquet(self, relative_path: str) -> pd.DataFrame:
        path = self.verified_path(relative_path)
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise Phase9ReferenceError(
                f"cannot read frozen reference parquet: {relative_path}"
            ) from exc
        if not isinstance(frame, pd.DataFrame):
            raise Phase9ReferenceError(
                f"frozen reference parquet is not a DataFrame: {relative_path}"
            )
        return frame


@dataclass(frozen=True, slots=True)
class FrozenReferenceFold:
    """One original frozen outer fold plus its already-selected decision policy."""

    source: str
    fold: str
    records: pd.DataFrame
    policy: DecisionPolicy
    original_sample_digest: str


@dataclass(frozen=True, slots=True)
class Phase9ReferenceBundle:
    """Frozen Phase-7 classical and Phase-8 neural references for Phase 9."""

    phase7: VerifiedReferenceRun
    phase8: VerifiedReferenceRun
    phase7_champions: Phase7ChampionConfig
    test_years: tuple[int, ...]

    def _validate_fold(self, fold_name: str) -> None:
        allowed = {f"test_{year}" for year in self.test_years}
        if fold_name not in allowed:
            raise Phase9ReferenceError(
                f"unexpected Phase-9 reference fold {fold_name!r}; expected {sorted(allowed)}"
            )

    def phase7_fold(self, fold_name: str) -> FrozenReferenceFold:
        self._validate_fold(fold_name)
        champion = self._phase7_15m_champion()
        base = f"horizon_{_REFERENCE_HORIZON}/{champion.variant}/{fold_name}"
        fold_summary = self.phase7.read_json(f"{base}/fold_summary.json")
        split = fold_summary.get("split")
        if not isinstance(split, dict):
            raise Phase9ReferenceError(
                f"Phase-7 fold summary has no split record: {fold_name}"
            )
        expected_digest = _require_digest(
            split.get("test_digest"),
            context=f"Phase-7 {fold_name} test digest",
        )
        models = fold_summary.get("models")
        if not isinstance(models, dict):
            raise Phase9ReferenceError(
                f"Phase-7 fold summary has no model mapping: {fold_name}"
            )
        model = models.get(champion.family)
        if not isinstance(model, dict):
            raise Phase9ReferenceError(
                f"Phase-7 frozen champion is absent from fold summary: {fold_name}"
            )
        model_digest = _require_digest(
            model.get("sample_digest"),
            context=f"Phase-7 {fold_name} model digest",
        )
        if model_digest != expected_digest:
            raise Phase9ReferenceError(
                f"Phase-7 fold/model sample digest mismatch: {fold_name}"
            )
        family_root = f"{base}/{champion.family}"
        selection = self.phase7.read_json(f"{family_root}/selection.json")
        policy = _policy_from_payload(
            selection.get("selected_policy"),
            context=f"Phase-7 {fold_name} selected policy",
        )
        records = self.phase7.read_parquet(
            f"{family_root}/outer_predictions.parquet"
        )
        _validate_outer_records(
            records,
            fold_name=fold_name,
            expected_digest=expected_digest,
            source="Phase-7",
        )
        return FrozenReferenceFold(
            source="phase7",
            fold=fold_name,
            records=records,
            policy=policy,
            original_sample_digest=expected_digest,
        )

    def phase8_fold(self, fold_name: str) -> FrozenReferenceFold:
        self._validate_fold(fold_name)
        base = f"horizon_{_REFERENCE_HORIZON}/{fold_name}"
        audit = self.phase8.read_json(f"{base}/training_audit.json")
        expected_digest = _require_digest(
            audit.get("test_digest"),
            context=f"Phase-8 {fold_name} test digest",
        )
        policy = _policy_from_payload(
            audit.get("selected_policy"),
            context=f"Phase-8 {fold_name} selected policy",
        )
        records = self.phase8.read_parquet(
            f"{base}/ensemble/outer_predictions.parquet"
        )
        _validate_outer_records(
            records,
            fold_name=fold_name,
            expected_digest=expected_digest,
            source="Phase-8",
        )
        return FrozenReferenceFold(
            source="phase8",
            fold=fold_name,
            records=records,
            policy=policy,
            original_sample_digest=expected_digest,
        )

    def _phase7_15m_champion(self) -> ResearchChampion:
        try:
            champion = self.phase7_champions.research_champions[
                _REFERENCE_HORIZON
            ]
        except KeyError as exc:
            raise Phase9ReferenceError(
                "frozen Phase-7 config has no 15-minute research champion"
            ) from exc
        if champion.variant != "mvp" or champion.family != "logistic":
            raise Phase9ReferenceError(
                "Phase-9 requires the frozen Phase-7 15m mvp/logistic champion"
            )
        return champion


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase9ReferenceError(f"cannot read {context}") from exc
    if not isinstance(payload, dict):
        raise Phase9ReferenceError(f"{context} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _manifest_index(completion: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = completion.get("files")
    if not isinstance(entries, list):
        raise Phase9ReferenceError("reference completion manifest has no file list")
    result: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise Phase9ReferenceError("reference completion manifest has an invalid entry")
        entry = cast(dict[str, Any], raw_entry)
        relative = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise Phase9ReferenceError("reference completion entry has an invalid path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise Phase9ReferenceError("reference completion entry escapes its run")
        if relative in result:
            raise Phase9ReferenceError(
                f"reference completion manifest repeats path: {relative}"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Phase9ReferenceError(
                f"reference completion entry has invalid size: {relative}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase9ReferenceError(
                f"reference completion entry has invalid sha256: {relative}"
            )
        result[relative] = entry
    return result


def _verified_manifest_file(
    root: Path,
    files: dict[str, dict[str, Any]],
    relative_path: str,
) -> Path:
    entry = files.get(relative_path)
    if entry is None:
        raise Phase9ReferenceError(
            f"completion manifest does not cover required artifact: {relative_path}"
        )
    try:
        path = (root / relative_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise Phase9ReferenceError(
            f"required frozen reference artifact is missing: {relative_path}"
        ) from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise Phase9ReferenceError(
            f"reference artifact escapes or is not a file: {relative_path}"
        )
    if (
        entry.get("size_bytes") != path.stat().st_size
        or entry.get("sha256") != sha256_file(path)
    ):
        raise Phase9ReferenceError(
            f"frozen reference artifact digest mismatch: {relative_path}"
        )
    return path


def _load_verified_run(
    project_root: Path,
    *,
    phase_directory: str,
    expected_run_id: str,
    expected_protocol: str,
    expected_code_version: str,
    expected_completion_version: str,
) -> VerifiedReferenceRun:
    expected_parent = (project_root / "reports" / phase_directory).resolve()
    try:
        reference = (expected_parent / expected_run_id).resolve(strict=True)
    except FileNotFoundError as exc:
        raise Phase9ReferenceError(
            f"required frozen reference run is missing: {phase_directory}/{expected_run_id}"
        ) from exc
    if not reference.is_relative_to(expected_parent) or reference.name != expected_run_id:
        raise Phase9ReferenceError("frozen reference run escapes its expected reports root")
    run = _read_json_object(reference / "run.json", context="reference run.json")
    completion = _read_json_object(
        reference / "completion.json",
        context="reference completion.json",
    )
    if (
        run.get("status") != "succeeded"
        or run.get("run_id") != expected_run_id
        or run.get("code_version") != expected_code_version
    ):
        raise Phase9ReferenceError(
            f"{phase_directory} run identity does not match the frozen reference"
        )
    files = _manifest_index(completion)
    actual_completion_version = completion.get("version")
    if actual_completion_version != content_version({"files": completion.get("files", [])}):
        raise Phase9ReferenceError(
            f"{phase_directory} completion manifest version is internally inconsistent"
        )
    if actual_completion_version != expected_completion_version:
        raise Phase9ReferenceError(
            f"{phase_directory} completion version differs from the frozen Phase-9 pin"
        )
    summary_path = _verified_manifest_file(reference, files, "summary.json")
    summary = _read_json_object(summary_path, context=f"{phase_directory} summary.json")
    if (
        summary.get("protocol") != expected_protocol
        or summary.get("holdout_opened") is not False
    ):
        raise Phase9ReferenceError(
            f"{phase_directory} summary violates the frozen protocol/holdout contract"
        )
    return VerifiedReferenceRun(
        root=reference,
        run_id=expected_run_id,
        protocol=expected_protocol,
        code_version=expected_code_version,
        completion_version=expected_completion_version,
        summary=summary,
        completion=completion,
        files=files,
    )


def _finite_policy_value(
    value: object,
    *,
    context: str,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Phase9ReferenceError(
            f"{context} has invalid {name}"
        )
    return float(value)


def _policy_from_payload(payload: object, *, context: str) -> DecisionPolicy:
    if not isinstance(payload, dict):
        raise Phase9ReferenceError(f"{context} is missing")
    return DecisionPolicy(
        confidence_threshold=_finite_policy_value(
            payload.get("confidence_threshold"),
            context=context,
            name="confidence_threshold",
        ),
        min_expected_net_bps=_finite_policy_value(
            payload.get("min_expected_net_bps"),
            context=context,
            name="min_expected_net_bps",
        ),
    )


def _require_digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Phase9ReferenceError(f"{context} is invalid")
    hexadecimal = value.removeprefix("sha256:")
    if (
        len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise Phase9ReferenceError(f"{context} is invalid")
    return value


def _validate_outer_records(
    records: pd.DataFrame,
    *,
    fold_name: str,
    expected_digest: str,
    source: str,
) -> None:
    required = {"sample_id", "fold", "horizon_minutes"}
    missing = required.difference(records.columns)
    if missing:
        raise Phase9ReferenceError(
            f"{source} {fold_name} outer predictions miss columns: {sorted(missing)}"
        )
    if records.empty or records["sample_id"].duplicated().any():
        raise Phase9ReferenceError(
            f"{source} {fold_name} outer predictions require unique nonempty sample ids"
        )
    if not records["horizon_minutes"].eq(_REFERENCE_HORIZON).all():
        raise Phase9ReferenceError(
            f"{source} {fold_name} outer predictions contain the wrong horizon"
        )
    if not records["fold"].eq(fold_name).all():
        raise Phase9ReferenceError(
            f"{source} {fold_name} outer predictions contain the wrong fold identity"
        )
    actual_digest = sample_id_digest(records["sample_id"])
    if actual_digest != expected_digest:
        raise Phase9ReferenceError(
            f"{source} {fold_name} outer prediction sample digest mismatch"
        )


def load_phase9_reference_bundle(
    project_root: str | Path,
    config: Phase9Config,
    champion_config: str | Path,
) -> Phase9ReferenceBundle:
    """Load only the canonical Phase-7/8 runs pinned by the Phase-9 protocol."""

    root = Path(project_root).resolve(strict=True)
    champions = Phase7ChampionConfig.model_validate(
        _load_yaml_mapping(Path(champion_config))
    )
    if (
        champions.protocol != "phase7-v2"
        or champions.release != "v0.2.0"
        or champions.run_id != config.phase7_reference_run
        or champions.code_version != config.phase7_reference_code
        or champions.holdout_opened
        or champions.trading_champion is not None
    ):
        raise Phase9ReferenceError(
            "Phase-7 champion config differs from the frozen Phase-9 reference contract"
        )
    if _REFERENCE_HORIZON not in champions.research_champions:
        raise Phase9ReferenceError("Phase-7 champion config has no 15-minute champion")

    phase7 = _load_verified_run(
        root,
        phase_directory="phase7_runs",
        expected_run_id=config.phase7_reference_run,
        expected_protocol="phase7-v2",
        expected_code_version=config.phase7_reference_code,
        expected_completion_version=config.phase7_reference_completion,
    )
    phase8 = _load_verified_run(
        root,
        phase_directory="phase8_runs",
        expected_run_id=config.phase8_reference_run,
        expected_protocol="phase8-v1",
        expected_code_version=config.phase8_reference_code,
        expected_completion_version=config.phase8_reference_completion,
    )
    if phase8.summary.get("phase7_reference_run") != phase7.run_id:
        raise Phase9ReferenceError(
            "frozen Phase-8 run does not point to the pinned Phase-7 reference"
        )
    if tuple(phase8.summary.get("test_years", ())) != config.test_years:
        raise Phase9ReferenceError(
            "frozen Phase-8 run uses different outer test years"
        )
    return Phase9ReferenceBundle(
        phase7=phase7,
        phase8=phase8,
        phase7_champions=champions,
        test_years=tuple(config.test_years),
    )


__all__ = [
    "FrozenReferenceFold",
    "Phase9ReferenceBundle",
    "Phase9ReferenceError",
    "VerifiedReferenceRun",
    "load_phase9_reference_bundle",
]
